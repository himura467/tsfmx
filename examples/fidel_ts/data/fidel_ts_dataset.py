"""Fidel-TS dataset loader with as-of text retrieval."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from tsfmx.data.dataset import MultimodalDatasetBase
from tsfmx.types import RawSample
from tsfmx.utils.logging import get_logger

_logger = get_logger()

#: Format of the timestamp keys in every Fidel-TS heterogeneous report.
_REPORT_KEY_FORMAT = "%Y%m%d%H%M"

Split = Literal["train", "val", "test", "all"]

SPLITS: tuple[Split, ...] = ("train", "val", "test", "all")


@dataclass(frozen=True)
class TextSource:
    """One stream of timestamped text to retrieve alongside the series.

    Attributes:
        name: Label prefixed to each retrieved sentence, so the model can tell the streams apart.
        path: JSON file mapping timestamp keys to a dict of field name to sentence. A path inside
            a zip archive is written as `archive.zip/member/path.json`.
        per_entity: Whether `path` contains an `{entity}` placeholder to fill in per series.
    """

    name: str
    path: str
    per_entity: bool = False


class FidelTsDataset(MultimodalDatasetBase):
    """Sliding-window samples over one Fidel-TS series, with text retrieved as of the prediction time.

    Text is retrieved by the time it *became available*, not by the period it describes. Every
    report legitimately usable at prediction time was issued at or before it, so each one lands on
    a context patch and the additive fusion needs no horizon-side slot. That is what lets a
    forecast-bearing report — "the weather is expected to remain overcast" — reach the model.

    Reports are sampled far more coarsely than the series (weather every 6 hours against 5-minute
    readings), so a patch rarely contains one. Each patch therefore takes the most recent report at
    or before its final timestamp: the statement in force over that patch, which is also what a
    forecaster would have had in hand.

    Static text (`general_info`, `channel_info`) is deliberately left out. It is constant per
    series, and a constant is exactly what the fusion branch degenerates into when the text carries
    nothing else; including it would make that degeneracy indistinguishable from success.

    Args:
        data_dir: Root of one downloaded sub-dataset, e.g. `data/Fidel-TS/Bear_room`.
        entity: Series identifier, matching a file in `time_series/` (e.g. '104').
        target_column: Column of the parquet file to forecast.
        text_sources: Text streams to retrieve.
        patch_len: Length of input patches, which the text patches align to.
        context_len: Length of context. Must be an integer multiple of patch_len.
        horizon_len: Length of horizon. Must be an integer multiple of patch_len.
        timestamp_column: Column holding the timestamps.
        augment: If True, generate one sample set per shift in range(patch_len).
        split: Which contiguous part of the series to use. The series is cut before any window
            is formed, so no window straddles a split boundary.
        train_ratio: Fraction of the series given to 'train'.
        val_ratio: Fraction given to 'val'; 'test' takes the remainder.

    Raises:
        FileNotFoundError: If data_dir or the entity's time series file does not exist.
        ValueError: If context_len or horizon_len is not a multiple of patch_len, if split is
            unknown, if the split ratios do not leave a test split, or if target_column or
            timestamp_column is missing from the time series file.
    """

    def __init__(
        self,
        data_dir: Path,
        entity: str,
        target_column: str,
        text_sources: list[TextSource],
        patch_len: int = 16,
        context_len: int = 32,
        horizon_len: int = 32,
        timestamp_column: str = "Timestamp",
        augment: bool = False,
        split: Split = "all",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.entity = entity
        self.target_column = target_column
        self.text_sources = text_sources
        self.patch_len = patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.timestamp_column = timestamp_column
        self.augment = augment
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.data: list[RawSample] = []

        self._validate()
        self._load_data()

    def _validate(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Fidel-TS sub-dataset not found: {self.data_dir}")
        if self.context_len % self.patch_len != 0:
            raise ValueError(f"context_len ({self.context_len}) must be a multiple of patch_len ({self.patch_len})")
        if self.horizon_len % self.patch_len != 0:
            raise ValueError(f"horizon_len ({self.horizon_len}) must be a multiple of patch_len ({self.patch_len})")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split!r}. Expected one of {list(SPLITS)}")
        if not 0.0 < self.train_ratio + self.val_ratio < 1.0:
            raise ValueError(f"train_ratio + val_ratio must lie in (0, 1), got {self.train_ratio} + {self.val_ratio}")

    def _split_bounds(self, length: int) -> tuple[int, int]:
        """Return the half-open index range of the configured split.

        The series is cut before any window is formed, so no window straddles a boundary and no
        training window can see a value a later split is evaluated on.

        Args:
            length: Number of points in the series.

        Returns:
            Tuple of (start, end) indices into the series.
        """
        train_end = int(length * self.train_ratio)
        val_end = train_end + int(length * self.val_ratio)
        match self.split:
            case "train":
                return 0, train_end
            case "val":
                return train_end, val_end
            case "test":
                return val_end, length
            case _:
                return 0, length

    @property
    def _time_series_path(self) -> Path:
        return self.data_dir / "time_series" / f"{self.entity}.parquet"

    @staticmethod
    def list_entities(data_dir: Path) -> list[str]:
        """Return the entity ids that have a time series file, sorted.

        Args:
            data_dir: Root of one downloaded sub-dataset.

        Returns:
            Entity ids taken from the parquet filenames.
        """
        return sorted(p.stem for p in (Path(data_dir) / "time_series").glob("*.parquet"))

    def _read_report(self, path: str) -> dict[str, dict[str, str]]:
        """Read one timestamped report, from a plain file or from inside a zip archive.

        The room reports ship as a single archive that expands to several gigabytes, so members
        are read out of it directly rather than extracted.

        Args:
            path: Path relative to data_dir, optionally naming a member inside a `.zip` component.

        Returns:
            Mapping from timestamp key to a dict of field name to sentence.

        Raises:
            FileNotFoundError: If the file, or the named member of the archive, is absent.
        """
        parts = Path(path).parts
        zip_index = next((i for i, part in enumerate(parts) if part.endswith(".zip")), None)
        if zip_index is None:
            full_path = self.data_dir / path
            if not full_path.exists():
                raise FileNotFoundError(f"Text source not found: {full_path}")
            with open(full_path) as f:
                report: dict[str, dict[str, str]] = json.load(f)
            return report

        archive_path = self.data_dir / Path(*parts[: zip_index + 1])
        member = "/".join(parts[zip_index + 1 :])
        if not archive_path.exists():
            raise FileNotFoundError(f"Text source archive not found: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            if member not in archive.namelist():
                raise FileNotFoundError(f"{member!r} not found in {archive_path}")
            with archive.open(member) as f:
                return json.load(f)

    def _load_text_source(self, source: TextSource) -> tuple[npt.NDArray[np.int64], list[str]]:
        """Load one text stream as timestamps paired with rendered sentences.

        Args:
            source: Stream to load.

        Returns:
            Tuple of (sorted timestamps as integer nanoseconds, rendered text per timestamp).
        """
        path = source.path.format(entity=self.entity) if source.per_entity else source.path
        report = self._read_report(path)

        keys = pd.to_datetime(list(report), format=_REPORT_KEY_FORMAT)
        order = np.argsort(keys.values)
        timestamps = keys.values[order].astype("datetime64[ns]").astype(np.int64)
        raw = [report[key] for key in np.asarray(list(report))[order]]
        texts = [
            f"{source.name}: " + " ".join(str(v).strip() for v in fields.values() if str(v).strip()) for fields in raw
        ]
        _logger.info("Loaded %d %s reports for entity %s", len(texts), source.name, self.entity)
        return timestamps, texts

    def _patched_texts(
        self,
        patch_end_times: npt.NDArray[np.int64],
        sources: list[tuple[TextSource, npt.NDArray[np.int64], list[str]]],
    ) -> list[list[str]]:
        """Collect, for each context patch, the report in force at the end of that patch.

        Args:
            patch_end_times: Final timestamp of each context patch, as integer nanoseconds.
            sources: Loaded text streams as (source, timestamps, texts).

        Returns:
            One list of sentences per context patch. A patch is empty only when every stream
            begins after it.
        """
        patches: list[list[str]] = [[] for _ in patch_end_times]
        for _, timestamps, texts in sources:
            # searchsorted with 'right' gives the count of reports at or before the patch end, so
            # subtracting one indexes the most recent of them. -1 means the stream had not started.
            positions = np.searchsorted(timestamps, patch_end_times, side="right") - 1
            for patch, position in zip(patches, positions):
                if position >= 0:
                    patch.append(texts[position])
        return patches

    def _normalize_sample(
        self, context: npt.NDArray[np.float64], horizon: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Z-score both windows using the context statistics, as the Time-MMD loader does.

        Args:
            context: Context values of shape (context_len,).
            horizon: Horizon values of shape (horizon_len,).

        Returns:
            Tuple of (normalized context, normalized horizon, context mean, context std).
        """
        context_mean = float(np.mean(context))
        context_std = float(np.std(context))
        if context_std < 1e-6:
            context_std = 1.0
        return (context - context_mean) / context_std, (horizon - context_mean) / context_std, context_mean, context_std

    def _load_data(self) -> None:
        """Build every sliding-window sample for this entity.

        Raises:
            FileNotFoundError: If the entity's time series file is absent.
            ValueError: If the configured timestamp or target column is missing.
        """
        if not self._time_series_path.exists():
            raise FileNotFoundError(f"Time series not found: {self._time_series_path}")

        frame = pd.read_parquet(self._time_series_path)
        for column in (self.timestamp_column, self.target_column):
            if column not in frame.columns:
                raise ValueError(f"Column {column!r} not in {self._time_series_path} (has {list(frame.columns)})")

        times: npt.NDArray[np.int64] = np.asarray(
            pd.to_datetime(frame[self.timestamp_column]).to_numpy(), dtype="datetime64[ns]"
        ).astype(np.int64)
        values = frame[self.target_column].to_numpy(dtype=np.float64)

        split_start, split_end = self._split_bounds(len(values))
        times, values = times[split_start:split_end], values[split_start:split_end]
        sources = [(source, *self._load_text_source(source)) for source in self.text_sources]

        window = self.context_len + self.horizon_len
        if len(values) < window:
            _logger.warning(
                "Entity %s split %s has %d points, fewer than the %d-step window",
                self.entity,
                self.split,
                len(values),
                window,
            )
            return

        shifts = range(self.patch_len) if self.augment else range(1)
        for shift in shifts:
            for start in range(shift, len(values) - window + 1, self.horizon_len):
                context_end = start + self.context_len
                context, horizon = values[start:context_end], values[context_end : context_end + self.horizon_len]
                context_normalized, horizon_normalized, mean, std = self._normalize_sample(context, horizon)

                patch_end_indices = np.arange(start + self.patch_len - 1, context_end, self.patch_len)
                self.data.append(
                    RawSample(
                        context=context_normalized.astype(np.float32),
                        horizon=horizon_normalized.astype(np.float32),
                        patched_texts=self._patched_texts(times[patch_end_indices], sources),
                        metadata={
                            "entity": self.entity,
                            "split": self.split,
                            "column": self.target_column,
                            "shift": shift,
                            "start_index": start,
                            "prediction_time": str(pd.Timestamp(times[context_end - 1])),
                            "mean": mean,
                            "std": std,
                        },
                    )
                )
        _logger.info("Built %d samples for entity %s split %s", len(self.data), self.entity, self.split)

    @override
    def __getitem__(self, index: int) -> RawSample:
        return self.data[index]

    @override
    def __len__(self) -> int:
        return len(self.data)

    def describe(self) -> dict[str, Any]:
        """Return a summary of what was loaded, for logging and cache provenance."""
        return {
            "entity": self.entity,
            "split": self.split,
            "target_column": self.target_column,
            "num_samples": len(self.data),
            "text_sources": [source.name for source in self.text_sources],
        }
