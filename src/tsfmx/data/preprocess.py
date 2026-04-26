"""Preprocessing pipeline for multimodal and baseline datasets."""

import pickle
from pathlib import Path
from typing import Callable

import torch

from tsfmx.data.dataset import MultimodalDatasetBase
from tsfmx.text_encoder.base import TextEncoderBase
from tsfmx.types import PreprocessedSample
from tsfmx.utils.logging import get_logger

_logger = get_logger()


class PreprocessPipeline:
    """End-to-end preprocessing pipeline: path generation, persistence, and execution."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(
        self,
        dataset_name: str,
        entity: str,
        text_encoder_type: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        augment: bool = False,
    ) -> Path:
        """Generate a unique file path for the given configuration.

        Args:
            dataset_name: Name of the dataset (e.g. 'time_mmd').
            entity: Domain or split-domain identifier (e.g. 'Environment_train').
            text_encoder_type: Text encoder type string.
            patch_len: Patch length.
            context_len: Context length.
            horizon_len: Horizon length.
            augment: Whether patch-boundary shift augmentation was applied.

        Returns:
            Path to the cache file.
        """
        parts = [
            dataset_name,
            entity,
            text_encoder_type,
            f"p{patch_len}",
            f"c{context_len}",
            f"h{horizon_len}",
        ]
        if augment:
            parts.append("aug")
        return self.cache_dir / ("_".join(parts) + ".pkl")

    def load(self, path: Path) -> list[PreprocessedSample]:
        _logger.info("Loading preprocessed data from %s", path)
        with open(path, "rb") as f:
            data: list[PreprocessedSample] = pickle.load(f)
        _logger.info("Loaded %s samples", len(data))
        return data

    def _save(self, path: Path, data: list[PreprocessedSample]) -> None:
        _logger.info("Saving %s samples to %s", len(data), path)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024 * 1024)
        _logger.info("Saved %.2f MB", size_mb)

    def _preprocess(
        self,
        dataset: MultimodalDatasetBase,
        text_encoder: TextEncoderBase | None,
        device: torch.device | None,
    ) -> list[PreprocessedSample]:
        _logger.info(
            "Preprocessing %s samples (%s)",
            len(dataset),
            "with text" if text_encoder is not None else "without text",
        )
        if text_encoder is not None:
            if device is None:
                raise ValueError("device must be provided when text_encoder is specified")
            text_encoder.eval()
        result: list[PreprocessedSample] = []
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                entry = PreprocessedSample(
                    context=sample["context"],
                    horizon=sample["horizon"],
                    metadata=sample["metadata"],
                )
                if text_encoder is not None and device is not None:
                    texts = [" ".join(patch) if patch else "" for patch in sample["patched_texts"]]
                    embeddings = text_encoder(texts)
                    entry["text_embeddings"] = embeddings.cpu().numpy()
                result.append(entry)
                if (i + 1) % 100 == 0:
                    _logger.info("Preprocessed %s/%s samples", i + 1, len(dataset))
        _logger.info("Preprocessing complete")
        return result

    def prepare(
        self,
        path: Path,
        dataset_factory: Callable[[], MultimodalDatasetBase],
        text_encoder: TextEncoderBase | None = None,
        device: torch.device | None = None,
        force_rebuild: bool = False,
    ) -> list[PreprocessedSample]:
        """Load preprocessed data from disk, or create and save it if absent.

        Args:
            path: Target file path.
            dataset_factory: Callable that returns a raw MultimodalDatasetBase.
            text_encoder: Text encoder for fusion/finetune modes. None for adapter mode.
            device: Required when text_encoder is provided.
            force_rebuild: Rebuild even if file exists.

        Returns:
            List of preprocessed samples.
        """
        if not force_rebuild and path.exists():
            return self.load(path)

        dataset = dataset_factory()
        data = self._preprocess(dataset, text_encoder, device)
        self._save(path, data)
        return data
