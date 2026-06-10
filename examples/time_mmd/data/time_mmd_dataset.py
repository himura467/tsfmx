"""Time-MMD dataset loader for multimodal time series forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import override

from examples.time_mmd.configs.domain_columns import DEFAULT_TIME_MMD_CONFIGS, DomainColumnConfig
from tsfmx.data.dataset import MultimodalDatasetBase
from tsfmx.types import RawSample


class TimeMmdDataset(MultimodalDatasetBase):
    """Dataset loader for Time-MMD dataset with time series and text data.

    This class loads multimodal time series data from the Time-MMD dataset structure,
    which contains numerical time series data in domain-specific CSV files and
    corresponding textual information (reports and search data) for each domain.

    Expected directory structure:
        data_dir/
        ├── numerical/
        │   └── (Domain)/
        │       └── (Domain).csv
        └── textual/
            └── (Domain)/
                ├── (Domain)_report.csv
                └── (Domain)_search.csv
    """

    def __init__(
        self,
        data_dir: Path,
        domain: str,
        patch_len: int = 32,
        context_len: int = 32,
        horizon_len: int = 32,
        column_config: DomainColumnConfig | None = None,
        augment: bool = False,
    ) -> None:
        """Initializes Time-MMD dataset loader.

        Args:
            data_dir: Root directory containing Time-MMD dataset.
            domain: Domain name (e.g., 'Agriculture').
            patch_len: Length of input patches for temporal alignment with time series data.
            context_len: Length of context for input sequences.
                context_len must be an integer multiple of patch_len.
            horizon_len: Length of horizon.
                horizon_len must be an integer multiple of patch_len.
            column_config: Optional column configuration for this domain.
                If None, uses the default configuration from DEFAULT_TIME_MMD_CONFIGS.
            augment: If True, generate one sample set per shift in range(patch_len),
                increasing dataset size by up to patch_len times.
        """
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.patch_len = patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.column_config = column_config or DEFAULT_TIME_MMD_CONFIGS.get_config_for_domain(domain)
        self.augment = augment
        self.data: list[RawSample] = []

        self._validate()

        self._load_data()

    def _validate(self) -> None:
        """Validates dataset configuration parameters.

        Raises:
            FileNotFoundError: If data_dir does not exist.
            ValueError: If context_len or horizon_len is not an integer multiple of patch_len.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if self.context_len % self.patch_len != 0:
            raise ValueError(
                f"context_len ({self.context_len}) must be an integer multiple of patch_len ({self.patch_len})"
            )
        if self.horizon_len % self.patch_len != 0:
            raise ValueError(
                f"horizon_len ({self.horizon_len}) must be an integer multiple of patch_len ({self.patch_len})"
            )

    def _sanitize_time_series(
        self, time_series_values: np.ndarray, start_dates: pd.Series, end_dates: pd.Series
    ) -> tuple[np.ndarray, pd.Series, pd.Series] | None:
        """Sanitize time series by removing leading/trailing invalid values and interpolating all invalid values.

        Args:
            time_series_values: Raw time series values from the dataset.
            start_dates: Series of start dates corresponding to each time series value.
            end_dates: Series of end dates corresponding to each time series value.

        Returns:
            Tuple of (sanitized_values, trimmed_start_dates, trimmed_end_dates) if successful,
            None if the series cannot be sanitized (e.g., no valid values exist).
        """
        # Convert to float for consistent handling
        sanitized_values = time_series_values.astype(float)

        # Strip leading and trailing invalid values (NaN/inf/None)
        valid_mask = pd.notna(sanitized_values) & np.isfinite(sanitized_values)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None

        first_valid_index = valid_indices[0]
        last_valid_index = valid_indices[-1]

        # Trim to valid range (remove leading/trailing invalid values)
        sanitized_values = sanitized_values[first_valid_index : last_valid_index + 1]
        trimmed_start_dates = start_dates.iloc[first_valid_index : last_valid_index + 1].reset_index(drop=True)
        trimmed_end_dates = end_dates.iloc[first_valid_index : last_valid_index + 1].reset_index(drop=True)

        # Interpolate any remaining invalid values in the middle of the series
        # This handles NaN, inf, and -inf values uniformly
        if not np.all(np.isfinite(sanitized_values)):
            # Use pandas Series for easy interpolation
            ts_series = pd.Series(sanitized_values)
            # Replace inf/-inf with NaN so pandas can interpolate everything uniformly
            ts_series = ts_series.replace([np.inf, -np.inf], np.nan)
            # Interpolate: first try linear, then forward fill, then backward fill
            ts_series = ts_series.interpolate(method="linear", limit_direction="both")
            ts_series = ts_series.ffill().bfill()
            sanitized_values = ts_series.to_numpy()

        return sanitized_values, trimmed_start_dates, trimmed_end_dates

    def _normalize_sample(
        self, context: np.ndarray, horizon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Normalize context and horizon using z-score normalization based on context statistics.

        This ensures that samples from different domains with vastly different scales
        (e.g., Energy: 0.95-1.75 vs Security: millions-billions) are normalized to
        comparable ranges for stable training.

        Args:
            context: Context values of shape (context_len,).
            horizon: Horizon values of shape (horizon_len,).

        Returns:
            Tuple of (normalized_context, normalized_horizon, context_mean, context_std).
            The normalization parameters can be used to denormalize predictions.
        """
        context_mean = np.mean(context)
        context_std = np.std(context)

        # Avoid division by zero
        epsilon = 1e-6
        if context_std < epsilon:
            context_std = 1.0

        # Normalize both context and horizon using context statistics
        context_normalized = (context - context_mean) / context_std
        horizon_normalized = (horizon - context_mean) / context_std

        return context_normalized, horizon_normalized, float(context_mean), float(context_std)

    def _clean_and_validate_text(self, text: str | None) -> str | None:
        """Clean and validate text, returning cleaned text if valid or None if invalid.

        Filters out text that is:
        - None or NaN
        - Empty or whitespace-only
        - Starts with "NA" (case-sensitive, indicating no information available)

        Args:
            text: Text string to validate and clean.

        Returns:
            Cleaned text string if valid, None if invalid.
        """
        if text is None or pd.isna(text):
            return None

        # Convert to string and strip whitespace
        text_str = str(text).strip()

        # Empty or whitespace-only
        if not text_str:
            return None

        # Starts with "NA" (case-sensitive)
        if text_str.startswith("NA"):
            return None

        return text_str

    def _get_patched_texts_for_period(
        self, start_date: str, end_date: str, textual_data: dict[str, pd.DataFrame], text_patches_num: int
    ) -> list[list[str]]:
        """Gets patched textual descriptions for a specific time period.

        Args:
            start_date: Start date of the time period (YYYY-MM-DD format).
            end_date: End date of the time period (YYYY-MM-DD format).
            textual_data: Dictionary containing textual dataframes.
            text_patches_num: Number of text patches to generate for this period.

        Returns:
            List of lists where each inner list contains text data for one patch period.
            Returns text_patches_num number of lists.
        """
        period_start = pd.to_datetime(start_date)
        period_end = pd.to_datetime(end_date)
        period_duration = period_end - period_start
        patch_duration = period_duration / text_patches_num

        patches = []

        for i in range(text_patches_num):
            patch_start = period_start + i * patch_duration
            patch_end = period_start + (i + 1) * patch_duration

            patch_reports = []

            # Get reports that overlap with this patch period
            if "reports" in textual_data:
                reports_df = textual_data["reports"]
                if "start_date" in reports_df.columns and "end_date" in reports_df.columns:
                    reports_df = reports_df.copy()
                    reports_df["start_date"] = pd.to_datetime(reports_df["start_date"])
                    reports_df["end_date"] = pd.to_datetime(reports_df["end_date"])

                    matching_reports = reports_df[
                        (reports_df["start_date"] <= patch_end) & (reports_df["end_date"] >= patch_start)
                    ]

                    for _, row in matching_reports.iterrows():
                        if "fact" in reports_df.columns:
                            cleaned_fact = self._clean_and_validate_text(row["fact"])
                            if cleaned_fact is not None:
                                patch_reports.append(f"Report: {cleaned_fact}")
                        if "preds" in reports_df.columns:
                            cleaned_preds = self._clean_and_validate_text(row["preds"])
                            if cleaned_preds is not None:
                                patch_reports.append(f"Report Prediction: {cleaned_preds}")

            # Get search data that overlaps with this patch period
            if "search" in textual_data:
                search_df = textual_data["search"]
                if "start_date" in search_df.columns and "end_date" in search_df.columns:
                    search_df = search_df.copy()
                    search_df["start_date"] = pd.to_datetime(search_df["start_date"])
                    search_df["end_date"] = pd.to_datetime(search_df["end_date"])

                    matching_search = search_df[
                        (search_df["start_date"] <= patch_end) & (search_df["end_date"] >= patch_start)
                    ]

                    for _, row in matching_search.iterrows():
                        if "fact" in search_df.columns:
                            cleaned_fact = self._clean_and_validate_text(row["fact"])
                            if cleaned_fact is not None:
                                patch_reports.append(f"Search: {cleaned_fact}")
                        if "preds" in search_df.columns:
                            cleaned_preds = self._clean_and_validate_text(row["preds"])
                            if cleaned_preds is not None:
                                patch_reports.append(f"Search prediction: {cleaned_preds}")

            patches.append(patch_reports)

        return patches

    def _process_data(self, numerical_df: pd.DataFrame, textual_data: dict[str, pd.DataFrame]) -> None:
        """Processes loaded dataframes into internal format.

        Args:
            numerical_df: Dataframe containing numerical time series data.
            textual_data: Dictionary containing textual dataframes (reports, search).
        """
        # Use column configuration to determine which columns to use
        numeric_cols = self.column_config.get_time_series_columns(all_columns=numerical_df.columns.tolist())
        if not numeric_cols:
            raise ValueError(f"No time series columns found for domain {self.domain!r} with the given configuration")

        start_date_col = self.column_config.start_date_col
        end_date_col = self.column_config.end_date_col
        if start_date_col not in numerical_df.columns:
            raise ValueError(
                f"Start date column {start_date_col!r} not found in numerical data. "
                f"Available columns: {numerical_df.columns.tolist()}"
            )
        if end_date_col not in numerical_df.columns:
            raise ValueError(
                f"End date column {end_date_col!r} not found in numerical data. "
                f"Available columns: {numerical_df.columns.tolist()}"
            )
        full_start_dates = numerical_df[start_date_col]
        full_end_dates = numerical_df[end_date_col]

        # Process each numeric column as a separate time series
        for column in numeric_cols:
            # Extract time series from this column
            time_series_values = numerical_df.loc[:, column].to_numpy()

            sanitized = self._sanitize_time_series(time_series_values, full_start_dates, full_end_dates)
            if sanitized is None:
                continue
            sanitized_values, trimmed_start_dates, trimmed_end_dates = sanitized

            ts_data = sanitized_values
            start_dates = trimmed_start_dates
            end_dates = trimmed_end_dates

            # Skip if insufficient data
            if len(ts_data) < self.context_len + self.horizon_len:
                continue

            # Augmentation generates patch_len variants by shifting the sliding-window start position by 0..patch_len-1 steps.
            shifts = range(self.patch_len) if self.augment else range(1)

            text_patches_num = self.context_len // self.patch_len
            for shift in shifts:
                for start_index in range(
                    shift, len(ts_data) - self.context_len - self.horizon_len + 1, self.horizon_len
                ):
                    context_end = start_index + self.context_len
                    context = ts_data[start_index:context_end]

                    horizon_end = context_end + self.horizon_len
                    horizon = ts_data[context_end:horizon_end]

                    context_normalized, horizon_normalized, context_mean, context_std = self._normalize_sample(
                        context, horizon
                    )

                    window_start_date = str(start_dates.iloc[start_index])
                    window_end_date = str(end_dates.iloc[context_end - 1])
                    patched_texts = self._get_patched_texts_for_period(
                        window_start_date, window_end_date, textual_data, text_patches_num
                    )

                    sample = RawSample(
                        context=context_normalized.astype(np.float32),
                        horizon=horizon_normalized.astype(np.float32),
                        patched_texts=patched_texts,
                        metadata={
                            "domain": self.domain,
                            "column": column,
                            "shift": shift,
                            "start_index": start_index,
                            "mean": context_mean,
                            "std": context_std,
                        },
                    )
                    self.data.append(sample)

    def _load_data(self) -> None:
        """Loads Time-MMD dataset from files."""
        numerical_file = self.data_dir / "numerical" / self.domain / f"{self.domain}.csv"
        textual_dir = self.data_dir / "textual" / self.domain

        if not numerical_file.exists():
            raise FileNotFoundError(f"Numerical data file not found: {numerical_file}")

        numerical_df = pd.read_csv(numerical_file)

        # Sort numerical_df by start_date to ensure chronological order
        start_date_col = self.column_config.start_date_col
        if start_date_col in numerical_df.columns:
            numerical_df = numerical_df.sort_values(start_date_col).reset_index(drop=True)

        report_file = textual_dir / f"{self.domain}_report.csv"
        search_file = textual_dir / f"{self.domain}_search.csv"
        textual_data = {}
        if report_file.exists():
            textual_data["reports"] = pd.read_csv(report_file)
        if search_file.exists():
            textual_data["search"] = pd.read_csv(search_file)

        self._process_data(numerical_df, textual_data)

    @classmethod
    def get_domains(cls, path: Path) -> list[str]:
        """Return all available domain names from a Time-MMD dataset directory.

        Args:
            path: Root directory containing Time-MMD dataset.

        Returns:
            Sorted list of domain names.

        Raises:
            FileNotFoundError: If the numerical data directory does not exist.
        """
        numerical_dir = path / "numerical"
        if not numerical_dir.exists():
            raise FileNotFoundError(f"Numerical data directory not found: {numerical_dir}")

        domains = [d.name for d in numerical_dir.iterdir() if d.is_dir()]
        domains.sort()
        return domains

    @override
    def __getitem__(self, index: int) -> RawSample:
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.data)}")
        return self.data[index]

    @override
    def __len__(self) -> int:
        return len(self.data)
