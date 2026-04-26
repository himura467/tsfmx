"""Dataset classes for multimodal time series."""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from tsfmx.types import PreprocessedSample, RawSample, TrainingMode


class MultimodalDatasetBase(Dataset[RawSample], ABC):
    """Abstract base class for multimodal time series datasets."""

    @abstractmethod
    def __getitem__(self, index: int) -> RawSample: ...

    @abstractmethod
    def __len__(self) -> int: ...


class PreprocessedDataset(Dataset[PreprocessedSample]):
    """Dataset wrapping pre-processed samples with pre-computed text embeddings.

    Args:
        data: List of preprocessed samples.
        mode: 'fusion' and 'finetune' require text_embeddings in every sample;
            'adapter' does not use text_embeddings.
    """

    def __init__(self, data: list[PreprocessedSample], mode: TrainingMode) -> None:
        self.data = data
        self.mode = mode

        self._validate()

    def _validate(self) -> None:
        if self.mode in ("fusion", "finetune") and not all("text_embeddings" in s for s in self.data):
            raise ValueError(f"All samples must contain 'text_embeddings' for {self.mode!r} mode")

    def __getitem__(self, index: int) -> PreprocessedSample:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
