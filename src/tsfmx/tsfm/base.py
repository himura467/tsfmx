"""Abstract adapter interface for time series foundation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
from typing_extensions import override


@dataclass
class PreprocessResult:
    """Result of the preprocessing.

    Attributes:
        input_embeddings: Embeddings produced by the adapter's tokenizer.
        masks: Boolean masks. True = padded, False = valid.
        normalization_stats: Adapter-specific normalization statistics.
    """

    input_embeddings: torch.Tensor
    masks: torch.Tensor
    normalization_stats: dict[str, torch.Tensor]


class TsfmAdapter(nn.Module, ABC):
    """Base interface for time series foundation model adapters.

    Pipeline: preprocess -> [fusion injection point] -> decode -> postprocess
    """

    @property
    @abstractmethod
    def model_dims(self) -> int:
        """Hidden dimension of the adapter's transformer."""
        ...

    @property
    @abstractmethod
    def patch_len(self) -> int:
        """Number of raw time series steps per input patch."""
        ...

    @property
    @abstractmethod
    def point_forecast_index(self) -> int:
        """Index into the last dimension of postprocess output that gives the point forecast."""
        ...

    @abstractmethod
    def preprocess(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> PreprocessResult: ...

    @override
    @abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def postprocess(
        self,
        horizon: int,
        output_embeddings: torch.Tensor,
        normalization_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor: ...

    @abstractmethod
    def freeze_parameters(self) -> None: ...

    @abstractmethod
    def unfreeze_parameters(self) -> None: ...
