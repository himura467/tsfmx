"""Abstract text encoder interface."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from typing_extensions import override

from tsfmx.utils.device import resolve_device


class TextEncoderBase(nn.Module, ABC):
    """Abstract base class for text encoders."""

    def __init__(self, embedding_dim: int, device: torch.device | str | None = None) -> None:
        """Initialize the base text encoder.

        Args:
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations. Can be torch.device, str, or None for auto-detection.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = resolve_device(device)

    @override
    @abstractmethod
    def forward(self, texts: str | list[str] | np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def encode_tokens(self, texts: list[str]) -> list[torch.Tensor]:
        """Encode texts into per-token embeddings, before any pooling.

        Args:
            texts: Input texts to encode.

        Returns:
            One tensor of shape (num_tokens, embedding_dim) per text, without padding.
        """
        ...

    @abstractmethod
    def freeze_parameters(self) -> None: ...

    @abstractmethod
    def unfreeze_parameters(self) -> None: ...
