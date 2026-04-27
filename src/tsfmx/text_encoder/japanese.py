"""Japanese text encoder using SentenceTransformer models."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from tsfmx.text_encoder.base import TextEncoderBase


class JapaneseTextEncoder(TextEncoderBase):
    """Text encoder for Japanese text using SentenceTransformer models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-310m",
        embedding_dim: int = 768,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the Japanese text encoder.

        Args:
            model_name: Name of the SentenceTransformer model to use.
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations. Can be torch.device, str, or None for auto-detection.
        """
        super().__init__(embedding_dim, device)
        self.sentence_transformer = SentenceTransformer(model_name, device=self.device.type)

        self._validate()

    def _validate(self) -> None:
        actual_dim = self.sentence_transformer.get_embedding_dimension()
        if actual_dim is None:
            raise ValueError("Could not determine embedding dimension from sentence transformer")
        if actual_dim != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {actual_dim}.")

    def forward(self, texts: str | list[str] | np.ndarray) -> torch.Tensor:
        """Encode texts into embeddings.

        Args:
            texts: Input texts to encode.

        Returns:
            Tensor of shape (embedding_dim,) for a single text or (N, embedding_dim) for N texts.
        """
        return self.sentence_transformer.encode(texts, convert_to_tensor=True)

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
