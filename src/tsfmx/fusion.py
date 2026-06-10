"""Multimodal fusion mechanism for combining time series and text embeddings."""

import torch
from torch import nn
from typing_extensions import override


class MultimodalFusion(nn.Module):
    """Addition-based fusion of time series and text embeddings.

    Projects text_embeddings to ts_embedding_dims, then adds element-wise.
    """

    def __init__(
        self,
        ts_embedding_dims: int,
        text_embedding_dims: int,
        num_layers: int = 1,
        hidden_dims: list[int] = [],
    ) -> None:
        super().__init__()

        self._validate(num_layers, hidden_dims)

        dims = [text_embedding_dims, *hidden_dims, ts_embedding_dims]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))  # bias deemed unnecessary by W&B Sweeps
            layers.append(nn.ReLU())
        self.projection = nn.Sequential(*layers)

        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _validate(self, num_layers: int, hidden_dims: list[int]) -> None:
        if num_layers < 1 or num_layers > 3:
            raise ValueError(f"num_layers must be between 1 and 3, got {num_layers}")
        if len(hidden_dims) != num_layers - 1:
            raise ValueError(
                f"hidden_dims must have {num_layers - 1} elements for {num_layers} layers, got {len(hidden_dims)}"
            )

    @override
    def forward(self, ts_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project text_embeddings to ts_embedding_dims and add to ts_embeddings."""
        projected: torch.Tensor = self.projection(text_embeddings)
        return ts_embeddings + projected

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
