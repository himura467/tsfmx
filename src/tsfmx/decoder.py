"""Multimodal decoder for time series forecasting with text fusion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing_extensions import override

import torch
from torch import nn

from tsfmx.fusion import MultimodalFusion
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.types import Batch, TrainingMode


@dataclass
class MultimodalDecoderConfig:
    """Configuration for MultimodalDecoder."""

    text_embedding_dims: int = 384
    num_fusion_layers: int = 1
    fusion_hidden_dims: list[int] = field(default_factory=list)


class MultimodalDecoder(nn.Module):
    """Decoder for multimodal time series forecasting.

    Pipeline: adapter.preprocess -> fusion -> adapter.forward -> adapter.postprocess
    """

    def __init__(self, adapter: TsfmAdapter, config: MultimodalDecoderConfig) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config
        self.fusion = MultimodalFusion(
            ts_embedding_dims=adapter.model_dims,
            text_embedding_dims=config.text_embedding_dims,
            num_layers=config.num_fusion_layers,
            hidden_dims=config.fusion_hidden_dims,
        )

    def load_checkpoint(self, path: Path) -> TrainingMode:
        """Load a training checkpoint, auto-detecting the checkpoint type.

        Supports all three training modes (fusion, finetune, and adapter) by
        inspecting which state-dict keys are present in the file.

        Args:
            path: Path to a .pt checkpoint file.

        Returns:
            Detected training mode: 'fusion' if only fusion weights are present,
            'finetune' if both fusion and adapter weights are present, or 'adapter'
            if only adapter weights are present.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the checkpoint format cannot be recognized.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
        has_fusion = "fusion_state_dict" in checkpoint
        has_adapter = "adapter_state_dict" in checkpoint
        if not has_fusion and not has_adapter:
            raise ValueError(
                f"Unrecognized checkpoint format in {path!r}. "
                "Expected at least one of 'fusion_state_dict' or 'adapter_state_dict'."
            )

        if has_fusion:
            self.fusion.load_state_dict(checkpoint["fusion_state_dict"])
        if has_adapter:
            self.adapter.load_state_dict(checkpoint["adapter_state_dict"])

        if has_fusion and has_adapter:
            return "finetune"
        if has_fusion:
            return "fusion"
        return "adapter"

    def forward_full(
        self,
        horizon: int,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forecasting pipeline, returning all output channels.

        When text_embeddings is provided, fusion is applied before decoding.
        When None, the pipeline runs without fusion.

        Args:
            horizon: Number of time steps to forecast.
            inputs: Input time series (batch_size, context_len).
            masks: Boolean masks (batch_size, context_len). True = padded, False = valid.
            text_embeddings: Pre-computed text embeddings (batch_size, num_patches, text_dims).

        Returns:
            Predictions (batch_size, horizon, num_outputs).

        Raises:
            ValueError: If masks shape does not match inputs shape.
        """
        if masks.shape != inputs.shape:
            raise ValueError(f"masks shape {masks.shape} must match inputs shape {inputs.shape}")
        masks = masks.bool()
        preprocessed = self.adapter.preprocess(inputs, masks)
        embeddings = (
            self.fusion(preprocessed.input_embeddings, text_embeddings)
            if text_embeddings is not None
            else preprocessed.input_embeddings
        )
        output_embeddings = self.adapter(embeddings, preprocessed.masks)
        return self.adapter.postprocess(horizon, output_embeddings, preprocessed.normalization_stats)

    @override
    def forward(
        self,
        horizon: int,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forecasting pipeline, returning the point forecast.

        Args:
            horizon: Number of time steps to forecast.
            inputs: Input time series (batch_size, context_len).
            masks: Boolean masks (batch_size, context_len). True = padded, False = valid.
            text_embeddings: Pre-computed text embeddings (batch_size, num_patches, text_dims).

        Returns:
            Point forecast (batch_size, horizon).
        """
        return self.forward_full(horizon, inputs, masks, text_embeddings)[..., self.adapter.point_forecast_index]

    def forecast(self, batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on a single batch.

        Args:
            batch: A collated batch of preprocessed samples.
            device: Device to move tensors onto.

        Returns:
            Tuple of (forecast, context, horizon), all on device.
            forecast: Point forecast (batch_size, horizon_len).
            context: Input context (batch_size, context_len).
            horizon: True future values (batch_size, horizon_len).
        """
        context = batch["context"].to(device)
        horizon = batch["horizon"].to(device)
        input_padding = torch.zeros_like(context, dtype=torch.bool)
        text_embeddings = batch["text_embeddings"].to(device) if "text_embeddings" in batch else None
        forecast = self.forward(horizon.shape[-1], context, input_padding, text_embeddings)
        return forecast, context, horizon
