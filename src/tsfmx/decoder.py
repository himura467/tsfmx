"""Multimodal decoder for time series forecasting with text fusion."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from typing_extensions import override

import torch
from torch import nn

from tsfmx.fusion import CrossAttentionFusion, MultimodalFusion
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.types import Batch, TrainingMode
from tsfmx.utils.logging import get_logger

_logger = get_logger()

_PROJECTION_WEIGHT = re.compile(r"^projection\.(\d+)\.weight$")

FusionType = Literal["linear", "cross_attention"]
FusionPosition = Literal["context", "forecast"]


@dataclass
class MultimodalDecoderConfig:
    """Configuration for MultimodalDecoder.

    `fusion_type` picks the head: 'linear' projects one pooled embedding per patch and reads
    `num_fusion_layers`, `fusion_hidden_dims` and `fusion_normalize`; 'cross_attention' attends
    to token embeddings and reads `num_attention_heads` and `fusion_position`. The latter places
    the queries on the context patches, each reading its own patch's text, or on the forecast
    positions, all reading the text in force at the forecast origin.
    """

    text_embedding_dims: int = 384
    num_fusion_layers: int = 1
    fusion_hidden_dims: list[int] = field(default_factory=list)
    fusion_normalize: bool = False
    fusion_type: FusionType = "linear"
    num_attention_heads: int = 8
    fusion_position: FusionPosition = "context"


class MultimodalDecoder(nn.Module):
    """Decoder for multimodal time series forecasting.

    Pipeline: adapter.preprocess -> fusion -> adapter.forward -> adapter.postprocess
    """

    def __init__(self, adapter: TsfmAdapter, config: MultimodalDecoderConfig) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config
        self.fusion = self._build_fusion(config)

    def _build_fusion(self, config: MultimodalDecoderConfig) -> MultimodalFusion | CrossAttentionFusion:
        """Construct the fusion head that `config` describes.

        Raises:
            ValueError: If a linear head is placed on the forecast positions, which have no patch
                text of their own to project.
        """
        match config.fusion_type:
            case "linear":
                if config.fusion_position == "forecast":
                    raise ValueError("A linear fusion head reads context patches only; use cross_attention")
                return MultimodalFusion(
                    ts_embedding_dims=self.adapter.model_dims,
                    text_embedding_dims=config.text_embedding_dims,
                    num_layers=config.num_fusion_layers,
                    hidden_dims=config.fusion_hidden_dims,
                    normalize=config.fusion_normalize,
                )
            case "cross_attention":
                return CrossAttentionFusion(
                    ts_embedding_dims=self.adapter.model_dims,
                    text_embedding_dims=config.text_embedding_dims,
                    num_heads=config.num_attention_heads,
                    forecast_positions=config.fusion_position == "forecast",
                )

    @staticmethod
    def _fusion_config_from_state_dict(state_dict: dict[str, Any]) -> MultimodalDecoderConfig:
        """Recover the fusion architecture from the shapes its weights were saved with.

        A checkpoint records no architecture, and a sweep trial picks its own, so the loading
        script would otherwise have to restate that trial's hyperparameters. Within `projection`
        the Linear weights are the 2-D entries and the optional trailing RMSNorm is the 1-D one.
        A CrossAttentionFusion is recognized by its key projection and saves its head count.

        Args:
            state_dict: A MultimodalFusion or CrossAttentionFusion state dict.

        Returns:
            Config describing the architecture the weights belong to.

        Raises:
            ValueError: If the state dict holds no projection weight to read.
        """
        if "key.weight" in state_dict:
            return MultimodalDecoderConfig(
                text_embedding_dims=int(state_dict["key.weight"].shape[1]),
                fusion_type="cross_attention",
                num_attention_heads=int(state_dict["num_heads"]),
                fusion_position="forecast" if bool(state_dict["forecast_positions"]) else "context",
            )

        entries: list[tuple[int, torch.Tensor]] = []
        for key, value in state_dict.items():
            match = _PROJECTION_WEIGHT.match(key)
            if match is not None:
                entries.append((int(match.group(1)), value))
        entries.sort(key=lambda entry: entry[0])

        linear_weights = [value for _, value in entries if value.ndim == 2]
        if not linear_weights:
            raise ValueError("Fusion state dict holds no 2-D projection weight to read an architecture from.")

        return MultimodalDecoderConfig(
            text_embedding_dims=int(linear_weights[0].shape[1]),
            num_fusion_layers=len(linear_weights),
            fusion_hidden_dims=[int(weight.shape[0]) for weight in linear_weights[:-1]],
            fusion_normalize=any(value.ndim == 1 for _, value in entries),
        )

    def _rebuild_fusion(self, config: MultimodalDecoderConfig) -> None:
        """Replace the fusion module with one shaped by `config`, on the device it already uses."""
        device = next(self.fusion.parameters()).device
        self.config = config
        self.fusion = self._build_fusion(config).to(device)

    def load_checkpoint(self, path: Path) -> TrainingMode:
        """Load a training checkpoint, auto-detecting the checkpoint type.

        Supports all three training modes (fusion, finetune, and adapter) by
        inspecting which state-dict keys are present in the file.

        The fusion module is rebuilt whenever the checkpoint's saved shapes disagree with the one
        this decoder was constructed with.

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
            saved_config = self._fusion_config_from_state_dict(checkpoint["fusion_state_dict"])
            if saved_config != self.config:
                _logger.info("Rebuilding fusion to match %s: %s (constructed as %s)", path, saved_config, self.config)
                self._rebuild_fusion(saved_config)
            fusion_state: dict[str, Any] = checkpoint["fusion_state_dict"]
            if isinstance(self.fusion, MultimodalFusion) and "text_mean" not in fusion_state:
                # Predates centering, so it was trained on uncentered embeddings: zero leaves the
                # projection's input unchanged.
                fusion_state = {**fusion_state, "text_mean": torch.zeros_like(self.fusion.text_mean)}
            if isinstance(self.fusion, MultimodalFusion) and "constant_text" not in fusion_state:
                # Predates the constant-text control, so it was trained on the real text.
                fusion_state = {**fusion_state, "constant_text": torch.tensor(False)}
            self.fusion.load_state_dict(fusion_state)
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
            text_embeddings: Pre-computed text in the form `select_text` returns: pooled
                (batch_size, num_patches, text_dims), or token embeddings
                (batch_size, num_patches, num_tokens, text_dims) for cross-attention.

        Returns:
            Predictions (batch_size, horizon, num_outputs).

        Raises:
            ValueError: If masks shape does not match inputs shape.
        """
        if masks.shape != inputs.shape:
            raise ValueError(f"masks shape {masks.shape} must match inputs shape {inputs.shape}")
        masks = masks.bool()
        preprocessed = self.adapter.preprocess(inputs, masks)
        embeddings = preprocessed.input_embeddings
        forecast_embeddings = None
        if text_embeddings is not None and self.config.fusion_position == "forecast":
            # The last context patch's text is the one in force at the forecast origin.
            forecast_embeddings = self.fusion(self.adapter.forecast_embeddings(embeddings), text_embeddings[:, -1:])
        elif text_embeddings is not None:
            embeddings = self.fusion(embeddings, text_embeddings)
        output_embeddings = self.adapter(embeddings, preprocessed.masks, forecast_embeddings)
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
            text_embeddings: Pre-computed text in the form `select_text` returns: pooled
                (batch_size, num_patches, text_dims), or token embeddings
                (batch_size, num_patches, num_tokens, text_dims) for cross-attention.

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
        text = self.select_text(batch)
        forecast = self.forward(
            horizon.shape[-1], context, input_padding, text.to(device) if text is not None else None
        )
        return forecast, context, horizon

    def select_text(self, batch: Batch) -> torch.Tensor | None:
        """Return the batch's text in the form the fusion head reads.

        Args:
            batch: A collated batch.

        Returns:
            Token embeddings for a CrossAttentionFusion, pooled embeddings otherwise, or None
            when the batch carries no text, which runs the pipeline without fusion.

        Raises:
            ValueError: If the batch has pooled embeddings but a CrossAttentionFusion needs tokens,
                which would otherwise skip fusion without a word.
        """
        if isinstance(self.fusion, CrossAttentionFusion):
            if "text_tokens" in batch:
                return batch["text_tokens"]
            if "text_embeddings" in batch:
                raise ValueError("Cross-attention fusion needs 'text_tokens'; load a cache built with text tokens.")
            return None
        return batch.get("text_embeddings")
