"""Chronos adapters."""

from __future__ import annotations

from typing import cast

import torch
from chronos import Chronos2Model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tsfmx.tsfm.base import PreprocessResult, TsfmAdapter
from tsfmx.utils.logging import get_logger


class Chronos2Adapter(TsfmAdapter):
    """Adapter wrapping Amazon Chronos-2 (120M encoder-only model)."""

    def __init__(self, model: Chronos2Model) -> None:
        super().__init__()
        self._model = model

    @property
    def model_dims(self) -> int:
        return self._model.model_dim

    @property
    def patch_len(self) -> int:
        return self._model.chronos_config.input_patch_size

    @property
    def point_forecast_index(self) -> int:
        return list(self._model.chronos_config.quantiles).index(0.5)

    def preprocess(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> PreprocessResult:
        """Normalize, patch, time-encode, and embed the input time series.

        Returns:
            PreprocessResult with input_embeddings (batch_size, num_context_patches, model_dims),
            masks in project convention (True = padded), and normalization_stats
            containing "loc" and "scale" each of shape (batch_size, 1).
        """
        # Flip to Chronos-2 convention: 1.0 = valid, 0.0 = padded.
        context_mask = (~masks).to(inputs.dtype)

        patched_context, attention_mask, (loc, scale) = self._model._prepare_patched_context(
            context=inputs, context_mask=context_mask
        )

        input_embeds: torch.Tensor = self._model.input_patch_embedding(patched_context)

        return PreprocessResult(
            input_embeddings=input_embeds,
            masks=attention_mask == 0,
            normalization_stats={"loc": loc, "scale": scale},
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the Chronos-2 encoder.

        Appends the [REG] token (if enabled) and zero future-patch embeddings,
        runs all encoder blocks, and returns only the forecast-position hidden states.

        Returns:
            Output embeddings (batch_size, max_output_patches, model_dims).
        """
        batch_size = input_embeddings.shape[0]
        dtype = input_embeddings.dtype
        device = input_embeddings.device
        num_output_patches = self._model.chronos_config.max_output_patches
        output_patch_size = self._model.chronos_config.output_patch_size
        time_encoding_scale = cast(int, self._model.chronos_config.time_encoding_scale)

        patched_future_covariates = torch.zeros(
            batch_size, num_output_patches, output_patch_size, dtype=dtype, device=device
        )
        patched_future_covariates_mask = torch.zeros(
            batch_size, num_output_patches, output_patch_size, dtype=dtype, device=device
        )

        final_future_length = num_output_patches * output_patch_size
        future_time_enc = (
            torch.arange(0, final_future_length, dtype=torch.float32, device=device)
            .div(time_encoding_scale)
            .reshape(1, num_output_patches, output_patch_size)
            .expand(batch_size, -1, -1)
            .to(dtype)
        )

        future_embeds: torch.Tensor = self._model.input_patch_embedding(
            torch.cat([future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1)
        )

        # Flip to Chronos-2 convention: 1.0 = valid, 0.0 = padded.
        attention_mask = (~masks).to(dtype)

        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=dtype, device=device)
        if self._model.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self._model.config.reg_token_id, device=device)
            reg_embeds: torch.Tensor = self._model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeddings, reg_embeds, future_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(reg_input_ids).to(dtype), future_attention_mask], dim=-1
            )
        else:
            input_embeds = torch.cat([input_embeddings, future_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        group_ids = torch.arange(batch_size, dtype=torch.long, device=device)

        encoder_outputs = self._model.encoder(
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            attention_mask=attention_mask,
        )

        hidden_states: torch.Tensor = encoder_outputs[0]
        return hidden_states[:, -num_output_patches:]

    def postprocess(
        self,
        horizon: int,
        output_embeddings: torch.Tensor,
        normalization_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project to quantiles, denormalize, and slice to the requested horizon.

        Returns:
            Predictions (batch_size, horizon, num_quantiles).

        Raises:
            ValueError: If horizon exceeds the model's maximum prediction length.
        """
        num_output_patches = self._model.chronos_config.max_output_patches
        output_patch_size = self._model.chronos_config.output_patch_size

        max_horizon = num_output_patches * output_patch_size
        if horizon > max_horizon:
            raise ValueError(
                f"horizon ({horizon}) exceeds the maximum prediction length "
                f"({max_horizon} = {num_output_patches} patches * {output_patch_size} steps)."
            )

        batch_size = output_embeddings.shape[0]
        num_quantiles = self._model.num_quantiles

        loc = normalization_stats["loc"]
        scale = normalization_stats["scale"]

        quantile_preds: torch.Tensor = self._model.output_patch_embedding(output_embeddings)
        quantile_preds = (
            quantile_preds.reshape(batch_size, num_output_patches, num_quantiles, output_patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_quantiles, max_horizon)
        )

        quantile_preds = self._model.instance_norm.inverse(
            quantile_preds.reshape(batch_size, num_quantiles * max_horizon), (loc, scale)
        ).reshape(batch_size, num_quantiles, max_horizon)

        return quantile_preds[:, :, :horizon].permute(0, 2, 1)

    def load_checkpoint(self, path: str) -> None:
        """Load a Chronos-2 model from a checkpoint."""
        tensors = load_file(path)
        self._model.load_state_dict(tensors, strict=True)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        repo_id: str = "amazon/chronos-2",
    ) -> Chronos2Adapter:
        """Create a Chronos2Adapter with pretrained weights loaded.

        Args:
            device: Device to place the adapter on.
            repo_id: Hugging Face repository ID for pretrained weights.

        Returns:
            Chronos2Adapter instance with pretrained weights.
        """
        logger = get_logger()
        config = Chronos2Model.config_class.from_pretrained(repo_id)
        instance = cls(Chronos2Model(config))
        instance.to(device)
        logger.info("Downloading checkpoint from Hugging Face repo %s", repo_id)
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        logger.info("Loading checkpoint from %s", checkpoint_path)
        instance.load_checkpoint(checkpoint_path)
        return instance

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
