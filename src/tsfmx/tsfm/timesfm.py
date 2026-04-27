"""TimesFM adapters."""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from timesfm.torch.util import revin, update_running_stats

from tsfmx.tsfm.base import PreprocessResult, TsfmAdapter
from tsfmx.utils.logging import get_logger


class TimesFM2p5Adapter(TsfmAdapter):
    """Adapter wrapping TimesFM 2.5 200M."""

    def __init__(self) -> None:
        super().__init__()
        self._model = TimesFM_2p5_200M_torch_module()

    @property
    def model_dims(self) -> int:
        return self._model.md

    @property
    def patch_len(self) -> int:
        return self._model.p

    @property
    def point_forecast_index(self) -> int:
        return self._model.config.decode_index

    def preprocess(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> PreprocessResult:
        """Patch, normalize (RevIN), and tokenize input time series.

        Raises:
            ValueError: If context length is not divisible by patch length, or
                if masks shape does not match inputs shape.
        """
        batch_size, context = inputs.shape[0], inputs.shape[1]
        if context % self._model.p != 0:
            raise ValueError(f"context length ({context}) must be divisible by patch length ({self._model.p})")
        if masks.shape != inputs.shape:
            raise ValueError(f"masks shape {masks.shape} must match inputs shape {inputs.shape}")
        num_input_patches = context // self._model.p

        patched_inputs = inputs.reshape(batch_size, -1, self._model.p)
        patched_masks = masks.reshape(batch_size, -1, self._model.p)

        # Compute running stats per patch for RevIN
        n = torch.zeros(batch_size, device=inputs.device)
        mu = torch.zeros(batch_size, device=inputs.device)
        sigma = torch.zeros(batch_size, device=inputs.device)
        patch_mu: list[torch.Tensor] = []
        patch_sigma: list[torch.Tensor] = []
        for i in range(num_input_patches):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        tokenizer_inputs = torch.cat([normed_inputs, patched_masks.to(normed_inputs.dtype)], dim=-1)
        input_embeddings = self._model.tokenizer(tokenizer_inputs)

        return PreprocessResult(
            input_embeddings=input_embeddings,
            masks=patched_masks,
            normalization_stats={
                "context_mu": context_mu,
                "context_sigma": context_sigma,
            },
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers.

        Returns:
            Output embeddings (batch_size, num_patches, model_dims).
        """
        output_embeddings = input_embeddings
        for layer in self._model.stacked_xf:
            output_embeddings, _ = layer(output_embeddings, masks[..., -1], None)
        return output_embeddings

    def postprocess(
        self,
        horizon: int,
        output_embeddings: torch.Tensor,
        normalization_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project embeddings to forecast quantiles and reverse RevIN.

        horizon must fit within a single output patch (no AR decode).

        Returns:
            Predictions (batch_size, horizon, num_outputs).

        Raises:
            ValueError: If horizon > output_patch_len.
        """
        if horizon > self._model.o:
            raise ValueError(
                f"horizon must be <= output_patch_len ({self._model.o}), got {horizon}. AR decode is not supported."
            )

        batch_size = output_embeddings.shape[0]
        context_mu = normalization_stats["context_mu"]
        context_sigma = normalization_stats["context_sigma"]

        output_ts = self._model.output_projection_point(output_embeddings)
        renormed_outputs = torch.reshape(
            revin(output_ts, context_mu, context_sigma, reverse=True), (batch_size, -1, self._model.o, self._model.q)
        )
        return renormed_outputs[:, -1, :horizon, :]

    def load_checkpoint(self, path: str) -> None:
        """Load a TimesFM 2.5 model from a checkpoint."""
        tensors = load_file(path)
        self._model.load_state_dict(tensors, strict=True)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
    ) -> TimesFM2p5Adapter:
        """Create a TimesFM2p5Adapter with pretrained weights loaded.

        Args:
            device: Device to place the adapter on.
            repo_id: Hugging Face repository ID for pretrained weights.

        Returns:
            TimesFM2p5Adapter instance with pretrained weights.
        """
        logger = get_logger()
        instance = cls()
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
