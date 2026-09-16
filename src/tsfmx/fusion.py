"""Multimodal fusion mechanism for combining time series and text embeddings."""

import torch
from torch import nn
from typing_extensions import override


class MultimodalFusion(nn.Module):
    """Addition-based fusion of time series and text embeddings.

    Projects text_embeddings to ts_embedding_dims, then adds element-wise.

    Diagnostics on a trained checkpoint showed the projection output reaching a magnitude
    rivalling the time series embeddings it is added to, which `normalize` exists to control;
    `projection_vs_ts_rms` in scripts/diagnose_time_mmd_text_fusion.py reports that ratio.
    It defaults to off, so the module is unchanged unless asked.
    """

    text_mean: torch.Tensor
    constant_text: torch.Tensor

    def __init__(
        self,
        ts_embedding_dims: int,
        text_embedding_dims: int,
        num_layers: int = 1,
        hidden_dims: list[int] = [],
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self._validate(num_layers, hidden_dims)

        dims = [text_embedding_dims, *hidden_dims, ts_embedding_dims]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))  # bias deemed unnecessary by W&B Sweeps
            # No activation after the last Linear: it would confine the projection to the
            # non-negative orthant, where fusion can only add to the time series embedding and
            # every projected sample is similar to every other by construction.
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if normalize:
            # Divides out the projection's own output scale and replaces it with an explicit
            # learned one, so the model can admit less text instead of being forced to accept
            # whatever magnitude the projection happens to produce. RMSNorm rather than
            # LayerNorm because scale, not offset, is the failure it corrects.
            layers.append(nn.RMSNorm(ts_embedding_dims))
        # Normalization lives inside `projection` so that it stays the single expression of
        # what fusion adds, and callers cannot read a partial transform.
        self.projection = nn.Sequential(*layers)

        # Sentence embeddings are strongly anisotropic: measured on Fidel-TS, 97% of the magnitude
        # of an all-MiniLM-L6-v2 embedding lies in a direction shared by every sample, which a
        # bias-free projection can only pass on as a constant offset. Subtracting it leaves the
        # part that varies. Zeros until set_text_mean is called, so centering is off by default.
        self.register_buffer("text_mean", torch.zeros(text_embedding_dims))
        # When set, the projection sees text_mean in place of every sample's text instead of having
        # it subtracted, so fusion can learn only what a constant input allows. A buffer rather than
        # an attribute so that evaluation scripts inherit the control from the checkpoint.
        self.register_buffer("constant_text", torch.tensor(False))

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

    def set_text_mean(self, mean: torch.Tensor) -> None:
        """Center the projection's input on `mean`, which then travels with the checkpoint.

        Compute it on the training split alone, and as one mean across entities rather than one
        each: a per-entity mean would subtract exactly the between-entity component that the
        cross_domain ablation exists to measure.

        Note that centering makes the `mean` ablation nearly equivalent to `drop`, since the
        replacement embedding then projects to approximately zero.

        Args:
            mean: Mean text embedding, of shape (text_embedding_dims,).

        Raises:
            ValueError: If mean does not match the configured text embedding dimension.
        """
        if mean.shape != self.text_mean.shape:
            raise ValueError(f"mean must have shape {tuple(self.text_mean.shape)}, got {tuple(mean.shape)}")
        self.text_mean.copy_(mean.to(device=self.text_mean.device, dtype=self.text_mean.dtype))

    def set_constant_text(self, mean: torch.Tensor) -> None:
        """Replace every sample's text with `mean` before projecting, as a control for reading it.

        A fusion head can beat the text-free forecast without reading anything, by learning an
        input-independent offset. Training on a constant input leaves it only that route, so the
        gap between this control and a head trained on the real text is what the text's content
        adds. With a single bias-free Linear the projection of `mean` can reach any vector, which
        makes the control equivalent to a learned bias of the same optimizer and search space.

        Samples whose text is withheld (all-zero embeddings, as training-time text dropout
        produces) stay zero, so dropout still skips fusion for them. Mutually exclusive with
        centering, which would reduce the constant to zero.

        Args:
            mean: Mean text embedding of the training split, of shape (text_embedding_dims,).

        Raises:
            ValueError: If mean does not match the configured text embedding dimension.
        """
        self.set_text_mean(mean)
        self.constant_text.fill_(True)

    @override
    def forward(self, ts_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project text_embeddings to ts_embedding_dims and add to ts_embeddings."""
        if self.constant_text:
            present = text_embeddings.ne(0).any(dim=-1, keepdim=True).to(text_embeddings.dtype)
            inputs = present * self.text_mean
        else:
            inputs = text_embeddings - self.text_mean
        projected: torch.Tensor = self.projection(inputs)
        return ts_embeddings + projected

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
