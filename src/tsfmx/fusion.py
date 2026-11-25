"""Multimodal fusion mechanism for combining time series and text embeddings."""

import torch
import torch.nn.functional as F
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


class CrossAttentionFusion(nn.Module):
    """Cross-attention from time series positions to the tokens of a text.

    MultimodalFusion receives one mean-pooled sentence embedding per patch, in which a report's
    content has been averaged into a vector that 97% of its magnitude shares with every other
    report. Here positions query the token embeddings of a text, so the head can weigh the tokens
    that bear on the forecast instead of receiving their average.

    Where the queries come from is set by `forecast_positions`, which MultimodalDecoder reads:

    - False: each context patch queries the text aligned with it.
    - True: every forecast position queries the text in force at the forecast origin. A report
      that forecasts the weather describes the horizon rather than the context, and the forecast
      positions, carrying their own time encoding, can each read the part of it they cover.

    The output projection starts at zero, so an untrained head leaves the time series embeddings
    unchanged and training departs from the text-free forecast. Every projection is bias-free,
    as in MultimodalFusion, so whatever the head adds comes through the tokens it attends to.

    Token rows that are entirely zero are padding. A position whose text has no tokens, because
    it was empty or withheld by text dropout, receives nothing.
    """

    num_heads: torch.Tensor
    forecast_positions: torch.Tensor

    def __init__(
        self,
        ts_embedding_dims: int,
        text_embedding_dims: int,
        num_heads: int = 8,
        forecast_positions: bool = False,
    ) -> None:
        super().__init__()

        if ts_embedding_dims % num_heads != 0:
            raise ValueError(f"ts_embedding_dims ({ts_embedding_dims}) must be divisible by num_heads ({num_heads})")

        self.query = nn.Linear(ts_embedding_dims, ts_embedding_dims, bias=False)
        self.key = nn.Linear(text_embedding_dims, ts_embedding_dims, bias=False)
        self.value = nn.Linear(text_embedding_dims, ts_embedding_dims, bias=False)
        self.output = nn.Linear(ts_embedding_dims, ts_embedding_dims, bias=False)
        nn.init.zeros_(self.output.weight)

        # Saved as buffers because no weight shape records them, and MultimodalDecoder reads them
        # back to rebuild the head before loading a checkpoint. The int copy spares forward a
        # device sync.
        self.register_buffer("num_heads", torch.tensor(num_heads))
        self.register_buffer("forecast_positions", torch.tensor(forecast_positions))
        self._num_heads = num_heads

    @override
    def forward(self, ts_embeddings: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Attend from positions to their text's tokens and add the result to ts_embeddings.

        The positions are split into as many consecutive runs as there are texts, and each run
        queries its own text: one position per text for context patches, every forecast position
        for the single text at the forecast origin.

        Args:
            ts_embeddings: Time series embeddings (batch_size, num_positions, ts_dims).
            text_tokens: Zero-padded token embeddings (batch_size, num_texts, num_tokens, text_dims),
                where num_texts divides num_positions.

        Returns:
            Fused embeddings (batch_size, num_positions, ts_dims).

        Raises:
            ValueError: If num_texts does not divide num_positions.
        """
        batch_size, num_positions, ts_dims = ts_embeddings.shape
        _, num_texts, num_tokens, _ = text_tokens.shape
        if num_positions % num_texts != 0:
            raise ValueError(f"{num_texts} texts cannot be shared evenly by {num_positions} positions")
        queries_per_text = num_positions // num_texts
        heads = self._num_heads
        head_dims = ts_dims // heads

        valid = text_tokens.ne(0).any(dim=-1)
        # A text without tokens would softmax over nothing, which PyTorch's CPU math backend maps to
        # zero but not every fused kernel does. It is attended over its zero padding instead, whose
        # bias-free values are zero, so its positions receive nothing on any backend.
        valid = valid | ~valid.any(dim=-1, keepdim=True)

        query = self.query(ts_embeddings).view(batch_size, num_texts, queries_per_text, heads, head_dims)
        key = self.key(text_tokens).view(batch_size, num_texts, num_tokens, heads, head_dims)
        value = self.value(text_tokens).view(batch_size, num_texts, num_tokens, heads, head_dims)
        attended = F.scaled_dot_product_attention(
            query.transpose(2, 3), key.transpose(2, 3), value.transpose(2, 3), attn_mask=valid[:, :, None, None, :]
        )
        return ts_embeddings + self.output(attended.transpose(2, 3).reshape(batch_size, num_positions, ts_dims))
