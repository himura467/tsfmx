"""Collate functions for DataLoader batching."""

from typing import Callable

import numpy as np
import torch

from tsfmx.types import Batch, PreprocessedSample, TrainingMode


def _build_batch(batch: list[PreprocessedSample]) -> Batch:
    context = torch.from_numpy(np.stack([s["context"] for s in batch]))
    horizon = torch.from_numpy(np.stack([s["horizon"] for s in batch]))
    metadata = [s["metadata"] for s in batch]
    return Batch(
        context=context,
        horizon=horizon,
        metadata=metadata,
    )


def _pad_text_tokens(batch: list[PreprocessedSample]) -> torch.Tensor:
    """Stack per-patch token embeddings into (batch_size, num_patches, num_tokens, text_dims).

    Patches are zero-padded to the longest text in the batch, and to at least one token so that
    attention always has a slot; CrossAttentionFusion reads all-zero rows as padding.
    """
    tokens = [s["text_tokens"] for s in batch]
    num_tokens = max(1, *(patch.shape[0] for sample in tokens for patch in sample))
    padded = np.zeros((len(tokens), len(tokens[0]), num_tokens, tokens[0][0].shape[-1]), dtype=np.float32)
    for i, sample in enumerate(tokens):
        for j, patch in enumerate(sample):
            padded[i, j, : patch.shape[0]] = patch
    return torch.from_numpy(padded)


def multimodal_collate_fn(batch: list[PreprocessedSample]) -> Batch:
    """Collate function for multimodal batches with pre-computed text embeddings."""
    result = _build_batch(batch)
    result["text_embeddings"] = torch.from_numpy(np.stack([s["text_embeddings"] for s in batch]))
    if "text_tokens" in batch[0]:
        result["text_tokens"] = _pad_text_tokens(batch)
    return result


def adapter_collate_fn(batch: list[PreprocessedSample]) -> Batch:
    """Collate function for adapter batches (no text embeddings)."""
    return _build_batch(batch)


def collate_fn_for_mode(mode: TrainingMode) -> Callable[[list[PreprocessedSample]], Batch]:
    """Return the collate function matching a training mode.

    Args:
        mode: Training mode. 'fusion' and 'finetune' consume text embeddings; 'adapter' does not.

    Returns:
        `multimodal_collate_fn` for text-consuming modes, `adapter_collate_fn` otherwise.
    """
    return multimodal_collate_fn if mode in ("fusion", "finetune") else adapter_collate_fn
