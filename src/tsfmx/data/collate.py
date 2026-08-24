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


def multimodal_collate_fn(batch: list[PreprocessedSample]) -> Batch:
    """Collate function for multimodal batches with pre-computed text embeddings."""
    result = _build_batch(batch)
    result["text_embeddings"] = torch.from_numpy(np.stack([s["text_embeddings"] for s in batch]))
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
