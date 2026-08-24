"""DataLoader construction for preprocessed datasets."""

from typing import Callable, cast

import torch
from torch.utils.data import DataLoader, Dataset

from tsfmx.types import Batch, PreprocessedSample
from tsfmx.utils.device import pin_memory


def build_dataloader(
    dataset: Dataset[PreprocessedSample],
    batch_size: int,
    collate_fn: Callable[[list[PreprocessedSample]], Batch],
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader[Batch]:
    """Build a DataLoader over preprocessed samples.

    num_workers is fixed at 0 because samples are already fully preprocessed in memory,
    so worker processes would only add pickling overhead for the text embeddings.

    Args:
        dataset: Dataset of preprocessed samples.
        batch_size: Number of samples per batch.
        collate_fn: Collate function matching the training mode.
        device: Device used to determine whether to pin memory.
        shuffle: Whether to shuffle the samples.

    Returns:
        DataLoader yielding collated batches.
    """
    return cast(
        DataLoader[Batch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=pin_memory(device),
        ),
    )
