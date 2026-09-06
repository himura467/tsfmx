"""Statistics over preprocessed datasets."""

from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import Dataset

from tsfmx.types import PreprocessedSample


def text_embedding_mean(dataset: Dataset[PreprocessedSample]) -> torch.Tensor:
    """Average every text embedding in `dataset`, over patches as well as samples.

    Patches are averaged alongside samples because the fusion layer sees one embedding per patch
    and treats them alike; a per-patch mean would instead encode position within the context.

    Args:
        dataset: Preprocessed samples carrying `text_embeddings`.

    Returns:
        Mean embedding of shape (text_embedding_dims,).

    Raises:
        ValueError: If no sample in the dataset carries text embeddings.
    """
    total: torch.Tensor | None = None
    count = 0
    for i in range(len(cast(Sized, dataset))):
        sample = dataset[i]
        if "text_embeddings" not in sample:
            continue
        embeddings = torch.as_tensor(sample["text_embeddings"], dtype=torch.float32)
        flattened = embeddings.reshape(-1, embeddings.shape[-1])
        total = flattened.sum(dim=0) if total is None else total + flattened.sum(dim=0)
        count += flattened.shape[0]

    if total is None or count == 0:
        raise ValueError("Dataset holds no text embeddings to average.")
    return total / count
