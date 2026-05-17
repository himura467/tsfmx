"""Cross-validation utilities for Time-MMD dataset."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset, Dataset

from tsfmx.data.dataset import PreprocessedDataset
from tsfmx.data.preprocess import PreprocessPipeline
from tsfmx.types import PreprocessedSample, TrainingMode


@dataclass
class DomainSpec:
    """Pairs a domain name with its augmentation flag.

    Attributes:
        name: Domain name (e.g., 'Agriculture_train').
        augment: Whether to load the augmented cache for this domain.
    """

    name: str
    augment: bool = field(default=False)


def load_fold_datasets(
    train_domain_specs: list[DomainSpec],
    val_domain_specs: list[DomainSpec],
    test_domain_specs: list[DomainSpec],
    text_encoder_type: Literal["english", "japanese"],
    patch_len: int,
    context_len: int,
    horizon_len: int,
    cache_dir: Path,
    mode: TrainingMode,
) -> tuple[ConcatDataset[PreprocessedSample], ConcatDataset[PreprocessedSample], ConcatDataset[PreprocessedSample]]:
    """Load cached datasets for a single fold from pre-computed cache.

    Args:
        train_domain_specs: Domain specs for training.
        val_domain_specs: Domain specs for validation.
        test_domain_specs: Domain specs for testing.
        text_encoder_type: Type of text encoder used for caching.
        patch_len: Length of input patches.
        context_len: Length of context.
        horizon_len: Length of horizon.
        cache_dir: Directory containing pre-computed cached datasets.
        mode: Training mode passed to PreprocessedDataset.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    cache = PreprocessPipeline(cache_dir)

    def load_cached_domains(domain_specs: list[DomainSpec]) -> list[Dataset[PreprocessedSample]]:
        datasets: list[Dataset[PreprocessedSample]] = []
        for spec in domain_specs:
            cache_path = cache.get_path(
                dataset_name="time_mmd",
                entity=spec.name,
                text_encoder_type=text_encoder_type,
                patch_len=patch_len,
                context_len=context_len,
                horizon_len=horizon_len,
                augment=spec.augment,
            )
            data = cache.load(cache_path)
            datasets.append(PreprocessedDataset(data, mode=mode))
        return datasets

    return (
        ConcatDataset(load_cached_domains(train_domain_specs)),
        ConcatDataset(load_cached_domains(val_domain_specs)),
        ConcatDataset(load_cached_domains(test_domain_specs)),
    )
