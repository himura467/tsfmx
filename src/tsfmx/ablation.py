"""Text ablations for measuring how much a trained model relies on text."""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import override

from tsfmx.types import PreprocessedSample
from tsfmx.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy.typing as npt

_logger = get_logger()

TextAblation = Literal["none", "drop", "shuffle", "permute_patches", "noise"]

TEXT_ABLATIONS: tuple[TextAblation, ...] = ("none", "drop", "shuffle", "permute_patches", "noise")


def _derangement(size: int, seed: int) -> npt.NDArray[np.int64]:
    """Build a permutation of range(size) that leaves no element in place.

    A plain random permutation would leave roughly one sample paired with its own text,
    which weakens the ablation. Swapping each fixed point with its successor can never
    introduce a new one, because the successor cannot already hold the swapped-in value.

    Args:
        size: Number of elements to permute.
        seed: Seed for the permutation.

    Returns:
        Array of shape (size,) where result[i] != i for every i.

    Raises:
        ValueError: If size is less than 2, where no derangement exists.
    """
    if size < 2:
        raise ValueError(f"A derangement requires at least 2 elements, got {size}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(size)
    for i in range(size):
        if perm[i] == i:
            j = (i + 1) % size
            perm[i], perm[j] = perm[j], perm[i]
    return perm.astype(np.int64)


class TextAblatedDataset(Dataset[PreprocessedSample]):
    """Wraps a preprocessed dataset and perturbs the text side of every sample.

    Beating a unimodal baseline does not prove that a model reads its text: the fusion
    branch can also act as a plain regularizer. Comparing metrics across these ablations
    separates those explanations. `context` and `horizon` are always passed through, so
    any change in metrics is attributable to the text alone.

    - drop: skips fusion entirely, so batches must be collated with `adapter_collate_fn`.
    - shuffle: isolates whether the model uses the content of the text or its mere presence.
    - permute_patches: destroys only the temporal alignment, leaving the content intact.
    - noise: grades the degradation instead of breaking the text outright.

    Perturbations are keyed on the sample index rather than applied per batch, so results
    do not depend on batch size or iteration order.

    Args:
        dataset: Dataset of preprocessed samples to wrap. Must implement __len__.
        ablation: Which perturbation to apply.
        seed: Seed for the derangement and the per-sample noise and patch permutations.
        noise_scale: Multiplier on the embedding std for the 'noise' ablation.
        std_sample_size: Number of samples used to estimate the embedding std.

    Raises:
        ValueError: If ablation is unknown, if the wrapped dataset is empty, or if an
            ablation that needs text is applied to samples without text_embeddings.
    """

    def __init__(
        self,
        dataset: Dataset[PreprocessedSample],
        ablation: TextAblation,
        seed: int = 0,
        noise_scale: float = 1.0,
        std_sample_size: int = 256,
    ) -> None:
        if ablation not in TEXT_ABLATIONS:
            raise ValueError(f"Unknown text ablation: {ablation!r}. Expected one of {list(TEXT_ABLATIONS)}")

        self.dataset = dataset
        self.ablation = ablation
        self.seed = seed
        self.noise_scale = noise_scale
        self.std_sample_size = std_sample_size

        self._len = len(cast(Sized, dataset))
        if self._len == 0:
            raise ValueError("Cannot apply a text ablation to an empty dataset")

        self._validate()

        self._perm = _derangement(self._len, seed) if ablation == "shuffle" else None
        self._noise_std = self._estimate_noise_std() if ablation == "noise" else 0.0

    def _validate(self) -> None:
        if self.ablation in ("none", "drop"):
            return

        probe = self.dataset[0]
        if "text_embeddings" not in probe:
            raise ValueError(f"Ablation {self.ablation!r} requires samples with 'text_embeddings'")

        if self.ablation == "shuffle" and self._len < 2:
            raise ValueError("Ablation 'shuffle' requires at least 2 samples to pair text across")

        if self.ablation == "permute_patches":
            num_patches = probe["text_embeddings"].shape[0]
            if num_patches < 2:
                _logger.warning(
                    "Ablation 'permute_patches' is a no-op with %d text patch; "
                    "increase context_len relative to patch_len for it to be meaningful",
                    num_patches,
                )

    def _estimate_noise_std(self) -> float:
        """Estimate the embedding standard deviation, to put the injected noise on the same scale.

        Returns:
            Standard deviation over the first std_sample_size samples.
        """
        num_samples = min(self.std_sample_size, self._len)
        embeddings = np.stack([self.dataset[i]["text_embeddings"] for i in range(num_samples)])
        std = float(np.std(embeddings))
        _logger.info("Estimated text embedding std over %d samples: %.6f", num_samples, std)
        return std

    def __len__(self) -> int:
        return self._len

    def _ablate_embeddings(self, index: int, embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the configured perturbation to one sample's text embeddings.

        Args:
            index: Index of the sample, used to seed a per-sample generator.
            embeddings: Text embeddings of shape (num_patches, text_dims).

        Returns:
            Perturbed embeddings of the same shape and dtype.
        """
        rng = np.random.default_rng([self.seed, index])
        match self.ablation:
            case "permute_patches":
                return embeddings[rng.permutation(embeddings.shape[0])]
            case "noise":
                noise = rng.normal(0.0, self._noise_std * self.noise_scale, size=embeddings.shape)
                return (embeddings + noise).astype(embeddings.dtype)
            case _:
                return embeddings

    @override
    def __getitem__(self, index: int) -> PreprocessedSample:
        sample = self.dataset[index]
        result = PreprocessedSample(
            context=sample["context"],
            horizon=sample["horizon"],
            metadata={**sample["metadata"], "text_ablation": self.ablation},
        )

        if self.ablation == "drop":
            return result

        if self._perm is not None:  # non-None exactly for the 'shuffle' ablation
            result["text_embeddings"] = self.dataset[int(self._perm[index])]["text_embeddings"]
            return result

        if "text_embeddings" in sample:
            result["text_embeddings"] = self._ablate_embeddings(index, sample["text_embeddings"])
        return result
