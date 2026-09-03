"""Text ablations for measuring how much a trained model relies on text."""

from __future__ import annotations

from collections.abc import Callable, Sized
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import override

from tsfmx.types import PreprocessedSample
from tsfmx.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy.typing as npt

_logger = get_logger()

TextAblation = Literal[
    "none",
    "drop",
    "mean",
    "shuffle",
    "cross_domain",
    "permute_patches",
    "noise",
    "oracle",
    "oracle_trend",
]

TEXT_ABLATIONS: tuple[TextAblation, ...] = (
    "none",
    "drop",
    "mean",
    "shuffle",
    "cross_domain",
    "permute_patches",
    "noise",
    "oracle",
    "oracle_trend",
)

#: Ablations that synthesize new text and therefore need a text encoder.
ORACLE_ABLATIONS: tuple[TextAblation, ...] = ("oracle", "oracle_trend")

#: Text encoding callable: maps a list of strings to embeddings of shape (len(texts), text_dims).
TextEncodeFn = Callable[[list[str]], "npt.NDArray[np.float32]"]


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


def _oracle_sentence(sample: PreprocessedSample) -> str:
    """Write out the sample's own horizon as a sentence.

    The horizon is stored already normalized, so the numbers are on one scale across domains
    and are exactly what the model is asked to output.

    Args:
        sample: Sample whose horizon is described.

    Returns:
        Sentence listing every horizon value.
    """
    horizon = sample["horizon"]
    values = ", ".join(f"{v:.2f}" for v in horizon)
    return f"The next {len(horizon)} values of the series are: {values}."


def _oracle_trend_sentence(sample: PreprocessedSample) -> str:
    """Describe the sample's own horizon in words, without stating any number.

    Sentence encoders represent magnitude poorly, so a failure of `_oracle_sentence` alone is
    ambiguous: the fusion branch may be unable to carry sample-specific information, or the
    encoder may simply be unable to read a list of floats. This carries the same information
    in words instead, which such an encoder does represent.

    Thresholds are in units of the context standard deviation, which normalization fixes at 1.

    Args:
        sample: Sample whose horizon is described.

    Returns:
        Sentence naming the direction and the volatility of the horizon.
    """
    horizon = sample["horizon"]
    context = sample["context"]
    delta = float(horizon.mean() - context[-1])
    volatility = float(horizon.std())

    if delta < -1.0:
        direction = "falls sharply"
    elif delta < -0.25:
        direction = "falls"
    elif delta <= 0.25:
        direction = "stays roughly flat"
    elif delta <= 1.0:
        direction = "rises"
    else:
        direction = "rises sharply"

    if volatility < 0.25:
        steadiness = "steady"
    elif volatility < 1.0:
        steadiness = "moderately variable"
    else:
        steadiness = "highly volatile"

    return f"Over the next {len(horizon)} steps the series {direction} and is {steadiness}."


class TextAblatedDataset(Dataset[PreprocessedSample]):
    """Wraps a preprocessed dataset and perturbs the text side of every sample.

    Beating a unimodal baseline does not prove that a model reads its text: the fusion
    branch can also act as a plain regularizer. Comparing metrics across these ablations
    separates those explanations. `context` and `horizon` are always passed through, so
    any change in metrics is attributable to the text alone.

    - drop: skips fusion entirely, so batches must be collated with `adapter_collate_fn`.
    - mean: removes between-sample variation while keeping a plausible input. Matching `none`
      means the fusion output carries no sample-specific information and has collapsed to a
      learned constant, which `drop` and `shuffle` together can only suggest.
    - shuffle: isolates whether the model uses the content of the text or its mere presence.
    - cross_domain: pairs each sample with another domain's text by position. Where `shuffle`
      leaves the domain intact, this destroys it too, separating text the model reads from
      text it only uses to recognize which domain a sample came from.
    - permute_patches: destroys only the temporal alignment, leaving the content intact.
    - noise: grades the degradation instead of breaking the text outright.
    - oracle: replaces the text with the sample's own future, written out as numbers.
    - oracle_trend: replaces it with the same future described in words instead.

    The oracles run the comparison in the other direction: every other ablation degrades the
    text and asks whether the forecast notices, while these hand the model text that is by
    construction worth reading. No improvement means the model cannot use text at all,
    whatever the corpus says; an improvement places the fault in the corpus instead. They
    consume the labels, so neither is a forecasting result.

    Perturbations are keyed on the sample index rather than applied per batch, so results
    do not depend on batch size or iteration order.

    Args:
        dataset: Dataset of preprocessed samples to wrap. Must implement __len__.
        ablation: Which perturbation to apply.
        seed: Seed for the derangement and the per-sample noise and patch permutations.
        noise_scale: Multiplier on the embedding std for the 'noise' ablation.
        std_sample_size: Number of samples used to estimate the embedding std.
        donor: Dataset supplying replacement text for 'cross_domain', normally another
            domain's split. Samples are paired by position, cycling if it is shorter.
        encode: Text encoder callable required by the oracle ablations.
        encode_batch_size: Number of sentences encoded per call to `encode`.

    Raises:
        ValueError: If ablation is unknown, if the wrapped dataset is empty, if an ablation
            that needs text is applied to samples without text_embeddings, or if the inputs
            an ablation requires (`donor`, `encode`) are missing or shaped inconsistently.
    """

    def __init__(
        self,
        dataset: Dataset[PreprocessedSample],
        ablation: TextAblation,
        seed: int = 0,
        noise_scale: float = 1.0,
        std_sample_size: int = 256,
        donor: Dataset[PreprocessedSample] | None = None,
        encode: TextEncodeFn | None = None,
        encode_batch_size: int = 256,
    ) -> None:
        if ablation not in TEXT_ABLATIONS:
            raise ValueError(f"Unknown text ablation: {ablation!r}. Expected one of {list(TEXT_ABLATIONS)}")

        self.dataset = dataset
        self.ablation = ablation
        self.seed = seed
        self.noise_scale = noise_scale
        self.std_sample_size = std_sample_size
        self.donor = donor
        self.encode = encode
        self.encode_batch_size = encode_batch_size

        self._len = len(cast(Sized, dataset))
        if self._len == 0:
            raise ValueError("Cannot apply a text ablation to an empty dataset")
        self._donor_len = len(cast(Sized, donor)) if donor is not None else 0

        self._validate()

        self._perm = _derangement(self._len, seed) if ablation == "shuffle" else None
        self._mean_embeddings = self._compute_mean_embeddings() if ablation == "mean" else None
        self._noise_std = self._estimate_noise_std() if ablation == "noise" else 0.0
        self._oracle_embeddings = self._compute_oracle_embeddings() if ablation in ORACLE_ABLATIONS else None

    def _validate(self) -> None:
        if self.ablation in ("none", "drop"):
            return

        probe = self.dataset[0]
        if "text_embeddings" not in probe:
            raise ValueError(f"Ablation {self.ablation!r} requires samples with 'text_embeddings'")

        if self.ablation == "shuffle" and self._len < 2:
            raise ValueError("Ablation 'shuffle' requires at least 2 samples to pair text across")

        if self.ablation == "cross_domain":
            if self.donor is None:
                raise ValueError("Ablation 'cross_domain' requires a donor dataset to take text from")
            if self._donor_len == 0:
                raise ValueError("Ablation 'cross_domain' requires a non-empty donor dataset")
            donor_shape = self.donor[0]["text_embeddings"].shape
            if donor_shape != probe["text_embeddings"].shape:
                raise ValueError(
                    f"Donor text embeddings of shape {donor_shape} do not match "
                    f"the wrapped dataset's {probe['text_embeddings'].shape}; "
                    "both splits must be cached with the same text encoder, patch_len and context_len."
                )

        if self.ablation in ORACLE_ABLATIONS and self.encode is None:
            raise ValueError(f"Ablation {self.ablation!r} synthesizes text and requires an 'encode' callable")

        if self.ablation == "permute_patches":
            num_patches = probe["text_embeddings"].shape[0]
            if num_patches < 2:
                _logger.warning(
                    "Ablation 'permute_patches' is a no-op with %d text patch; "
                    "increase context_len relative to patch_len for it to be meaningful",
                    num_patches,
                )

    def _compute_mean_embeddings(self) -> npt.NDArray[np.float32]:
        """Average the text embeddings over every sample, keeping the patch axis.

        Every sample is used rather than a subsample, so that the result is the exact
        quantity the ablation claims to substitute.

        Returns:
            Mean embeddings of shape (num_patches, text_dims).
        """
        total = np.zeros_like(self.dataset[0]["text_embeddings"], dtype=np.float64)
        for i in range(self._len):
            total += self.dataset[i]["text_embeddings"]
        _logger.info("Computed mean text embeddings over %d samples", self._len)
        return (total / self._len).astype(np.float32)

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

    def _compute_oracle_embeddings(self) -> npt.NDArray[np.float32]:
        """Encode one oracle sentence per sample, describing that sample's own horizon.

        Identical sentences are encoded once and shared, which costs nothing for 'oracle' but
        collapses 'oracle_trend' to its handful of distinct phrasings.

        Returns:
            Embeddings of shape (num_samples, text_dims), one row per sample.

        Raises:
            RuntimeError: If the encoder returns a row count or a dimension that does not
                match the sentences it was given or the cached embeddings it must replace.
        """
        if self.encode is None:
            raise RuntimeError("_compute_oracle_embeddings requires an 'encode' callable")

        write = _oracle_sentence if self.ablation == "oracle" else _oracle_trend_sentence
        sentences = [write(self.dataset[i]) for i in range(self._len)]

        unique: dict[str, int] = {}
        index = np.empty(self._len, dtype=np.int64)
        for i, sentence in enumerate(sentences):
            index[i] = unique.setdefault(sentence, len(unique))
        distinct = list(unique)
        _logger.info("Encoding %d distinct oracle sentences for %d samples", len(distinct), self._len)

        encoded = [
            self.encode(distinct[start : start + self.encode_batch_size])
            for start in range(0, len(distinct), self.encode_batch_size)
        ]
        embeddings = np.concatenate(encoded).astype(np.float32)
        if embeddings.shape[0] != len(distinct):
            raise RuntimeError(f"Encoder returned {embeddings.shape[0]} embeddings for {len(distinct)} sentences")

        text_dims = self.dataset[0]["text_embeddings"].shape[-1]
        if embeddings.shape[-1] != text_dims:
            raise RuntimeError(
                f"Encoder produced {embeddings.shape[-1]}-dimensional embeddings, but the cached "
                f"text embeddings are {text_dims}-dimensional; the ablation must use the encoder "
                "the cache was built with."
            )
        return embeddings[index]

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

        if self._mean_embeddings is not None:  # non-None exactly for the 'mean' ablation
            result["text_embeddings"] = self._mean_embeddings
            return result

        if self._perm is not None:  # non-None exactly for the 'shuffle' ablation
            result["text_embeddings"] = self.dataset[int(self._perm[index])]["text_embeddings"]
            return result

        if self._oracle_embeddings is not None:  # non-None exactly for the oracle ablations
            # One sentence describes the whole horizon, so every patch receives it. Tiling keeps
            # the (num_patches, text_dims) shape the cached embeddings have, leaving fusion and
            # collation unaware that this sample's text was synthesized.
            num_patches = sample["text_embeddings"].shape[0]
            result["text_embeddings"] = np.tile(self._oracle_embeddings[index], (num_patches, 1))
            return result

        if self.ablation == "cross_domain" and self.donor is not None:  # _validate guarantees the donor
            result["text_embeddings"] = self.donor[index % self._donor_len]["text_embeddings"]
            return result

        if "text_embeddings" in sample:
            result["text_embeddings"] = self._ablate_embeddings(index, sample["text_embeddings"])
        return result
