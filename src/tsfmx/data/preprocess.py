"""Preprocessing pipeline for multimodal and baseline datasets."""

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch

from tsfmx.data.dataset import MultimodalDatasetBase
from tsfmx.text_encoder.base import TextEncoderBase
from tsfmx.types import PreprocessedSample
from tsfmx.utils.logging import get_logger

_logger = get_logger()


class PreprocessPipeline:
    """End-to-end preprocessing pipeline: path generation, persistence, and execution."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _head_parts(dataset_name: str) -> list[str]:
        """Build the filename components that precede the entity segment."""
        return [dataset_name]

    @staticmethod
    def _tail_parts(
        text_encoder_type: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        augment: bool,
        text_tokens: bool,
    ) -> list[str]:
        """Build the filename components that follow the entity segment."""
        parts = [text_encoder_type, f"p{patch_len}", f"c{context_len}", f"h{horizon_len}"]
        if augment:
            parts.append("aug")
        if text_tokens:
            parts.append("tok")
        return parts

    def get_path(
        self,
        dataset_name: str,
        entity: str,
        text_encoder_type: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        augment: bool = False,
        text_tokens: bool = False,
    ) -> Path:
        """Generate a unique file path for the given configuration.

        Args:
            dataset_name: Name of the dataset (e.g. 'time_mmd').
            entity: Domain or split-domain identifier (e.g. 'Environment_train').
            text_encoder_type: Text encoder type string.
            patch_len: Patch length.
            context_len: Context length.
            horizon_len: Horizon length.
            augment: Whether patch-boundary shift augmentation was applied.
            text_tokens: Whether the cache also holds per-token text embeddings.

        Returns:
            Path to the cache file.
        """
        parts = [
            *self._head_parts(dataset_name),
            entity,
            *self._tail_parts(text_encoder_type, patch_len, context_len, horizon_len, augment, text_tokens),
        ]
        return self.cache_dir / ("_".join(parts) + ".pkl")

    def find_entities(
        self,
        dataset_name: str,
        text_encoder_type: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        augment: bool = False,
        text_tokens: bool = False,
    ) -> set[str]:
        """Discover entity names that have a cached file matching the given configuration.

        Args:
            dataset_name: Name of the dataset (e.g. 'time_mmd').
            text_encoder_type: Text encoder type string.
            patch_len: Patch length.
            context_len: Context length.
            horizon_len: Horizon length.
            augment: Whether patch-boundary shift augmentation was applied.
            text_tokens: Whether the cache also holds per-token text embeddings.

        Returns:
            Set of entity strings found in the cache directory.
        """
        prefix = "_".join(self._head_parts(dataset_name)) + "_"
        suffix = "_" + "_".join(
            self._tail_parts(text_encoder_type, patch_len, context_len, horizon_len, augment, text_tokens)
        )
        entities: set[str] = set()
        for path in self.cache_dir.glob(f"{prefix}*{suffix}.pkl"):
            entity = path.stem.removeprefix(prefix).removesuffix(suffix)
            entities.add(entity)
        return entities

    def load(self, path: Path) -> list[PreprocessedSample]:
        _logger.info("Loading preprocessed data from %s", path)
        with open(path, "rb") as f:
            data: list[PreprocessedSample] = pickle.load(f)
        _logger.info("Loaded %s samples", len(data))
        return data

    def _save(self, path: Path, data: list[PreprocessedSample]) -> None:
        _logger.info("Saving %s samples to %s", len(data), path)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024 * 1024)
        _logger.info("Saved %.2f MB", size_mb)

    @staticmethod
    def _encode_tokens(
        dataset: MultimodalDatasetBase, text_encoder: TextEncoderBase, batch_size: int = 256
    ) -> dict[str, npt.NDArray[np.float16]]:
        """Encode each distinct patch text of the dataset into token embeddings once.

        Overlapping windows, and augmentation above all, repeat every report many times over.
        Samples share the returned arrays, which pickle then stores once, so the cache grows with
        the number of reports rather than of samples. Half precision halves it again.

        Args:
            dataset: Raw dataset whose patch texts to encode.
            text_encoder: Encoder producing the token embeddings.
            batch_size: Number of texts encoded per call.

        Returns:
            Token embeddings of shape (num_tokens, text_dims) keyed by patch text. The empty text
            maps to zero tokens, so that fusion sees an empty patch as one without text.
        """
        distinct = list(
            dict.fromkeys(
                " ".join(patch) for i in range(len(dataset)) for patch in dataset[i]["patched_texts"] if patch
            )
        )
        _logger.info("Encoding tokens of %d distinct patch texts", len(distinct))
        tokens_by_text = {"": np.zeros((0, text_encoder.embedding_dim), dtype=np.float16)}
        for start in range(0, len(distinct), batch_size):
            texts = distinct[start : start + batch_size]
            for text, tokens in zip(texts, text_encoder.encode_tokens(texts)):
                tokens_by_text[text] = tokens.cpu().numpy().astype(np.float16)
        return tokens_by_text

    def _preprocess(
        self,
        dataset: MultimodalDatasetBase,
        text_encoder: TextEncoderBase | None,
        device: torch.device | None,
        text_tokens: bool,
    ) -> list[PreprocessedSample]:
        _logger.info(
            "Preprocessing %s samples (%s)",
            len(dataset),
            "with text" if text_encoder is not None else "without text",
        )
        if text_encoder is not None:
            if device is None:
                raise ValueError("device must be provided when text_encoder is specified")
            text_encoder.eval()
        tokens_by_text = (
            self._encode_tokens(dataset, text_encoder) if text_tokens and text_encoder is not None else None
        )
        result: list[PreprocessedSample] = []
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                entry = PreprocessedSample(
                    context=sample["context"],
                    horizon=sample["horizon"],
                    metadata=sample["metadata"],
                )
                if text_encoder is not None and device is not None:
                    texts = [" ".join(patch) if patch else "" for patch in sample["patched_texts"]]
                    embeddings = text_encoder(texts)
                    entry["text_embeddings"] = embeddings.cpu().numpy()
                    if tokens_by_text is not None:
                        entry["text_tokens"] = [tokens_by_text[text] for text in texts]
                result.append(entry)
                if (i + 1) % 100 == 0:
                    _logger.info("Preprocessed %s/%s samples", i + 1, len(dataset))
        _logger.info("Preprocessing complete")
        return result

    def prepare(
        self,
        path: Path,
        dataset_factory: Callable[[], MultimodalDatasetBase],
        text_encoder: TextEncoderBase | None = None,
        device: torch.device | None = None,
        force_rebuild: bool = False,
        text_tokens: bool = False,
    ) -> list[PreprocessedSample]:
        """Load preprocessed data from disk, or create and save it if absent.

        Args:
            path: Target file path.
            dataset_factory: Callable that returns a raw MultimodalDatasetBase.
            text_encoder: Text encoder for fusion/finetune modes. None for adapter mode.
            device: Required when text_encoder is provided.
            force_rebuild: Rebuild even if file exists.
            text_tokens: Also store per-token text embeddings, which cross-attention fusion reads.

        Returns:
            List of preprocessed samples.
        """
        if not force_rebuild and path.exists():
            return self.load(path)

        dataset = dataset_factory()
        data = self._preprocess(dataset, text_encoder, device, text_tokens)
        self._save(path, data)
        return data
