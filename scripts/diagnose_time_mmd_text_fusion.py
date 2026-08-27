#!/usr/bin/env python3
"""Measure how much sample-specific signal survives the text encoder and the fusion projection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_split_dataset
from tsfmx.data.collate import multimodal_collate_fn
from tsfmx.data.loader import build_dataloader
from tsfmx.decoder import MultimodalDecoder
from tsfmx.types import Batch, TrainingMode
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed

_logger = setup_logger()

_DEFAULT_DOMAINS = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose collapse of the text fusion branch.")

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--forecast-config", type=str)
    parser.add_argument("--domains", nargs="+", default=_DEFAULT_DOMAINS)
    parser.add_argument("--augment", action="store_true", help="Read the test split from the augmented cache.")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output", type=str, default="outputs/text_fusion_diagnostics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cosine-sample-size",
        type=int,
        default=256,
        help="Cap on samples used for the pairwise cosine similarity, which is quadratic in this count.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class _Moments:
    """Accumulates per-element mean and standard deviation over a stream of equally shaped arrays."""

    def __init__(self) -> None:
        self._count = 0
        self._total: np.ndarray | None = None
        self._total_sq: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        """Add a batch of samples, whose first axis is the sample axis."""
        batch = values.astype(np.float64)
        if self._total is None or self._total_sq is None:
            self._total = batch.sum(axis=0)
            self._total_sq = (batch**2).sum(axis=0)
        else:
            self._total += batch.sum(axis=0)
            self._total_sq += (batch**2).sum(axis=0)
        self._count += batch.shape[0]

    def summary(self) -> tuple[float, float]:
        """Reduce the accumulated moments to two scalars.

        Returns:
            Tuple of (mean_abs, between_sample_std). mean_abs is the mean absolute value of the
            per-element mean, which measures the component shared by every sample. between_sample_std
            is the mean per-element standard deviation across samples, which measures the component
            that varies with the sample.

        Raises:
            RuntimeError: If no samples were accumulated.
        """
        if self._total is None or self._total_sq is None or self._count == 0:
            raise RuntimeError("No samples accumulated")
        mean = self._total / self._count
        variance = np.maximum(self._total_sq / self._count - mean**2, 0.0)
        return float(np.mean(np.abs(mean))), float(np.mean(np.sqrt(variance)))


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Average cosine similarity between every pair of distinct samples.

    Args:
        embeddings: Array of shape (num_samples, ...), flattened per sample before comparison.

    Returns:
        Mean off-diagonal cosine similarity. Values near 1 mean the encoder maps different
        text to nearly the same vector, so no fusion mechanism could separate them.
    """
    flat = embeddings.reshape(embeddings.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normalized = flat / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    off_diagonal = ~np.eye(similarities.shape[0], dtype=bool)
    return float(similarities[off_diagonal].mean())


def _diagnose_domain(
    model: MultimodalDecoder,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cosine_sample_size: int,
) -> dict[str, float]:
    """Measure the text embeddings, the fusion projection, and the time series embeddings.

    Args:
        model: Decoder holding the trained fusion head and the pretrained adapter.
        dataloader: Loader over one domain's split.
        device: Device to run the forward passes on.
        cosine_sample_size: Cap on samples retained for the pairwise cosine similarity.

    Returns:
        Mapping of metric name to value.
    """
    text_moments = _Moments()
    projection_moments = _Moments()
    ts_moments = _Moments()
    retained: list[np.ndarray] = []
    retained_count = 0

    with torch.no_grad():
        for batch in dataloader:
            text_embeddings = batch["text_embeddings"].to(device)
            context = batch["context"].to(device)
            masks = torch.zeros_like(context, dtype=torch.bool)

            projected = model.fusion.projection(text_embeddings)
            ts_embeddings = model.adapter.preprocess(context, masks).input_embeddings

            text_moments.update(text_embeddings.cpu().numpy())
            projection_moments.update(projected.cpu().numpy())
            ts_moments.update(ts_embeddings.cpu().numpy())

            if retained_count < cosine_sample_size:
                take = min(cosine_sample_size - retained_count, text_embeddings.shape[0])
                retained.append(text_embeddings[:take].cpu().numpy())
                retained_count += take

    text_mean_abs, text_std = text_moments.summary()
    projection_mean_abs, projection_std = projection_moments.summary()
    ts_mean_abs, _ = ts_moments.summary()

    return {
        "text_mean_abs": text_mean_abs,
        "text_between_sample_std": text_std,
        "text_variation_ratio": text_std / max(text_mean_abs, 1e-12),
        "text_mean_pairwise_cosine": _mean_pairwise_cosine(np.concatenate(retained)),
        "projection_mean_abs": projection_mean_abs,
        "projection_between_sample_std": projection_std,
        "projection_variation_ratio": projection_std / max(projection_mean_abs, 1e-12),
        "ts_mean_abs": ts_mean_abs,
        "constant_vs_ts": projection_mean_abs / max(ts_mean_abs, 1e-12),
        "varying_vs_ts": projection_std / max(ts_mean_abs, 1e-12),
    }


def _log_table(results: dict[str, dict[str, float]]) -> None:
    """Log the per-domain metrics as a fixed-width table."""
    _logger.info(
        "%-14s %10s %10s %10s %10s %10s",
        "domain",
        "text_cos",
        "text_var",
        "proj_var",
        "const/ts",
        "vary/ts",
    )
    for domain, m in results.items():
        _logger.info(
            "%-14s %10.4f %10.4f %10.4f %10.4f %10.4f",
            domain,
            m["text_mean_pairwise_cosine"],
            m["text_variation_ratio"],
            m["projection_variation_ratio"],
            m["constant_vs_ts"],
            m["varying_vs_ts"],
        )


def main() -> int:
    """Entry point: load a checkpoint and measure its text and fusion statistics per domain.

    Returns:
        Exit code. 0 on success.

    Raises:
        ValueError: If the checkpoint was trained in 'adapter' mode, which has no fusion head.
        RuntimeError: If no domain could be diagnosed.
    """
    args = _parse_args()
    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()

    set_seed(args.seed)

    device = resolve_device()
    _logger.info("Using device: %s", device)

    model = build_decoder(model_config, device)
    mode: TrainingMode = model.load_checkpoint(Path(args.checkpoint_path))
    model.eval()
    _logger.info("Loaded %s-mode checkpoint from %s", mode, args.checkpoint_path)

    if mode == "adapter":
        raise ValueError("Checkpoint was trained in 'adapter' mode, which has no trained fusion head to diagnose.")

    results: dict[str, dict[str, float]] = {}
    num_samples: dict[str, int] = {}

    for domain in args.domains:
        try:
            test_dataset = load_split_dataset(
                domain_specs=[DomainSpec(name=f"{domain}_test", augment=args.augment)],
                text_encoder_type=model_config.fusion.text_encoder_type,
                patch_len=model_config.adapter.patch_len,
                context_len=forecast_config.context_len,
                horizon_len=forecast_config.horizon_len,
                cache_dir=Path(args.cache_dir),
                mode=mode,
            )
        except Exception as e:
            _logger.warning("Skipping %s: %s", domain, e)
            continue

        num_samples[domain] = len(test_dataset)
        _logger.info("%s: %d test samples", domain, num_samples[domain])
        dataloader = build_dataloader(test_dataset, args.batch_size, multimodal_collate_fn, device)
        results[domain] = _diagnose_domain(model, dataloader, device, args.cosine_sample_size)

    if not results:
        raise RuntimeError("No domains could be diagnosed; check --domains and --cache-dir.")

    _log_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"num_samples": num_samples, "metrics": results}, f, indent=2)
    _logger.info("Diagnostics written to %s", output_path)
    return 0


if __name__ == "__main__":
    exit(main())
