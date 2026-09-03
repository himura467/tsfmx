#!/usr/bin/env python3
"""Measure how much sample-specific signal survives the text encoder and the fusion projection."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
        help="Per-domain cap on samples used for the cosine similarities, which are quadratic in this count.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class _Moments:
    """Accumulates per-element mean and standard deviation over a stream of equally shaped arrays."""

    def __init__(self) -> None:
        self._count = 0
        self._total: npt.NDArray[np.float64] | None = None
        self._total_sq: npt.NDArray[np.float64] | None = None

    def update(self, values: npt.NDArray[np.float32]) -> None:
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
        """Reduce the accumulated moments to two magnitudes.

        Returns:
            Tuple of (constant_rms, varying_rms): the magnitude of the component every sample
            shares, and of the component that differs between samples. Both are root mean
            squares over elements, so they are orthogonal parts of the stream's total
            magnitude — hypot(constant_rms, varying_rms) — and either can be compared against
            another stream through that single denominator.

        Raises:
            RuntimeError: If no samples were accumulated.
        """
        if self._total is None or self._total_sq is None or self._count == 0:
            raise RuntimeError("No samples accumulated")
        mean = self._total / self._count
        variance = np.maximum(self._total_sq / self._count - mean**2, 0.0)
        return float(np.sqrt(np.mean(mean**2))), float(np.sqrt(np.mean(variance)))


def _unit_rows(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Flatten each sample to a vector and rescale it to unit length.

    Args:
        embeddings: Array of shape (num_samples, ...).

    Returns:
        Array of shape (num_samples, num_features) whose rows have unit norm.
    """
    flat = embeddings.reshape(embeddings.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norms, 1e-12)


def _mean_cosine(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64] | None) -> float:
    """Average cosine similarity between two sets of unit-norm rows, or within one set.

    Args:
        a: Unit-norm rows.
        b: Second set of unit-norm rows, or None to compare `a` against itself, excluding each
            row's similarity with itself.

    Returns:
        Mean cosine similarity, or NaN for a within-set comparison of fewer than two rows.
        Values near 1 within one domain mean the encoder maps different text to nearly the same
        vector, so no fusion mechanism could separate them.
    """
    if b is not None:
        return float((a @ b.T).mean())
    if a.shape[0] < 2:
        return float("nan")
    similarities = a @ a.T
    return float(similarities[~np.eye(similarities.shape[0], dtype=bool)].mean())


def _diagnose_domain(
    model: MultimodalDecoder,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cosine_sample_size: int,
) -> tuple[dict[str, float], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Measure the text embeddings, the fusion projection, and the time series embeddings.

    The projection is reported relative to the *total* magnitude of the time series embeddings
    fusion adds it to. Dividing by their shared component instead inflates every ratio, because
    time series embeddings vary strongly between samples and share correspondingly little.

    Args:
        model: Decoder holding the trained fusion head and the pretrained adapter.
        dataloader: Loader over one domain's split.
        device: Device to run the forward passes on.
        cosine_sample_size: Cap on samples retained for the cosine similarities.

    Returns:
        Tuple of (metrics, retained text embeddings, retained projection outputs). The retained
        arrays hold up to cosine_sample_size samples for the cross-domain comparison.
    """
    text_moments = _Moments()
    projection_moments = _Moments()
    ts_moments = _Moments()
    retained_text: list[npt.NDArray[np.float32]] = []
    retained_projection: list[npt.NDArray[np.float32]] = []
    retained_count = 0

    with torch.no_grad():
        for batch in dataloader:
            text_embeddings = batch["text_embeddings"].to(device)
            context = batch["context"].to(device)
            masks = torch.zeros_like(context, dtype=torch.bool)

            projected = model.fusion.projection(text_embeddings)
            ts_embeddings = model.adapter.preprocess(context, masks).input_embeddings

            text_batch = text_embeddings.cpu().numpy()
            projection_batch = projected.cpu().numpy()
            text_moments.update(text_batch)
            projection_moments.update(projection_batch)
            ts_moments.update(ts_embeddings.cpu().numpy())

            if retained_count < cosine_sample_size:
                take = min(cosine_sample_size - retained_count, text_batch.shape[0])
                retained_text.append(text_batch[:take])
                retained_projection.append(projection_batch[:take])
                retained_count += take

    text_constant, text_varying = text_moments.summary()
    projection_constant, projection_varying = projection_moments.summary()
    ts_constant, ts_varying = ts_moments.summary()
    text_total = math.hypot(text_constant, text_varying)
    projection_total = math.hypot(projection_constant, projection_varying)
    ts_total = max(math.hypot(ts_constant, ts_varying), 1e-12)

    metrics = {
        "text_constant_rms": text_constant,
        "text_varying_rms": text_varying,
        "text_varying_fraction": text_varying / max(text_total, 1e-12),
        "projection_constant_rms": projection_constant,
        "projection_varying_rms": projection_varying,
        "projection_varying_fraction": projection_varying / max(projection_total, 1e-12),
        "ts_constant_rms": ts_constant,
        "ts_varying_rms": ts_varying,
        "constant_vs_ts_rms": projection_constant / ts_total,
        "varying_vs_ts_rms": projection_varying / ts_total,
        "projection_vs_ts_rms": projection_total / ts_total,
    }
    return metrics, np.concatenate(retained_text), np.concatenate(retained_projection)


def _cross_domain_stats(retained: dict[str, npt.NDArray[np.float32]]) -> dict[str, Any]:
    """Measure how far apart the domains sit in one representation.

    Text can only act as a domain identifier if its embeddings separate the domains at all.
    Separability, the mean within-domain cosine minus the mean cross-domain cosine, decides
    that: at 0 samples of different domains sit as close together as samples of the same one,
    so nothing downstream could recover which domain a sample came from.

    Args:
        retained: Mapping from domain to that domain's retained samples, flattened per sample
            before comparison.

    Returns:
        Mapping with the domain-by-domain mean cosine matrix and the within, cross, and
        separability summary values.
    """
    unit = {domain: _unit_rows(samples) for domain, samples in retained.items()}
    pairwise = {
        a: {b: _mean_cosine(rows_a, None if a == b else rows_b) for b, rows_b in unit.items()}
        for a, rows_a in unit.items()
    }

    domains = list(pairwise)
    within = [pairwise[d][d] for d in domains]
    cross = [pairwise[a][b] for a in domains for b in domains if a != b]
    within_mean = float(np.mean(within)) if within else float("nan")
    cross_mean = float(np.mean(cross)) if cross else float("nan")

    return {
        "mean_pairwise_cosine": pairwise,
        "within_domain_mean": within_mean,
        "cross_domain_mean": cross_mean,
        "separability": within_mean - cross_mean,
    }


def _log_scale_table(results: dict[str, dict[str, float]]) -> None:
    """Log the per-domain magnitudes as a fixed-width table."""
    _logger.info(
        "%-14s %10s %10s %10s %10s %10s %10s %10s",
        "domain",
        "text_cos",
        "text_vary",
        "proj_cos",
        "proj_vary",
        "const/ts",
        "vary/ts",
        "proj/ts",
    )
    for domain, m in results.items():
        _logger.info(
            "%-14s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
            domain,
            m["text_mean_pairwise_cosine"],
            m["text_varying_fraction"],
            m["projection_mean_pairwise_cosine"],
            m["projection_varying_fraction"],
            m["constant_vs_ts_rms"],
            m["varying_vs_ts_rms"],
            m["projection_vs_ts_rms"],
        )


def _log_cross_domain(name: str, stats: dict[str, Any]) -> None:
    """Log one representation's cross-domain cosine matrix and its separability summary."""
    matrix: dict[str, dict[str, float]] = stats["mean_pairwise_cosine"]
    domains = list(matrix)
    _logger.info("%s mean pairwise cosine between domains (diagonal is within-domain)", name)
    _logger.info("%-14s %s", "domain", " ".join(f"{d[:10]:>10}" for d in domains))
    for a in domains:
        _logger.info("%-14s %s", a, " ".join(f"{matrix[a][b]:>10.4f}" for b in domains))
    _logger.info(
        "%s within=%.4f cross=%.4f separability=%.4f",
        name,
        stats["within_domain_mean"],
        stats["cross_domain_mean"],
        stats["separability"],
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
    retained_text: dict[str, npt.NDArray[np.float32]] = {}
    retained_projection: dict[str, npt.NDArray[np.float32]] = {}

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
        results[domain], retained_text[domain], retained_projection[domain] = _diagnose_domain(
            model, dataloader, device, args.cosine_sample_size
        )

    if not results:
        raise RuntimeError("No domains could be diagnosed; check --domains and --cache-dir.")

    cross_domain = {
        "text": _cross_domain_stats(retained_text),
        "projection": _cross_domain_stats(retained_projection),
    }
    # The within-domain similarity is the diagonal of the same matrix, so it is read back here
    # rather than computed a second time.
    for domain, m in results.items():
        m["text_mean_pairwise_cosine"] = cross_domain["text"]["mean_pairwise_cosine"][domain][domain]
        m["projection_mean_pairwise_cosine"] = cross_domain["projection"]["mean_pairwise_cosine"][domain][domain]

    _log_scale_table(results)
    _log_cross_domain("text", cross_domain["text"])
    _log_cross_domain("projection", cross_domain["projection"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"num_samples": num_samples, "metrics": results, "cross_domain": cross_domain}, f, indent=2)
    _logger.info("Diagnostics written to %s", output_path)
    return 0


if __name__ == "__main__":
    exit(main())
