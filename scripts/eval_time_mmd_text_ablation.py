#!/usr/bin/env python3
"""Evaluate a tsfmx checkpoint on Time-MMD test splits under text ablations and write results to JSON."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_fold_datasets
from tsfmx.ablation import TEXT_ABLATIONS, TextAblatedDataset, TextAblation
from tsfmx.data.collate import adapter_collate_fn, collate_fn_for_mode
from tsfmx.data.loader import build_dataloader
from tsfmx.evaluator import MultimodalEvaluator
from tsfmx.types import PreprocessedSample, TrainingMode
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed

_logger = setup_logger()

_DEFAULT_DOMAINS = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a tsfmx checkpoint under text ablations.")

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--forecast-config", type=str)
    parser.add_argument("--domains", nargs="+", default=_DEFAULT_DOMAINS)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(TEXT_ABLATIONS),
        default=list(TEXT_ABLATIONS),
        help="Which ablations to run. 'none' is always included as the reference.",
    )
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output", type=str, default="outputs/text_ablation_results.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Multiplier on the embedding std for 'noise'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the ablation perturbations.")

    return parser.parse_args()


def _order_ablations(ablations: list[str]) -> list[TextAblation]:
    """Return the requested ablations with 'none' first, deduplicated.

    Args:
        ablations: Ablation names as given on the command line.

    Returns:
        Ordered list of ablations beginning with 'none', which the deltas are measured against.
    """
    requested = set(ablations) | {"none"}
    return [a for a in TEXT_ABLATIONS if a in requested]


def _percent_delta(value: float, reference: float) -> float:
    """Return the change of value against reference as a percentage."""
    if reference == 0.0:
        return float("nan")
    return (value - reference) / reference * 100.0


def _evaluate_domain(
    evaluator: MultimodalEvaluator,
    test_dataset: ConcatDataset[PreprocessedSample],
    ablations: list[TextAblation],
    mode: TrainingMode,
    batch_size: int,
    seed: int,
    noise_scale: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate one domain's test split under every requested ablation.

    Args:
        evaluator: Evaluator wrapping the loaded model.
        test_dataset: Test split for this domain.
        ablations: Ablations to run, with 'none' first.
        mode: Training mode of the loaded checkpoint.
        batch_size: Evaluation batch size.
        seed: Seed for the ablation perturbations.
        noise_scale: Multiplier on the embedding std for the 'noise' ablation.
        device: Device to run inference on.

    Returns:
        Mapping from ablation name to its metrics and percentage deltas against 'none'.
    """
    domain_results: dict[str, dict[str, float]] = {}
    reference: dict[str, float] | None = None

    for ablation in ablations:
        ablated = TextAblatedDataset(test_dataset, ablation, seed=seed, noise_scale=noise_scale)
        # 'drop' strips text_embeddings, which the multimodal collate function requires.
        collate_fn = adapter_collate_fn if ablation == "drop" else collate_fn_for_mode(mode)
        metrics = evaluator.evaluate(build_dataloader(ablated, batch_size, collate_fn, device))

        entry = {"mse": metrics["mse"], "mae": metrics["mae"]}
        if reference is None:
            reference = dict(entry)
        entry["mse_delta_pct"] = _percent_delta(entry["mse"], reference["mse"])
        entry["mae_delta_pct"] = _percent_delta(entry["mae"], reference["mae"])
        domain_results[ablation] = entry

    return domain_results


def _macro_average(
    results: dict[str, dict[str, dict[str, float]]], ablations: list[TextAblation]
) -> dict[str, dict[str, float]]:
    """Average each ablation's metrics across all evaluated domains.

    Args:
        results: Per-domain results.
        ablations: Ablations that were run.

    Returns:
        Macro-averaged metrics, with deltas recomputed from the averaged values.
    """
    macro: dict[str, dict[str, float]] = {}
    reference: dict[str, float] | None = None

    for ablation in ablations:
        per_domain = [r[ablation] for r in results.values() if ablation in r]
        entry = {
            "mse": sum(d["mse"] for d in per_domain) / len(per_domain),
            "mae": sum(d["mae"] for d in per_domain) / len(per_domain),
        }
        if reference is None:
            reference = dict(entry)
        entry["mse_delta_pct"] = _percent_delta(entry["mse"], reference["mse"])
        entry["mae_delta_pct"] = _percent_delta(entry["mae"], reference["mae"])
        macro[ablation] = entry

    return macro


def _log_table(results: dict[str, dict[str, dict[str, float]]], ablations: list[TextAblation]) -> None:
    """Log the results as a fixed-width table, one row per domain and ablation."""
    _logger.info("%-14s %-16s %10s %10s %10s %10s", "domain", "ablation", "mse", "mae", "mse_d%", "mae_d%")
    for domain, per_ablation in results.items():
        for ablation in ablations:
            if ablation not in per_ablation:
                continue
            entry = per_ablation[ablation]
            _logger.info(
                "%-14s %-16s %10.6f %10.6f %+10.2f %+10.2f",
                domain,
                ablation,
                entry["mse"],
                entry["mae"],
                entry["mse_delta_pct"],
                entry["mae_delta_pct"],
            )


def main() -> int:
    """Entry point: load a checkpoint and evaluate it under each text ablation.

    Returns:
        Exit code. 0 on success.

    Raises:
        ValueError: If the checkpoint was trained in 'adapter' mode, which never uses text.
        RuntimeError: If no domain could be evaluated.
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
        raise ValueError(
            "Checkpoint was trained in 'adapter' mode, which never consumes text. "
            "Text ablations only apply to 'fusion' and 'finetune' checkpoints."
        )

    ablations = _order_ablations(args.ablations)
    _logger.info("Running ablations: %s", ablations)

    evaluator = MultimodalEvaluator(model, device)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for domain in args.domains:
        try:
            _, _, test_dataset = load_fold_datasets(
                train_domain_specs=[DomainSpec(name=f"{domain}_train")],
                val_domain_specs=[DomainSpec(name=f"{domain}_val")],
                test_domain_specs=[DomainSpec(name=f"{domain}_test")],
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

        results[domain] = _evaluate_domain(
            evaluator, test_dataset, ablations, mode, args.batch_size, args.seed, args.noise_scale, device
        )

    if not results:
        raise RuntimeError("No domains could be evaluated; check --domains and --cache-dir.")

    results["macro"] = _macro_average(results, ablations)
    _log_table(results, ablations)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info("Results written to %s", output_path)
    return 0


if __name__ == "__main__":
    exit(main())
