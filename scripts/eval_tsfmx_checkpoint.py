#!/usr/bin/env python3
"""Evaluate a tsfmx checkpoint on Time-MMD test splits and write per-domain MSE/MAE to JSON."""

import argparse
import json
from pathlib import Path

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from tsfmx.data.splits import DomainSpec, load_fold_datasets
from tsfmx.data.collate import collate_fn_for_mode
from tsfmx.data.loader import build_dataloader
from tsfmx.evaluator import MultimodalEvaluator
from tsfmx.types import TrainingMode
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger

_logger = setup_logger()

_DEFAULT_DOMAINS = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--forecast-config", type=str)
    parser.add_argument("--domains", nargs="+", default=_DEFAULT_DOMAINS)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output", type=str, default="outputs/tsfmx_eval_results.json")
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument(
        "--dataset",
        type=str,
        default="time_mmd",
        help="Name the cache was built under: 'time_mmd', or 'fidel_ts' for a Fidel-TS sub-dataset.",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()

    device = resolve_device()
    model = build_decoder(model_config, device)

    mode: TrainingMode = model.load_checkpoint(Path(args.checkpoint_path))
    model.eval()

    collate_fn = collate_fn_for_mode(mode)
    evaluator = MultimodalEvaluator(model, device)
    results: dict[str, dict[str, float]] = {}

    for domain in args.domains:
        try:
            _, _, test_dataset = load_fold_datasets(
                dataset_name=args.dataset,
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

        metrics = evaluator.evaluate(build_dataloader(test_dataset, args.batch_size, collate_fn, device))
        results[domain] = {"mse": metrics["mse"], "mae": metrics["mae"]}
        _logger.info("%s — MSE: %.6f  MAE: %.6f", domain, metrics["mse"], metrics["mae"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info("Results written to %s", output_path)
    return 0


if __name__ == "__main__":
    exit(main())
