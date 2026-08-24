#!/usr/bin/env python3
"""Visualize model forecasts on Time-MMD train/val/test splits from a saved checkpoint."""

import argparse
from pathlib import Path

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_fold_datasets
from tsfmx.data.collate import collate_fn_for_mode
from tsfmx.data.loader import build_dataloader
from tsfmx.types import TrainingMode
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.visualizer import PredictionVisualizer

_logger = setup_logger()

_DEFAULT_DOMAINS = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Time-MMD forecasts from a saved checkpoint.",
    )

    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--model-config", type=str, help="Path to a model config YAML.")
    parser.add_argument("--forecast-config", type=str, help="Path to a forecast config YAML.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to plot per split.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Dataset splits to visualize.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=_DEFAULT_DOMAINS,
        help="Domain names without split suffix (e.g., Agriculture Economy).",
    )
    parser.add_argument(
        "--augment",
        nargs="*",
        choices=["train", "val", "test"],
        default=[],
        help="Splits to load from augmented cache.",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache", help="Directory with pre-computed cached datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory where plot PNG files are saved.",
    )

    return parser.parse_args()


def main() -> int:
    """Entry point: load checkpoint, datasets, and generate forecast plots.

    Returns:
        Exit code — 0 on success.
    """
    args = _parse_args()

    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()

    device = resolve_device()
    _logger.info("Using device: %s", device)

    model = build_decoder(model_config, device)
    checkpoint_path = Path(args.checkpoint_path)
    _logger.info("Loading checkpoint from %s", checkpoint_path)
    mode: TrainingMode = model.load_checkpoint(checkpoint_path)
    model.eval()

    augment_splits = set(args.augment)
    domains: list[str] = args.domains

    train_domain_specs = [DomainSpec(name=f"{d}_train", augment="train" in augment_splits) for d in domains]
    val_domain_specs = [DomainSpec(name=f"{d}_val", augment="val" in augment_splits) for d in domains]
    test_domain_specs = [DomainSpec(name=f"{d}_test", augment="test" in augment_splits) for d in domains]

    _logger.info("Loading datasets for domains: %s", domains)
    train_dataset, val_dataset, test_dataset = load_fold_datasets(
        train_domain_specs=train_domain_specs,
        val_domain_specs=val_domain_specs,
        test_domain_specs=test_domain_specs,
        text_encoder_type=model_config.fusion.text_encoder_type,
        patch_len=model_config.adapter.patch_len,
        context_len=forecast_config.context_len,
        horizon_len=forecast_config.horizon_len,
        cache_dir=Path(args.cache_dir),
        mode=mode,
    )

    collate_fn = collate_fn_for_mode(mode)

    splits_to_visualize: set[str] = set(args.splits)
    train_loader = (
        build_dataloader(train_dataset, args.batch_size, collate_fn, device) if "train" in splits_to_visualize else None
    )
    val_loader = (
        build_dataloader(val_dataset, args.batch_size, collate_fn, device) if "val" in splits_to_visualize else None
    )
    test_loader = (
        build_dataloader(test_dataset, args.batch_size, collate_fn, device) if "test" in splits_to_visualize else None
    )

    output_dir = Path(args.output_dir)
    _logger.info("Saving plots to %s", output_dir)

    visualizer = PredictionVisualizer(
        model=model,
        device=device,
        figsize=(16, 9),
        max_samples=args.max_samples,
        output_dir=output_dir,
    )
    results = visualizer.visualize_all_splits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        show=args.show,
    )

    for split, figs in results.items():
        _logger.info("%s: %d plots saved to %s", split, len(figs), output_dir / split)

    return 0


if __name__ == "__main__":
    exit(main())
