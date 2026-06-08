#!/usr/bin/env python3
"""Visualize model forecasts on Time-MMD train/val/test splits from a saved checkpoint."""

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import ConcatDataset, DataLoader

from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_fold_datasets
from tsfmx.data.collate import adapter_collate_fn, multimodal_collate_fn
from tsfmx.decoder import MultimodalDecoder, MultimodalDecoderConfig
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.tsfm.chronos import Chronos2Adapter
from tsfmx.tsfm.timesfm import TimesFM2p5Adapter
from tsfmx.types import Batch, TrainingMode
from tsfmx.utils.device import pin_memory, resolve_device
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


def _make_loader(
    dataset: ConcatDataset[Any],
    batch_size: int,
    collate_fn: Any,
    device: torch.device,
) -> DataLoader[Batch]:
    """Wrap a dataset in a DataLoader for inference.

    Args:
        dataset: Preprocessed dataset to wrap.
        batch_size: Number of samples per batch.
        collate_fn: Collate function for the training mode.
        device: Device used to determine whether to pin memory.

    Returns:
        DataLoader with shuffle disabled.
    """
    return cast(
        DataLoader[Batch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=pin_memory(device),
        ),
    )


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

    _logger.info("Loading adapter from %s", model_config.adapter.pretrained_repo)
    adapter: TsfmAdapter
    match model_config.adapter.type:
        case "chronos":
            adapter = Chronos2Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case "timesfm":
            adapter = TimesFM2p5Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case _ as t:
            raise NotImplementedError(f"Unsupported adapter type: {t!r}")

    decoder_config = MultimodalDecoderConfig(
        text_embedding_dims=model_config.fusion.text_embedding_dims,
        num_fusion_layers=model_config.fusion.num_fusion_layers,
        fusion_hidden_dims=model_config.fusion.fusion_hidden_dims,
    )
    model = MultimodalDecoder(adapter, decoder_config).to(device)
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

    collate_fn = multimodal_collate_fn if mode in ("fusion", "finetune") else adapter_collate_fn

    splits_to_visualize: set[str] = set(args.splits)
    train_loader = (
        _make_loader(train_dataset, args.batch_size, collate_fn, device) if "train" in splits_to_visualize else None
    )
    val_loader = (
        _make_loader(val_dataset, args.batch_size, collate_fn, device) if "val" in splits_to_visualize else None
    )
    test_loader = (
        _make_loader(test_dataset, args.batch_size, collate_fn, device) if "test" in splits_to_visualize else None
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
