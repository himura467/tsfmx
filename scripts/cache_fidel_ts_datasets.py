#!/usr/bin/env python3
"""Pre-compute and cache text embeddings for one Fidel-TS sub-dataset.

Text embeddings must be cached before training or evaluation. Each entity and split pair is
encoded once and persisted, so the training scripts load them without re-encoding.
"""

import argparse
from pathlib import Path
from typing import Literal

from examples.fidel_ts.configs.dataset import FidelTsConfig
from examples.fidel_ts.data.fidel_ts_dataset import FidelTsDataset, Split
from examples.time_mmd.builders import build_text_encoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from tsfmx.data.preprocess import PreprocessPipeline
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed

_logger = setup_logger()

_SPLITS: tuple[Split, ...] = ("train", "val", "test")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Pre-compute and cache text embeddings for Fidel-TS entities.")

    parser.add_argument("--model-config", type=str, help="Path to a model config YAML file.")
    parser.add_argument("--forecast-config", type=str, help="Path to a forecast config YAML file.")
    parser.add_argument("--dataset-config", type=str, help="Path to a Fidel-TS dataset config YAML file.")
    parser.add_argument(
        "--text-encoder-type",
        type=str,
        choices=["english", "japanese"],
        required=True,
        help="Text encoder to use for embedding generation.",
    )
    parser.add_argument("--data-root", type=str, default="data/Fidel-TS", help="Directory holding the sub-datasets.")
    parser.add_argument(
        "--entities",
        type=str,
        nargs="+",
        help="Subset of entities to cache. Defaults to every entity with a time series file.",
    )
    parser.add_argument("--splits", type=str, nargs="+", choices=list(_SPLITS), default=list(_SPLITS))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="Apply patch-boundary shift augmentation.")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Directory to write cached datasets.")
    parser.add_argument("--force-rebuild", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    return parser.parse_args()


def main() -> int:
    """Entry point: cache text embeddings for every requested entity and split.

    Returns:
        Exit code. 0 on success.

    Raises:
        FileNotFoundError: If the sub-dataset directory does not exist.
    """
    args = _parse_args()
    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()
    dataset_config = FidelTsConfig.from_yaml(Path(args.dataset_config)) if args.dataset_config else FidelTsConfig()

    if args.seed is not None:
        _logger.info("Setting random seed to %d", args.seed)
        set_seed(args.seed)

    text_encoder_type: Literal["english", "japanese"] = args.text_encoder_type
    device = resolve_device()
    _logger.info("Using device: %s", device)

    text_encoder = build_text_encoder(text_encoder_type, device)

    data_dir = Path(args.data_root) / dataset_config.name
    if not data_dir.exists():
        raise FileNotFoundError(f"Sub-dataset not found: {data_dir}. Run scripts/download_fidel_ts.sh first.")

    entities = args.entities or FidelTsDataset.list_entities(data_dir)
    _logger.info("Caching %d entities of %s: %s", len(entities), dataset_config.name, entities)

    pipeline = PreprocessPipeline(Path(args.cache_dir))

    for entity in entities:
        for split in args.splits:
            _logger.info("Processing entity %s split %s", entity, split)
            cache_path = pipeline.get_path(
                # The cache is keyed per entity and split, matching how the Time-MMD cache keys
                # its per-domain splits, so the same loader can read either dataset.
                dataset_name="fidel_ts",
                entity=f"{entity}_{split}",
                text_encoder_type=text_encoder_type,
                patch_len=model_config.adapter.patch_len,
                context_len=forecast_config.context_len,
                horizon_len=forecast_config.horizon_len,
                augment=args.augment,
            )

            def _dataset_factory(entity: str = entity, split: Split = split) -> FidelTsDataset:
                return FidelTsDataset(
                    data_dir=data_dir,
                    entity=entity,
                    target_column=dataset_config.target_column,
                    text_sources=dataset_config.text_sources,
                    patch_len=model_config.adapter.patch_len,
                    context_len=forecast_config.context_len,
                    horizon_len=forecast_config.horizon_len,
                    timestamp_column=dataset_config.timestamp_column,
                    augment=args.augment,
                    split=split,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )

            pipeline.prepare(
                path=cache_path,
                dataset_factory=_dataset_factory,
                text_encoder=text_encoder,
                device=device,
                force_rebuild=args.force_rebuild,
            )

    _logger.info("Caching complete")
    return 0


if __name__ == "__main__":
    exit(main())
