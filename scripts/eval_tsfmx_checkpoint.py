#!/usr/bin/env python3
"""Evaluate a tsfmx checkpoint on Time-MMD test splits and write per-domain MSE/MAE to JSON."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import ConcatDataset, DataLoader

from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_fold_datasets
from tsfmx.data.collate import adapter_collate_fn, multimodal_collate_fn
from tsfmx.decoder import MultimodalDecoder, MultimodalDecoderConfig
from tsfmx.evaluator import MultimodalEvaluator
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.tsfm.chronos import Chronos2Adapter
from tsfmx.tsfm.timesfm import TimesFM2p5Adapter
from tsfmx.types import Batch, TrainingMode
from tsfmx.utils.device import pin_memory, resolve_device
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

    return parser.parse_args()


def _make_loader(
    dataset: ConcatDataset[Any], batch_size: int, collate_fn: Any, device: torch.device
) -> DataLoader[Batch]:
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
    args = _parse_args()
    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()

    device = resolve_device()
    adapter: TsfmAdapter
    match model_config.adapter.type:
        case "chronos":
            adapter = Chronos2Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case "timesfm":
            adapter = TimesFM2p5Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case _ as t:
            raise NotImplementedError(f"Unsupported adapter type: {t!r}")

    model = MultimodalDecoder(
        adapter,
        MultimodalDecoderConfig(
            text_embedding_dims=model_config.fusion.text_embedding_dims,
            num_fusion_layers=model_config.fusion.num_fusion_layers,
            fusion_hidden_dims=model_config.fusion.fusion_hidden_dims,
        ),
    ).to(device)

    mode: TrainingMode = model.load_checkpoint(Path(args.checkpoint_path))
    model.eval()

    collate_fn = multimodal_collate_fn if mode in ("fusion", "finetune") else adapter_collate_fn
    evaluator = MultimodalEvaluator(model, device)
    results: dict[str, dict[str, float]] = {}

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

        metrics = evaluator.evaluate(_make_loader(test_dataset, args.batch_size, collate_fn, device))
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
