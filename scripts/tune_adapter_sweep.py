#!/usr/bin/env python3
"""Hyperparameter tuning for adapter (fine-tuned) time series forecasting with W&B Sweeps."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
import wandb
from torch.utils.data import DataLoader

from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import DomainSpec, load_fold_datasets
from tsfmx.data.collate import adapter_collate_fn
from tsfmx.decoder import MultimodalDecoder, MultimodalDecoderConfig
from tsfmx.evaluator import MultimodalEvaluator
from tsfmx.trainer import MultimodalTrainer
from tsfmx.training_args import TrainingArguments
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.tsfm.chronos import Chronos2Adapter
from tsfmx.tsfm.timesfm import TimesFM2p5Adapter
from tsfmx.types import AdapterCheckpoint, Batch
from tsfmx.utils.device import pin_memory, resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed
from tsfmx.utils.yaml import load_yaml

_logger = setup_logger()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a W&B Sweeps hyperparameter search for adapter time series forecasting.",
    )

    parser.add_argument("--sweep-id", type=str, help="Existing W&B sweep ID to join.")
    parser.add_argument("--sweep-config", type=str, help="Path to a W&B sweep YAML config file.")
    parser.add_argument("--count", type=int, help="Number of sweep runs for the agent to execute.")
    parser.add_argument("--model-config", type=str, help="Path to a model config YAML file.")
    parser.add_argument("--forecast-config", type=str, help="Path to a forecast config YAML file.")
    parser.add_argument(
        "--augment",
        nargs="*",
        choices=["train", "val", "test"],
        default=["train"],
        help="Splits to load from augmented cache.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache", help="Directory with pre-computed cached datasets."
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    return parser.parse_args()


def _create_adapter_model(model_config: ModelConfig, device: torch.device) -> MultimodalDecoder:
    """Build a MultimodalDecoder with a pretrained adapter for baseline fine-tuning.

    The fusion head is constructed from model_config but remains unused during
    baseline training; only the adapter parameters are fine-tuned.

    Args:
        model_config: Static model configuration (adapter repo, embedding dims).
        device: Device to load the model onto.

    Returns:
        MultimodalDecoder with a pretrained adapter ready for fine-tuning.
    """
    _logger.info(
        "Loading pretrained adapter from %s on %s",
        model_config.adapter.pretrained_repo,
        device,
    )
    adapter: TsfmAdapter
    match model_config.adapter.type:
        case "chronos":
            adapter = Chronos2Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case "timesfm":
            adapter = TimesFM2p5Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case _:
            raise NotImplementedError(f"Unsupported adapter type: {model_config.adapter.type!r}")
    if adapter.patch_len != model_config.adapter.patch_len:
        raise ValueError(
            f"adapter.patch_len ({adapter.patch_len}) does not match "
            f"model_config.adapter.patch_len ({model_config.adapter.patch_len}); "
            "the cached dataset was built with the config value — rebuild the cache or fix the config."
        )
    config = MultimodalDecoderConfig(
        text_embedding_dims=model_config.fusion.text_embedding_dims,
        num_fusion_layers=model_config.fusion.num_fusion_layers,
        fusion_hidden_dims=model_config.fusion.fusion_hidden_dims,
    )
    return MultimodalDecoder(adapter, config)


def _train_and_evaluate(
    run: wandb.Run,
    base_training_args: TrainingArguments,
    model_config: ModelConfig,
    forecast_config: ForecastConfig,
    train_domain_specs: list[DomainSpec],
    val_domain_specs: list[DomainSpec],
    test_domain_specs: list[DomainSpec],
    device: torch.device,
    cache_dir: Path,
) -> None:
    """Run one sweep trial: fine-tune the adapter and log metrics to W&B.

    Reads hyperparameters from the active W&B run config, fine-tunes the
    adapter, loads the best checkpoint, evaluates on the test set, and logs
    val/best_loss, test/mse, and test/mae.
    The checkpoint directory is removed after evaluation.

    Args:
        run: Active W&B run whose config provides this trial's hyperparameters.
        base_training_args: Base training arguments partially overridden by sweep config.
        model_config: Static model architecture configuration.
        forecast_config: Forecasting parameters (context / horizon lengths).
        train_domain_specs: Domain specs used for training.
        val_domain_specs: Domain specs used for validation.
        test_domain_specs: Domain specs used for test evaluation.
        device: Device to train and evaluate on.
        cache_dir: Directory containing pre-computed cached datasets.
    """
    config = run.config
    _logger.info("Starting sweep run %s with config: %s", run.id, dict(config))

    training_args = replace(
        base_training_args,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        adapter_learning_rate=config.learning_rate,
        adapter_lr_scheduler_type=config.lr_scheduler_type,
        adapter_warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    _logger.info(
        "Loading datasets — train: %s, val: %s, test: %s",
        train_domain_specs,
        val_domain_specs,
        test_domain_specs,
    )
    train_dataset, val_dataset, test_dataset = load_fold_datasets(
        train_domain_specs=train_domain_specs,
        val_domain_specs=val_domain_specs,
        test_domain_specs=test_domain_specs,
        text_encoder_type=model_config.fusion.text_encoder_type,
        patch_len=model_config.adapter.patch_len,
        context_len=forecast_config.context_len,
        horizon_len=forecast_config.horizon_len,
        cache_dir=cache_dir,
    )

    model = _create_adapter_model(model_config, device)

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mode="adapter",
        device=device,
        wandb_run=run,
    )

    trainer.train()

    best_checkpoint_path = training_args.checkpoint_dir / "best_model.pt"
    _logger.info("Loading best checkpoint from %s", best_checkpoint_path)
    checkpoint = cast(AdapterCheckpoint, torch.load(best_checkpoint_path, weights_only=True))
    best_val_loss = checkpoint["best_val_loss"]
    model.adapter.load_state_dict(checkpoint["adapter_state_dict"])

    test_dataloader = cast(
        DataLoader[Batch],
        DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=adapter_collate_fn,
            pin_memory=pin_memory(device),
        ),
    )

    _logger.info("Evaluating on test domains: %s", test_domain_specs)
    evaluator = MultimodalEvaluator(model, device)
    test_metrics = evaluator.evaluate(test_dataloader)

    _logger.info(
        "Run %s — best_val_loss: %.6f, test_mse: %.6f, test_mae: %.6f",
        run.id,
        best_val_loss,
        test_metrics["mse"],
        test_metrics["mae"],
    )
    run.log(
        {"val/best_loss": best_val_loss, "test/mse": test_metrics["mse"], "test/mae": test_metrics["mae"]},
        step=trainer.global_step,
    )

    checkpoint_dir = training_args.checkpoint_dir
    if checkpoint_dir.exists():
        _logger.info("Removing checkpoint directory %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)


def main() -> int:
    """Entry point: resolve the sweep ID and start the W&B agent.

    Returns:
        Exit code — 0 on success, 1 if neither --sweep-id nor
        --sweep-config is provided.
    """
    args = _parse_args()

    if args.model_config:
        model_config = ModelConfig.from_yaml(Path(args.model_config))
        _logger.info("Loaded model config from %s", args.model_config)
    else:
        model_config = ModelConfig()
        _logger.info("Using default ModelConfig")

    if args.forecast_config:
        forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config))
        _logger.info("Loaded forecast config from %s", args.forecast_config)
    else:
        forecast_config = ForecastConfig()
        _logger.info("Using default ForecastConfig")

    base_training_args = TrainingArguments(
        output_dir="outputs/sweeps/adapter",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="best",
        seed=args.seed,
    )

    if args.seed is not None:
        _logger.info("Setting random seed to %d", args.seed)
        set_seed(args.seed)

    # Selected for high-quality textual data (low NA rates) and sufficient numerical data points.
    augment_splits = set(args.augment)
    _train_domain_names = [
        "Agriculture_train",
        "Economy_train",
        "Environment_train",
        "Health_US_train",
        "Traffic_train",
    ]
    train_domain_specs = [DomainSpec(name=d, augment="train" in augment_splits) for d in _train_domain_names]
    _val_domain_names = ["Agriculture_val", "Economy_val", "Environment_val", "Health_US_val", "Traffic_val"]
    val_domain_specs = [DomainSpec(name=d, augment="val" in augment_splits) for d in _val_domain_names]
    _test_domain_names = ["Agriculture_test", "Economy_test", "Environment_test", "Health_US_test", "Traffic_test"]
    test_domain_specs = [DomainSpec(name=d, augment="test" in augment_splits) for d in _test_domain_names]

    device = resolve_device()
    _logger.info("Using device: %s", device)

    wandb_project = f"adapter-{model_config.adapter.type}-time-mmd"

    def _sweep_fn() -> None:
        """Execute a single sweep trial inside a W&B run context."""
        with wandb.init(project=wandb_project) as run:
            _train_and_evaluate(
                run=run,
                base_training_args=base_training_args,
                model_config=model_config,
                forecast_config=forecast_config,
                train_domain_specs=train_domain_specs,
                val_domain_specs=val_domain_specs,
                test_domain_specs=test_domain_specs,
                device=device,
                cache_dir=Path(args.cache_dir),
            )

    if args.sweep_id:
        sweep_id = args.sweep_id
        _logger.info("Joining existing sweep %s", sweep_id)
    else:
        if not args.sweep_config:
            _logger.error("Either --sweep-id or --sweep-config must be provided.")
            return 1
        sweep_config = load_yaml(Path(args.sweep_config))
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project)
        _logger.info("Created new sweep %s", sweep_id)

    _logger.info("Starting W&B agent (count=%s)", args.count)
    wandb.agent(sweep_id, function=_sweep_fn, project=wandb_project, count=args.count)
    _logger.info("Sweep agent finished")

    return 0


if __name__ == "__main__":
    exit(main())
