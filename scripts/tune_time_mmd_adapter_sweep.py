#!/usr/bin/env python3
"""Hyperparameter tuning for adapter fine-tuning with W&B Sweeps."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
import wandb

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from tsfmx.data.splits import DomainSpec, load_fold_datasets
from tsfmx.data.collate import adapter_collate_fn
from tsfmx.data.loader import build_dataloader
from tsfmx.evaluator import MultimodalEvaluator
from tsfmx.trainer import MultimodalTrainer
from tsfmx.training_args import TrainingArguments
from tsfmx.types import AdapterCheckpoint
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed
from tsfmx.utils.yaml import load_yaml

_logger = setup_logger()

# Selected for high-quality textual data (low NA rates) and sufficient numerical data points.
_DEFAULT_ENTITIES = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a W&B Sweeps hyperparameter search for adapter fine-tuning on Time-MMD.",
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
    parser.add_argument(
        "--best-checkpoint-dir",
        type=str,
        default="outputs/sweeps/adapter/best_checkpoints",
        help="Directory to save the best cross-trial checkpoints. Not coordinated across multiple agents.",
    )
    parser.add_argument(
        "--keep-best-val-loss",
        action="store_true",
        help="Retain the cross-trial checkpoint with the lowest val_loss as best_val_loss.pt.",
    )
    parser.add_argument(
        "--keep-best-test-mse",
        action="store_true",
        help="Retain the cross-trial checkpoint with the lowest test MSE as best_test_mse.pt.",
    )
    parser.add_argument(
        "--keep-best-test-mae",
        action="store_true",
        help="Retain the cross-trial checkpoint with the lowest test MAE as best_test_mae.pt.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="time_mmd",
        help="Name the cache was built under: 'time_mmd', or 'fidel_ts' for a Fidel-TS sub-dataset.",
    )
    parser.add_argument(
        "--entities",
        type=str,
        nargs="+",
        default=_DEFAULT_ENTITIES,
        help="Entities to train on. Each is suffixed with _train, _val and _test to name its cached splits.",
    )

    return parser.parse_args()


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
    dataset_name: str,
    best_state: dict[str, float],
    best_checkpoint_dir: Path,
    keep_best_val_loss: bool,
    keep_best_test_mse: bool,
    keep_best_test_mae: bool,
) -> None:
    """Run one sweep trial: fine-tune the adapter and log metrics to W&B.

    Reads hyperparameters from the active W&B run config, fine-tunes the
    adapter, loads the best checkpoint, evaluates on the test set, and logs
    val/best_loss, test/mse, and test/mae.
    After evaluation, updates cross-trial best checkpoints for whichever
    metrics are enabled before the trial directory is removed.

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
        dataset_name: Name the cache was built under.
        best_state: Mutable dict tracking the best val_loss, test_mse, and test_mae seen so far.
        best_checkpoint_dir: Directory where per-metric best checkpoints are written.
        keep_best_val_loss: Whether to retain the cross-trial best val_loss checkpoint.
        keep_best_test_mse: Whether to retain the cross-trial best test_mse checkpoint.
        keep_best_test_mae: Whether to retain the cross-trial best test_mae checkpoint.
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
        dataset_name=dataset_name,
        train_domain_specs=train_domain_specs,
        val_domain_specs=val_domain_specs,
        test_domain_specs=test_domain_specs,
        text_encoder_type=model_config.fusion.text_encoder_type,
        patch_len=model_config.adapter.patch_len,
        context_len=forecast_config.context_len,
        horizon_len=forecast_config.horizon_len,
        cache_dir=cache_dir,
        mode="adapter",
    )

    model = build_decoder(model_config, device)

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

    trial_best_checkpoint_path = training_args.checkpoint_dir / "best_model.pt"
    _logger.info("Loading best checkpoint from %s", trial_best_checkpoint_path)
    checkpoint = cast(AdapterCheckpoint, torch.load(trial_best_checkpoint_path, weights_only=True))
    best_val_loss = checkpoint["best_val_loss"]
    model.adapter.load_state_dict(checkpoint["adapter_state_dict"])

    test_dataloader = build_dataloader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=adapter_collate_fn,
        device=device,
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

    if keep_best_val_loss and best_val_loss < best_state["val_loss"]:
        best_state["val_loss"] = best_val_loss
        dest = best_checkpoint_dir / "best_val_loss.pt"
        shutil.copy(trial_best_checkpoint_path, dest)
        _logger.info("New best adapter checkpoint (val_loss) saved to %s (val_loss=%.6f)", dest, best_val_loss)

    if keep_best_test_mse and test_metrics["mse"] < best_state["test_mse"]:
        best_state["test_mse"] = test_metrics["mse"]
        checkpoint["best_test_mse"] = test_metrics["mse"]
        dest = best_checkpoint_dir / "best_test_mse.pt"
        torch.save(checkpoint, dest)
        _logger.info("New best adapter checkpoint (test_mse) saved to %s (test_mse=%.6f)", dest, test_metrics["mse"])

    if keep_best_test_mae and test_metrics["mae"] < best_state["test_mae"]:
        best_state["test_mae"] = test_metrics["mae"]
        checkpoint["best_test_mae"] = test_metrics["mae"]
        dest = best_checkpoint_dir / "best_test_mae.pt"
        torch.save(checkpoint, dest)
        _logger.info("New best adapter checkpoint (test_mae) saved to %s (test_mae=%.6f)", dest, test_metrics["mae"])

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

    augment_splits = set(args.augment)
    train_domain_specs = [DomainSpec(name=f"{e}_train", augment="train" in augment_splits) for e in args.entities]
    val_domain_specs = [DomainSpec(name=f"{e}_val", augment="val" in augment_splits) for e in args.entities]
    test_domain_specs = [DomainSpec(name=f"{e}_test", augment="test" in augment_splits) for e in args.entities]

    device = resolve_device()
    _logger.info("Using device: %s", device)

    wandb_project = f"adapter-{model_config.adapter.type}-time-mmd"

    best_state: dict[str, float] = {"val_loss": float("inf"), "test_mse": float("inf"), "test_mae": float("inf")}
    best_checkpoint_dir = Path(args.best_checkpoint_dir)
    if args.keep_best_val_loss or args.keep_best_test_mse or args.keep_best_test_mae:
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                dataset_name=args.dataset,
                best_state=best_state,
                best_checkpoint_dir=best_checkpoint_dir,
                keep_best_val_loss=args.keep_best_val_loss,
                keep_best_test_mse=args.keep_best_test_mse,
                keep_best_test_mae=args.keep_best_test_mae,
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
