# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""Training arguments for multimodal time series forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tsfmx.utils.yaml import parse_yaml


@dataclass(frozen=True)
class TrainingArguments:
    # --- Output ---
    output_dir: str = field(default="outputs", metadata={"help": "The output directory."})

    # --- Training Duration and Batch Size ---
    per_device_train_batch_size: int = field(default=8, metadata={"help": "The batch size per device for training."})
    num_train_epochs: int = field(default=10, metadata={"help": "Total number of training epochs to perform."})

    # --- Learning Rate & Scheduler ---
    fusion_learning_rate: float = field(
        default=1e-4, metadata={"help": "The initial learning rate for the fusion optimizer."}
    )
    fusion_lr_scheduler_type: Literal["linear", "cosine"] = field(
        default="linear", metadata={"help": "The learning rate scheduler type for the fusion optimizer."}
    )
    fusion_warmup_steps: float = field(
        default=0.0, metadata={"help": "Warmup steps for the fusion optimizer (int or ratio)."}
    )
    adapter_learning_rate: float = field(
        default=1e-5, metadata={"help": "The initial learning rate for the adapter optimizer."}
    )
    adapter_lr_scheduler_type: Literal["linear", "cosine"] = field(
        default="linear", metadata={"help": "The learning rate scheduler type for the adapter optimizer."}
    )
    adapter_warmup_steps: float = field(
        default=0.0, metadata={"help": "Warmup steps for the adapter optimizer (int or ratio)."}
    )

    # --- Optimizer ---
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay coefficient applied by the optimizer."})

    # --- Regularization & Training Stability ---
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of update steps to accumulate gradients before performing a backward/update pass."},
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping. Set to 0 to disable."}
    )

    # --- Logging & Monitoring ---
    logging_strategy: Literal["no", "epoch", "steps"] = field(
        default="steps",
        metadata={"help": "The logging strategy to adopt during training."},
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Number of update steps between two logs if `logging_strategy='steps'`."}
    )

    # --- Experiment Tracking ---
    run_name: str | None = field(default=None, metadata={"help": "A descriptor for the run."})

    # --- Evaluation ---
    eval_strategy: Literal["no", "epoch", "steps"] = field(
        default="no",
        metadata={"help": "When to run evaluation."},
    )
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "The batch size per device for evaluation."})

    # --- Checkpointing & Saving ---
    save_strategy: Literal["no", "epoch", "steps", "best"] = field(
        default="steps", metadata={"help": "The checkpoint save strategy to adopt during training."}
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. The best checkpoint is always retained when `load_best_model_at_end=True`."
        },
    )

    # --- Best Model Tracking ---
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": "Load the best checkpoint at the end of training. Requires `eval_strategy` to be set. When enabled, the best checkpoint is always saved (see `save_total_limit`)."
        },
    )

    # --- Reproducibility ---
    seed: int | None = field(
        default=None, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def logging_dir(self) -> Path:
        """Directory for logs."""
        return Path(self.output_dir) / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for model checkpoints."""
        return Path(self.output_dir) / "checkpoints"

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> TrainingArguments:
        """Load from YAML file."""
        return parse_yaml(Path(yaml_path), cls)

    def get_warmup_steps(self, num_training_steps: int, warmup_steps: float) -> int:
        """Compute warmup steps from an int count or a float ratio of total steps.

        Args:
            num_training_steps: Total number of training steps.
            warmup_steps: If < 1, interpreted as a ratio of `num_training_steps`. If >= 1, must be an
                integer count of warmup steps.

        Returns:
            Number of warmup steps.

        Raises:
            ValueError: If `warmup_steps` is negative, or >= 1 but not an integer value.
        """
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps={warmup_steps}. Expected a non-negative value.")
        if warmup_steps < 1:
            return math.ceil(num_training_steps * warmup_steps)
        if not float(warmup_steps).is_integer():
            raise ValueError(
                f"Invalid warmup_steps={warmup_steps}. When warmup_steps >= 1 it must be an integer "
                "number of steps or a float numerically equal to an integer (e.g., 100.0)."
            )
        return int(warmup_steps)
