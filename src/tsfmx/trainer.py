"""Trainer for multimodal and baseline time series forecasting."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset, DataLoader

from tsfmx.data.collate import baseline_collate_fn, multimodal_collate_fn
from tsfmx.decoder import MultimodalDecoder
from tsfmx.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from tsfmx.training_args import TrainingArguments
from tsfmx.types import (
    BaselineCheckpoint,
    Batch,
    FinetuneCheckpoint,
    MultimodalCheckpoint,
    PreprocessedSample,
    TrainingMode,
)
from tsfmx.utils.device import pin_memory
from tsfmx.utils.logging import get_logger

if TYPE_CHECKING:
    import wandb

_logger = get_logger()


class MultimodalTrainer:
    """Trainer for multimodal, finetune, and baseline time series forecasting.

    - multimodal: adapter frozen, only fusion trained.
    - finetune: both adapter and fusion trained with separate optimizers (text embeddings are precomputed).
    - baseline: adapter fine-tuned, fusion unused.
    """

    def __init__(
        self,
        model: MultimodalDecoder,
        args: TrainingArguments,
        train_dataset: ConcatDataset[PreprocessedSample],
        val_dataset: ConcatDataset[PreprocessedSample],
        mode: TrainingMode,
        device: torch.device,
        wandb_run: wandb.Run | None,
        fusion_optimizers: tuple[Optimizer | None, LRScheduler | None] = (None, None),
        adapter_optimizers: tuple[Optimizer | None, LRScheduler | None] = (None, None),
    ) -> None:
        """Initialize MultimodalTrainer.

        Args:
            model: MultimodalDecoder model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            mode: Training mode.
            device: Device to train on.
            wandb_run: W&B run instance for logging. If None, W&B logging is disabled.
            fusion_optimizers: (optimizer, lr_scheduler) for the fusion module. Defaults created from args if None.
            adapter_optimizers: (optimizer, lr_scheduler) for the adapter. Defaults created from args if None.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mode = mode
        self.device = device
        self._wandb_run = wandb_run

        self.model.to(self.device)

        if mode == "multimodal":
            self.model.adapter.freeze_parameters()
        else:
            self.model.adapter.unfreeze_parameters()

        collate_fn = multimodal_collate_fn if mode in ("multimodal", "finetune") else baseline_collate_fn
        self.train_loader = cast(
            DataLoader[Batch],
            DataLoader(
                train_dataset,
                batch_size=args.per_device_train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory(self.device),
            ),
        )
        self.val_loader = cast(
            DataLoader[Batch],
            DataLoader(
                val_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory(self.device),
            ),
        )

        self.loss_fn = nn.MSELoss()

        num_training_steps = args.num_train_epochs * math.ceil(
            len(self.train_loader) / args.gradient_accumulation_steps
        )

        fusion_optimizer, fusion_scheduler = fusion_optimizers
        adapter_optimizer, adapter_scheduler = adapter_optimizers

        self.fusion_optimizer, self.adapter_optimizer = self._resolve_optimizer(fusion_optimizer, adapter_optimizer)
        self.fusion_scheduler, self.adapter_scheduler = self._resolve_scheduler(
            fusion_scheduler, adapter_scheduler, num_training_steps
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return all trainable parameters across active components."""
        match self.mode:
            case "multimodal":
                yield from self.model.fusion.parameters()
            case "finetune":
                yield from self.model.fusion.parameters()
                yield from (p for p in self.model.adapter.parameters() if p.requires_grad)
            case "baseline":
                yield from (p for p in self.model.adapter.parameters() if p.requires_grad)

    def _create_optimizer(self, component: Literal["fusion", "adapter"]) -> Optimizer:
        match component:
            case "fusion":
                return AdamW(
                    self.model.fusion.parameters(),
                    lr=self.args.fusion_learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            case "adapter":
                return AdamW(
                    (p for p in self.model.adapter.parameters() if p.requires_grad),
                    lr=self.args.adapter_learning_rate,
                    weight_decay=self.args.weight_decay,
                )

    def _create_scheduler(
        self, component: Literal["fusion", "adapter"], optimizer: Optimizer, num_training_steps: int
    ) -> LRScheduler:
        match component:
            case "fusion":
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps, self.args.fusion_warmup_steps)
                match self.args.fusion_lr_scheduler_type:
                    case "linear":
                        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
                    case "cosine":
                        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported fusion_lr_scheduler_type: {self.args.fusion_lr_scheduler_type!r}"
                        )
            case "adapter":
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps, self.args.adapter_warmup_steps)
                match self.args.adapter_lr_scheduler_type:
                    case "linear":
                        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
                    case "cosine":
                        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported adapter_lr_scheduler_type: {self.args.adapter_lr_scheduler_type!r}"
                        )

    def _resolve_optimizer(
        self,
        fusion_optimizer: Optimizer | None,
        adapter_optimizer: Optimizer | None,
    ) -> tuple[Optimizer | None, Optimizer | None]:
        match self.mode:
            case "multimodal":
                return fusion_optimizer or self._create_optimizer("fusion"), None
            case "finetune":
                return (
                    fusion_optimizer or self._create_optimizer("fusion"),
                    adapter_optimizer or self._create_optimizer("adapter"),
                )
            case "baseline":
                return None, adapter_optimizer or self._create_optimizer("adapter")
            case _ as mode:
                raise NotImplementedError(f"Unsupported mode: {mode!r}")

    def _resolve_scheduler(
        self,
        fusion_scheduler: LRScheduler | None,
        adapter_scheduler: LRScheduler | None,
        num_training_steps: int,
    ) -> tuple[LRScheduler | None, LRScheduler | None]:
        match self.mode:
            case "multimodal":
                assert self.fusion_optimizer is not None
                return fusion_scheduler or self._create_scheduler(
                    "fusion", self.fusion_optimizer, num_training_steps
                ), None
            case "finetune":
                assert self.fusion_optimizer is not None and self.adapter_optimizer is not None
                return (
                    fusion_scheduler or self._create_scheduler("fusion", self.fusion_optimizer, num_training_steps),
                    adapter_scheduler or self._create_scheduler("adapter", self.adapter_optimizer, num_training_steps),
                )
            case "baseline":
                assert self.adapter_optimizer is not None
                return None, adapter_scheduler or self._create_scheduler(
                    "adapter", self.adapter_optimizer, num_training_steps
                )
            case _ as mode:
                raise NotImplementedError(f"Unsupported mode: {mode!r}")

    def train_epoch(self) -> float:
        """Train one epoch.

        Returns:
            Average training loss for the epoch.

        Raises:
            RuntimeError: If the training dataset is empty.
        """
        self.model.train()
        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError("Training dataset is empty.")

        total_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            context = batch["context"].to(self.device)
            horizon = batch["horizon"].to(self.device)
            horizon_len = horizon.shape[-1]
            input_padding = torch.zeros_like(context, dtype=torch.bool)
            text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
            point_forecast = cast(torch.Tensor, self.model(horizon_len, context, input_padding, text_embeddings))

            loss = self.loss_fn(point_forecast, horizon)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            scaled_loss = loss.item() * self.args.gradient_accumulation_steps

            if (i + 1) % self.args.gradient_accumulation_steps == 0 or (i + 1) == num_batches:
                if self.args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._get_trainable_params(), self.args.max_grad_norm)
                if self.fusion_optimizer is not None:
                    self.fusion_optimizer.step()
                    self.fusion_optimizer.zero_grad()
                if self.adapter_optimizer is not None:
                    self.adapter_optimizer.step()
                    self.adapter_optimizer.zero_grad()
                if self.fusion_scheduler is not None:
                    self.fusion_scheduler.step()
                if self.adapter_scheduler is not None:
                    self.adapter_scheduler.step()
                self.global_step += 1

                if (
                    self.args.logging_strategy == "steps"
                    and self.global_step % self.args.logging_steps == 0
                    and self._wandb_run is not None
                ):
                    log: dict[str, float] = {"train/loss": scaled_loss}
                    if self.fusion_optimizer is not None:
                        log["train/fusion_lr"] = self.fusion_optimizer.param_groups[0]["lr"]
                    if self.adapter_optimizer is not None:
                        log["train/adapter_lr"] = self.adapter_optimizer.param_groups[0]["lr"]
                    self._wandb_run.log(log, step=self.global_step)

            total_loss += scaled_loss

            if i % self.args.logging_steps == 0:
                _logger.info(
                    "Epoch %d, Batch %d/%d, Loss: %.6f",
                    self.current_epoch,
                    i,
                    num_batches,
                    scaled_loss,
                )

        return total_loss / num_batches

    def validate_epoch(self) -> float:
        """Run one validation epoch.

        Returns:
            Average validation loss for the epoch.

        Raises:
            RuntimeError: If the validation dataset is empty.
        """
        self.model.eval()
        num_batches = len(self.val_loader)
        if num_batches == 0:
            raise RuntimeError("Validation dataset is empty.")

        total_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                context = batch["context"].to(self.device)
                horizon = batch["horizon"].to(self.device)
                horizon_len = horizon.shape[-1]
                input_padding = torch.zeros_like(context, dtype=torch.bool)
                text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
                point_forecast = cast(torch.Tensor, self.model(horizon_len, context, input_padding, text_embeddings))

                loss = self.loss_fn(point_forecast, horizon).item()
                total_loss += loss

                if i % self.args.logging_steps == 0:
                    _logger.info(
                        "Epoch %d, Batch %d/%d, Val Loss: %.6f",
                        self.current_epoch,
                        i,
                        num_batches,
                        loss,
                    )

        return total_loss / num_batches

    def _build_checkpoint(self) -> MultimodalCheckpoint | FinetuneCheckpoint | BaselineCheckpoint:
        """Build a mode-specific checkpoint dict from current training state."""
        match self.mode:
            case "multimodal":
                assert self.fusion_optimizer is not None and self.fusion_scheduler is not None
                return MultimodalCheckpoint(
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    fusion_state_dict=self.model.fusion.state_dict(),
                    fusion_optimizer_state_dict=self.fusion_optimizer.state_dict(),
                    fusion_scheduler_state_dict=self.fusion_scheduler.state_dict(),
                )
            case "finetune":
                assert (
                    self.fusion_optimizer is not None
                    and self.fusion_scheduler is not None
                    and self.adapter_optimizer is not None
                    and self.adapter_scheduler is not None
                )
                return FinetuneCheckpoint(
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    fusion_state_dict=self.model.fusion.state_dict(),
                    adapter_state_dict=self.model.adapter.state_dict(),
                    fusion_optimizer_state_dict=self.fusion_optimizer.state_dict(),
                    fusion_scheduler_state_dict=self.fusion_scheduler.state_dict(),
                    adapter_optimizer_state_dict=self.adapter_optimizer.state_dict(),
                    adapter_scheduler_state_dict=self.adapter_scheduler.state_dict(),
                )
            case "baseline":
                assert self.adapter_optimizer is not None and self.adapter_scheduler is not None
                return BaselineCheckpoint(
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    adapter_state_dict=self.model.adapter.state_dict(),
                    adapter_optimizer_state_dict=self.adapter_optimizer.state_dict(),
                    adapter_scheduler_state_dict=self.adapter_scheduler.state_dict(),
                )
            case _ as mode:
                raise NotImplementedError(f"Unsupported mode: {mode!r}")

    def _load_checkpoint_state(
        self, checkpoint: MultimodalCheckpoint | FinetuneCheckpoint | BaselineCheckpoint
    ) -> None:
        """Load mode-specific state from checkpoint."""
        match self.mode:
            case "multimodal":
                mc = cast(MultimodalCheckpoint, checkpoint)
                self.model.fusion.load_state_dict(mc["fusion_state_dict"])
                assert self.fusion_optimizer is not None and self.fusion_scheduler is not None
                self.fusion_optimizer.load_state_dict(mc["fusion_optimizer_state_dict"])
                self.fusion_scheduler.load_state_dict(mc["fusion_scheduler_state_dict"])
            case "finetune":
                fc = cast(FinetuneCheckpoint, checkpoint)
                self.model.fusion.load_state_dict(fc["fusion_state_dict"])
                self.model.adapter.load_state_dict(fc["adapter_state_dict"])
                assert (
                    self.fusion_optimizer is not None
                    and self.fusion_scheduler is not None
                    and self.adapter_optimizer is not None
                    and self.adapter_scheduler is not None
                )
                self.fusion_optimizer.load_state_dict(fc["fusion_optimizer_state_dict"])
                self.fusion_scheduler.load_state_dict(fc["fusion_scheduler_state_dict"])
                self.adapter_optimizer.load_state_dict(fc["adapter_optimizer_state_dict"])
                self.adapter_scheduler.load_state_dict(fc["adapter_scheduler_state_dict"])
            case "baseline":
                bc = cast(BaselineCheckpoint, checkpoint)
                self.model.adapter.load_state_dict(bc["adapter_state_dict"])
                assert self.adapter_optimizer is not None and self.adapter_scheduler is not None
                self.adapter_optimizer.load_state_dict(bc["adapter_optimizer_state_dict"])
                self.adapter_scheduler.load_state_dict(bc["adapter_scheduler_state_dict"])

    def _rotate_checkpoints(self) -> None:
        if self.args.save_total_limit is None:
            return

        checkpoints = sorted(
            self.args.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[: -self.args.save_total_limit]:
                checkpoint.unlink()
                _logger.info("Deleted old checkpoint: %s", checkpoint.name)

    def save_checkpoint(self, val_loss: float) -> None:
        """Save model checkpoint.

        For 'epoch' strategy, saves every epoch and separately tracks the best.
        For 'best' strategy, saves only when val_loss improves.

        Args:
            val_loss: Current validation loss.
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        if self.args.save_strategy == "best" and not is_best:
            return

        checkpoint = self._build_checkpoint()

        if self.args.save_strategy == "epoch":
            checkpoint_path = self.args.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            _logger.info("Saved checkpoint at epoch %d", self.current_epoch)

            if self.args.save_total_limit is not None:
                self._rotate_checkpoints()

        if is_best:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            _logger.info("Saved best model checkpoint at epoch %d", self.current_epoch)

    def train(self) -> None:
        """Main training loop."""
        if self.args.eval_strategy != "epoch":
            raise NotImplementedError(
                f"eval_strategy={self.args.eval_strategy!r} is not supported; only 'epoch' is implemented."
            )

        _logger.info("Starting %s training for %d epochs", self.mode, self.args.num_train_epochs)
        _logger.info("Training on %s", self.device)
        _logger.info("Train dataset size: %d", len(self.train_dataset))
        _logger.info("Validation dataset size: %d", len(self.val_dataset))

        for epoch in range(self.args.num_train_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            _logger.info("Epoch %d: Train Loss = %.6f, Val Loss = %.6f", epoch, train_loss, val_loss)

            if self._wandb_run is not None:
                log: dict[str, float] = {"val/loss": val_loss}
                if self.args.logging_strategy == "epoch":
                    log["train/loss"] = train_loss
                    if self.fusion_optimizer is not None:
                        log["train/fusion_lr"] = self.fusion_optimizer.param_groups[0]["lr"]
                    if self.adapter_optimizer is not None:
                        log["train/adapter_lr"] = self.adapter_optimizer.param_groups[0]["lr"]
                self._wandb_run.log(log, step=self.global_step)

            if self.args.save_strategy in ("epoch", "best"):
                self.save_checkpoint(val_loss)

        if self.args.load_best_model_at_end:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                checkpoint = cast(
                    MultimodalCheckpoint | FinetuneCheckpoint | BaselineCheckpoint,
                    torch.load(best_path, weights_only=True),
                )
                self._load_checkpoint_state(checkpoint)
                _logger.info("Loaded best model at end of training")

        _logger.info("Training completed")
