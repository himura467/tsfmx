"""Shared types for the tsfmx package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

TrainingMode = Literal["multimodal", "finetune", "baseline"]


class RawSample(TypedDict):
    """A single raw dataset sample before preprocessing."""

    context: npt.NDArray[np.float32]
    horizon: npt.NDArray[np.float32]
    patched_texts: list[list[str]]
    metadata: dict[str, Any]


class PreprocessedSample(TypedDict):
    """A single dataset sample after preprocessing."""

    context: npt.NDArray[np.float32]
    horizon: npt.NDArray[np.float32]
    text_embeddings: NotRequired[npt.NDArray[np.float32]]
    metadata: dict[str, Any]


class Batch(TypedDict):
    """A collated batch of samples."""

    context: torch.Tensor
    horizon: torch.Tensor
    text_embeddings: NotRequired[torch.Tensor]
    metadata: list[dict[str, Any]]


class CheckpointBase(TypedDict):
    """Base checkpoint fields."""

    epoch: int
    global_step: int
    best_val_loss: float


class MultimodalCheckpoint(CheckpointBase):
    """Checkpoint for multimodal mode (fusion only)."""

    fusion_state_dict: dict[str, Any]
    fusion_optimizer_state_dict: dict[str, Any]
    fusion_scheduler_state_dict: dict[str, Any]


class FinetuneCheckpoint(CheckpointBase):
    """Checkpoint for fine-tuning mode (adapter + fusion)."""

    fusion_state_dict: dict[str, Any]
    adapter_state_dict: dict[str, Any]
    fusion_optimizer_state_dict: dict[str, Any]
    fusion_scheduler_state_dict: dict[str, Any]
    adapter_optimizer_state_dict: dict[str, Any]
    adapter_scheduler_state_dict: dict[str, Any]


class BaselineCheckpoint(CheckpointBase):
    """Checkpoint for baseline mode (adapter only)."""

    adapter_state_dict: dict[str, Any]
    adapter_optimizer_state_dict: dict[str, Any]
    adapter_scheduler_state_dict: dict[str, Any]


class EvaluationMetrics(TypedDict):
    """Evaluation metrics."""

    mse: float
    mae: float
