"""Evaluator for multimodal decoder models."""

from typing import cast

import torch
from torch.utils.data import DataLoader

from tsfmx.decoder import MultimodalDecoder
from tsfmx.types import Batch, EvaluationMetrics


class MultimodalEvaluator:
    """Computes evaluation metrics for a multimodal decoder.

    In fusion/finetune mode, text embeddings from the batch are fused with time series embeddings.
    In baseline mode, no text embeddings are present in the batch and fusion is skipped.
    """

    def __init__(self, model: MultimodalDecoder, device: torch.device) -> None:
        """Initialize MultimodalEvaluator.

        Args:
            model: Multimodal decoder model to evaluate.
            device: Device to run inference on.
        """
        self.model = model
        self.device = device

    def evaluate(self, dataloader: DataLoader[Batch]) -> EvaluationMetrics:
        """Evaluate the model on the given dataloader and return aggregated metrics.

        Args:
            dataloader: DataLoader providing batches of preprocessed samples.

        Returns:
            EvaluationMetrics with mse and mae computed over all samples.

        Raises:
            RuntimeError: If the evaluation dataset is empty.
        """
        self.model.eval()

        total_mse = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                context = batch["context"].to(self.device)
                horizon = batch["horizon"].to(self.device)
                horizon_len = horizon.shape[-1]
                input_padding = torch.zeros_like(context, dtype=torch.bool)
                text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
                point_forecast = cast(
                    torch.Tensor,
                    self.model(horizon_len, context, input_padding, text_embeddings),
                )

                mse = torch.mean((point_forecast - horizon) ** 2)
                mae = torch.mean(torch.abs(point_forecast - horizon))
                total_mse += mse.item() * context.size(0)
                total_mae += mae.item() * context.size(0)
                num_samples += context.size(0)

        if num_samples == 0:
            raise RuntimeError("Evaluation dataset is empty.")

        return EvaluationMetrics(
            mse=total_mse / num_samples,
            mae=total_mae / num_samples,
        )
