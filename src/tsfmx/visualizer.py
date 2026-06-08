"""Visualization utilities for multimodal time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from tsfmx.decoder import MultimodalDecoder
from tsfmx.types import Batch

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class PredictionVisualizer:
    """Plots model forecasts versus ground-truth horizons for time series forecasting."""

    def __init__(
        self,
        model: MultimodalDecoder,
        device: torch.device,
        figsize: tuple[float, float],
        max_samples: int | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize PredictionVisualizer.

        Args:
            model: Trained MultimodalDecoder model.
            device: Device to run inference on.
            figsize: (width, height) in inches per figure.
            max_samples: Max samples plotted per split. None plots all samples.
            output_dir: Directory to save plot PNGs (one sub-directory per split). None to skip saving.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as _  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. "
                'Install the optional dependency with: pip install "tsfmx[viz]"'
            ) from exc

        self.model = model
        self.device = device
        self.figsize = figsize
        self.max_samples = max_samples
        self.output_dir = output_dir

    def _draw(
        self,
        ax: Axes,
        context: npt.NDArray[np.float32],
        horizon: npt.NDArray[np.float32],
        forecast: npt.NDArray[np.float32],
        sample_idx: int,
        split_name: str,
    ) -> None:
        """Draw context, true horizon, and point forecast onto ax."""
        context_len = len(context)
        horizon_len = len(horizon)

        x_ctx = np.arange(context_len)
        x_hrz = np.arange(context_len, context_len + horizon_len)

        ax.plot(x_ctx, context, color="r", linewidth=1, label="Context")
        ax.plot(x_hrz, horizon, color="g", linewidth=1, label="Horizon")
        ax.plot(x_hrz, forecast, color="b", linestyle="--", linewidth=1, label="Forecast")
        ax.axvline(x=context_len - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{split_name} — sample {sample_idx}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

    def plot_forecast(
        self,
        dataloader: DataLoader[Batch],
        split_name: str,
        show: bool = False,
    ) -> list[Figure]:
        """Generate and optionally save forecast plots for all samples in the dataloader.

        Args:
            dataloader: DataLoader providing batches of preprocessed samples.
            split_name: Label for the plot title and output sub-directory name.
            show: If True, call plt.show() after each figure.

        Returns:
            List of Figure objects, one per plotted sample.
        """
        import matplotlib.pyplot as plt

        split_dir: Path | None = None
        if self.output_dir is not None:
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        figures: list[Figure] = []
        sample_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                forecast, context, horizon = self.model.forecast(batch, self.device)

                context_np: npt.NDArray[np.float32] = context.cpu().numpy()
                horizon_np: npt.NDArray[np.float32] = horizon.cpu().numpy()
                forecast_np: npt.NDArray[np.float32] = forecast.cpu().numpy()

                for i in range(context_np.shape[0]):
                    if self.max_samples is not None and sample_idx >= self.max_samples:
                        return figures

                    fig, ax = plt.subplots(figsize=self.figsize)
                    self._draw(ax, context_np[i], horizon_np[i], forecast_np[i], sample_idx, split_name)

                    if split_dir is not None:
                        fig.savefig(split_dir / f"sample_{sample_idx:04d}.png", bbox_inches="tight", dpi=100)

                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

                    figures.append(fig)
                    sample_idx += 1

        return figures

    def visualize_all_splits(
        self,
        train_loader: DataLoader[Batch] | None = None,
        val_loader: DataLoader[Batch] | None = None,
        test_loader: DataLoader[Batch] | None = None,
        show: bool = False,
    ) -> dict[str, list[Figure]]:
        """Generate forecast plots for train, val, and test splits.

        Args:
            train_loader: DataLoader for training data, or None to skip.
            val_loader: DataLoader for validation data, or None to skip.
            test_loader: DataLoader for test data, or None to skip.
            show: If True, display each figure interactively via plt.show().

        Returns:
            Dict mapping each split name to its list of Figure objects.
        """
        results: dict[str, list[Figure]] = {}
        for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            if loader is not None:
                results[name] = self.plot_forecast(loader, name, show=show)
        return results
