#!/usr/bin/env python3
"""Measure whether the text carries any signal about what the unimodal forecast gets wrong.

Every other diagnostic in this repository asks whether a trained fusion head uses the text. None
of them can say whether there was anything to use: a fusion head that reads perfectly still shows
no ablation response when the text is uninformative for the task. This script settles that
question without the fusion head, and so bounds what any fusion mechanism could achieve.

It regresses the residual of the unimodal forecast on the text embeddings with ridge regression,
fitting on the training split and scoring on test. Two properties make the answer readable:

- Fitting an intercept absorbs any component shared by every embedding, so the score is unaffected
  by the anisotropy of the sentence encoder. A low score is a statement about information, not
  about the constant offset that scripts/diagnose_time_mmd_text_fusion.py reports.
- The same fit is repeated on deranged text, which carries the identical marginal distribution and
  no per-sample correspondence. That null gives the score a floor to be read against, which the
  ablation deltas currently lack.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from examples.time_mmd.builders import build_decoder
from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from tsfmx.ablation import derangement
from tsfmx.data.collate import multimodal_collate_fn
from tsfmx.data.loader import build_dataloader
from tsfmx.data.splits import DomainSpec, load_split_dataset
from tsfmx.decoder import MultimodalDecoder
from tsfmx.utils.device import resolve_device
from tsfmx.utils.logging import setup_logger
from tsfmx.utils.seed import set_seed

_logger = setup_logger()

_DEFAULT_DOMAINS = ["Agriculture", "Economy", "Environment", "Health_US", "Traffic"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether text embeddings predict the residual of the unimodal forecast.",
    )

    parser.add_argument("--model-config", type=str)
    parser.add_argument("--forecast-config", type=str)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Optional adapter-mode checkpoint. Without it the pretrained TSFM is used unchanged, "
        "which is the baseline the text would have to improve on.",
    )
    parser.add_argument("--domains", nargs="+", default=_DEFAULT_DOMAINS)
    parser.add_argument("--augment", action="store_true", help="Read both splits from the augmented cache.")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output", type=str, default="outputs/text_residual_probe.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4],
        help="Ridge penalties searched by leave-one-out cross-validation on the training split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deranged-text null.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="time_mmd",
        help="Name the cache was built under: 'time_mmd', or 'fidel_ts' for a Fidel-TS sub-dataset.",
    )

    return parser.parse_args()


def _collect(
    model: MultimodalDecoder, dataset: Any, batch_size: int, device: torch.device
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the unimodal forecast and pair each residual with its text embeddings.

    Args:
        model: Decoder whose fusion head is bypassed by withholding text.
        dataset: Preprocessed split.
        batch_size: Batch size for the forward pass.
        device: Device to run on.

    Returns:
        Tuple of (features, residuals). Features are the sample's text embeddings flattened over
        patches, of shape (num_samples, num_patches * text_dims). Residuals are the horizon minus
        the unimodal forecast, of shape (num_samples, horizon_len).

    Raises:
        ValueError: If the split carries no text embeddings.
    """
    dataloader = build_dataloader(dataset, batch_size, multimodal_collate_fn, device)

    features: list[npt.NDArray[np.float64]] = []
    residuals: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for batch in dataloader:
            if "text_embeddings" not in batch:
                raise ValueError("Split carries no text embeddings; rebuild the cache with a text encoder.")
            context = batch["context"].to(device)
            horizon = batch["horizon"].to(device)
            padding = torch.zeros_like(context, dtype=torch.bool)
            # No text, so fusion is skipped entirely: this is the forecast the text would improve on.
            forecast = model(horizon.shape[-1], context, padding, None)
            residuals.append((horizon - forecast).cpu().numpy().astype(np.float64))
            embeddings = batch["text_embeddings"]
            features.append(embeddings.reshape(embeddings.shape[0], -1).cpu().numpy().astype(np.float64))

    return np.concatenate(features), np.concatenate(residuals)


def _r2(
    train_features: npt.NDArray[np.float64],
    train_targets: npt.NDArray[np.float64],
    test_features: npt.NDArray[np.float64],
    test_targets: npt.NDArray[np.float64],
    alphas: list[float],
) -> float:
    """Fit ridge regression on the training split and score it on test.

    Standardizing and fitting an intercept means a direction shared by every embedding contributes
    nothing, so the score reflects between-sample information alone.

    Args:
        train_features: Training features.
        train_targets: Training targets.
        test_features: Test features.
        test_targets: Test targets.
        alphas: Ridge penalties to search.

    Returns:
        Coefficient of determination on the test split, averaged over target dimensions. Negative
        values mean the fit predicts worse than the training-split mean.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(train_features, train_targets)
    return float(r2_score(test_targets, model.predict(test_features), multioutput="uniform_average"))


def _probe_domain(
    model: MultimodalDecoder,
    train_dataset: Any,
    test_dataset: Any,
    batch_size: int,
    device: torch.device,
    alphas: list[float],
    seed: int,
) -> dict[str, float]:
    """Score real and deranged text on one domain.

    Args:
        model: Decoder used for the unimodal forecast.
        train_dataset: Training split.
        test_dataset: Test split.
        batch_size: Batch size for the forward pass.
        device: Device to run on.
        alphas: Ridge penalties to search.
        seed: Seed for the derangement.

    Returns:
        Scores for the full residual and for its mean, each against a deranged-text null.
    """
    train_features, train_residuals = _collect(model, train_dataset, batch_size, device)
    test_features, test_residuals = _collect(model, test_dataset, batch_size, device)

    # Deranging the training features alone leaves the test pairing intact, so the null differs
    # from the real fit only in whether the training pairs carried a correspondence.
    shuffled = train_features[derangement(len(train_features), seed)]

    scores = {
        "r2_horizon": _r2(train_features, train_residuals, test_features, test_residuals, alphas),
        "r2_horizon_null": _r2(shuffled, train_residuals, test_features, test_residuals, alphas),
        "r2_mean": _r2(
            train_features,
            train_residuals.mean(axis=1, keepdims=True),
            test_features,
            test_residuals.mean(axis=1, keepdims=True),
            alphas,
        ),
        "r2_mean_null": _r2(
            shuffled,
            train_residuals.mean(axis=1, keepdims=True),
            test_features,
            test_residuals.mean(axis=1, keepdims=True),
            alphas,
        ),
        "residual_rms": float(np.sqrt(np.mean(test_residuals**2))),
        "num_train": float(len(train_features)),
        "num_test": float(len(test_features)),
    }
    return scores


def main() -> int:
    """Entry point: probe every requested domain and write the scores.

    Returns:
        Exit code. 0 on success, 1 if no domain could be probed.
    """
    args = _parse_args()
    model_config = ModelConfig.from_yaml(Path(args.model_config)) if args.model_config else ModelConfig()
    forecast_config = ForecastConfig.from_yaml(Path(args.forecast_config)) if args.forecast_config else ForecastConfig()

    set_seed(args.seed)
    device = resolve_device()
    _logger.info("Using device: %s", device)

    model = build_decoder(model_config, device)
    if args.checkpoint_path:
        mode = model.load_checkpoint(Path(args.checkpoint_path))
        _logger.info("Loaded %s-mode checkpoint from %s", mode, args.checkpoint_path)
    model.eval()

    results: dict[str, dict[str, float]] = {}
    for domain in args.domains:
        try:
            splits = {
                split: load_split_dataset(
                    dataset_name=args.dataset,
                    domain_specs=[DomainSpec(name=f"{domain}_{split}", augment=args.augment)],
                    text_encoder_type=model_config.fusion.text_encoder_type,
                    patch_len=model_config.adapter.patch_len,
                    context_len=forecast_config.context_len,
                    horizon_len=forecast_config.horizon_len,
                    cache_dir=Path(args.cache_dir),
                    mode="fusion",
                )
                for split in ("train", "test")
            }
        except Exception as e:
            _logger.warning("Skipping %s: %s", domain, e)
            continue

        results[domain] = _probe_domain(
            model, splits["train"], splits["test"], args.batch_size, device, args.alphas, args.seed
        )
        _logger.info(
            "%s: r2_horizon=%.4f (null %.4f), r2_mean=%.4f (null %.4f)",
            domain,
            results[domain]["r2_horizon"],
            results[domain]["r2_horizon_null"],
            results[domain]["r2_mean"],
            results[domain]["r2_mean_null"],
        )

    if not results:
        _logger.error("No domain could be probed.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": results}, f, indent=2)
    _logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    exit(main())
