#!/usr/bin/env python3
"""Measure whether the text carries any signal about what the unimodal forecast gets wrong.

Every other diagnostic in this repository asks whether a trained fusion head uses the text. None
of them can say whether there was anything to use: a head that reads perfectly still shows no
ablation response when the text is uninformative for the task. This script settles that question
without the fusion head, and so bounds what any fusion mechanism could achieve.

It regresses the residual of the unimodal forecast on the text embeddings. Three choices decide
what the number means:

- Scoring is cross-validated *within* the training split, over contiguous blocks. A train-to-test
  score instead asks whether the relationship transfers across time, and these splits are cut
  contiguously, so a real relationship confined to the training period scores arbitrarily badly.
  That transfer score is still reported, as a separate question.
- The features are reduced by PCA over a range of dimensions. There are roughly 1300 unaugmented
  windows against 768 correlated embedding dimensions, where ridge overfits whether or not any
  signal exists; a signal shows up at low dimension, while overfitting needs high dimension. The
  shape of the curve separates the two, which a single score cannot.
- Every fit is repeated on deranged features, which carry the same marginals and no per-sample
  correspondence. That null is the floor to read the score against.
- A text embedding is partly a timestamp — weather bulletins turn with the season, control
  statements with the schedule — and the residual has its own daily and seasonal structure, so the
  two correlate through time alone. Three nested models separate that: time features by
  themselves, text by itself, and both. What the text is worth is `both` minus `time`.

Fitting an intercept on standardized features absorbs the direction shared by every embedding, so
none of this is affected by the anisotropy that dominates these embeddings.
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
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Read both splits from the augmented cache. Augmented windows are near-duplicates, "
        "which defeats the ridge penalty search and the block folds alike; prefer leaving it off.",
    )
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output", type=str, default="outputs/text_residual_probe.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5, help="Contiguous blocks the training split is cut into.")
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Feature dimensions to score at. Values above the fold size are dropped.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4],
        help="Ridge penalties searched within each fold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deranged-feature null.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="time_mmd",
        help="Name the cache was built under: 'time_mmd', or 'fidel_ts' for a Fidel-TS sub-dataset.",
    )

    return parser.parse_args()


def _collect(
    model: MultimodalDecoder, dataset: Any, batch_size: int, device: torch.device
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the unimodal forecast and pair each residual with its text and its timestamp.

    Args:
        model: Decoder whose fusion head is bypassed by withholding text.
        dataset: Preprocessed split.
        batch_size: Batch size for the forward pass.
        device: Device to run on.

    Returns:
        Tuple of (features, residuals, time_features). Features are the sample's text embeddings
        flattened over patches, of shape (num_samples, num_patches * text_dims). Residuals are the
        horizon minus the unimodal forecast, of shape (num_samples, horizon_len). Rows stay in time
        order, which the block folds rely on.

    Raises:
        ValueError: If the split carries no text embeddings.
    """
    dataloader = build_dataloader(dataset, batch_size, multimodal_collate_fn, device)

    features: list[npt.NDArray[np.float64]] = []
    residuals: list[npt.NDArray[np.float64]] = []
    prediction_times: list[str] = []
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
            prediction_times.extend(str(m["prediction_time"]) for m in batch["metadata"] if "prediction_time" in m)

    stacked = np.concatenate(features)
    return stacked, np.concatenate(residuals), _time_features(prediction_times, len(stacked))


def _time_features(prediction_times: list[str], num_samples: int) -> npt.NDArray[np.float64]:
    """Build the clock a text stream could be standing in for.

    Weather bulletins turn with the season and control statements with the schedule, so a text
    embedding is partly a timestamp. Scoring these features on their own says how much of the
    text's apparent predictive power is that and nothing more.

    Args:
        prediction_times: Parseable timestamps, one per sample, or empty when the dataset carries
            none — Time-MMD samples do not, and then only the trend terms are available.
        num_samples: Number of samples, used when timestamps are absent.

    Returns:
        Features of shape (num_samples, num_time_features).
    """
    position = np.linspace(0.0, 1.0, num_samples)
    columns = [position, position**2]

    if len(prediction_times) == num_samples:
        seconds = np.asarray(np.array(prediction_times, dtype="datetime64[s]"), dtype=np.float64)
        for period in (86400.0, 604800.0, 31557600.0):  # day, week, year
            columns += [np.sin(2 * np.pi * seconds / period), np.cos(2 * np.pi * seconds / period)]
    else:
        _logger.warning("No per-sample timestamps; the time control covers trend only.")

    return np.column_stack(columns)


def _pipeline(kind: str, n_components: int, alphas: list[float], num_text: int) -> Any:
    """Build the pipeline for one of the three nested models.

    Features arrive as the text columns followed by the time columns, and `kind` selects which of
    them reach the ridge: 'text', 'time', or 'both'. Comparing 'both' against 'time' is what
    isolates the text's contribution, and doing it by nesting rather than by residualizing keeps
    every fit inside its own fold.

    Args:
        kind: Which columns to use.
        n_components: PCA dimension applied to the text columns.
        alphas: Ridge penalties to search.
        num_text: Number of leading text columns.

    Returns:
        An unfitted pipeline.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    text_block = ("text", make_pipeline(StandardScaler(), PCA(n_components=n_components)), slice(0, num_text))
    time_block = ("time", StandardScaler(), slice(num_text, None))
    match kind:
        case "text":
            blocks = [text_block]
        case "time":
            blocks = [time_block]
        case _:
            blocks = [text_block, time_block]

    return make_pipeline(ColumnTransformer(blocks), RidgeCV(alphas=alphas))


def _cross_validate(
    features: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    kind: str,
    n_components: int,
    alphas: list[float],
    n_folds: int,
    num_text: int,
) -> tuple[float, float, float]:
    """Score the pipeline over contiguous blocks of a time-ordered split.

    Blocks rather than shuffled folds because consecutive windows overlap: a shuffled fold would
    hold out a window whose neighbour is still being trained on.

    Args:
        features: Text columns followed by time columns, in time order.
        targets: Targets in time order, of shape (num_samples,) or (num_samples, num_outputs).
        kind: Which columns the model may use — 'text', 'time' or 'both'.
        n_components: PCA dimension applied to the text columns.
        alphas: Ridge penalties searched inside each fold.
        n_folds: Number of contiguous blocks.
        num_text: Number of leading text columns.

    Returns:
        Tuple of (r2, correlation, mean ridge penalty). The correlation is between the pooled
        out-of-fold predictions and the targets, averaged over output dimensions, and is unaffected
        by a miscalibrated scale that would drive r2 arbitrarily negative.
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    targets_2d = targets.reshape(len(targets), -1)
    predictions = np.zeros_like(targets_2d)
    penalties: list[float] = []

    for train_index, val_index in KFold(n_splits=n_folds, shuffle=False).split(features):
        pipeline = _pipeline(kind, n_components, alphas, num_text)
        pipeline.fit(features[train_index], targets_2d[train_index])
        predictions[val_index] = pipeline.predict(features[val_index]).reshape(len(val_index), -1)
        penalties.append(float(np.mean(np.atleast_1d(pipeline[-1].alpha_))))

    correlations = [
        float(np.corrcoef(predictions[:, i], targets_2d[:, i])[0, 1])
        for i in range(targets_2d.shape[1])
        if predictions[:, i].std() > 0
    ]
    return (
        float(r2_score(targets_2d, predictions, multioutput="uniform_average")),
        float(np.mean(correlations)) if correlations else 0.0,
        float(np.mean(penalties)),
    )


def _transfer(
    train_features: npt.NDArray[np.float64],
    train_targets: npt.NDArray[np.float64],
    target_features: npt.NDArray[np.float64],
    target_targets: npt.NDArray[np.float64],
    n_components: int,
    alphas: list[float],
    num_text: int,
    prefix: str,
) -> dict[str, float]:
    """Fit on the whole training split and score on a later one.

    Scores the same three nested models the cross-validation uses, so that the increment the text
    is worth can be read out of period as well as within it.

    The validation split is the one that decides which fusion head a sweep keeps, and it sits
    between train and test. Text whose contribution is already gone by validation cannot be kept
    by a search that selects on validation loss, whatever the mechanism is capable of.

    Args:
        train_features: Training features, text columns then time columns.
        train_targets: Training targets.
        target_features: Features of the split being transferred to.
        target_targets: Targets of the split being transferred to.
        n_components: PCA dimension applied to the text columns.
        alphas: Ridge penalties to search.
        num_text: Number of leading text columns.
        prefix: Key prefix naming the split, 'val' or 'test'.

    Returns:
        Scores for the text, time and both models, and the increment of the last over the second.
    """
    from sklearn.metrics import r2_score

    scores: dict[str, float] = {}
    for kind in ("text", "time", "both"):
        pipeline = _pipeline(kind, n_components, alphas, num_text)
        pipeline.fit(train_features, train_targets.reshape(len(train_targets), -1))
        predictions = pipeline.predict(target_features).reshape(len(target_features), -1)
        scores[f"{prefix}_r2_{kind}"] = float(
            r2_score(target_targets.reshape(len(target_targets), -1), predictions, multioutput="uniform_average")
        )

    scores[f"{prefix}_increment"] = scores[f"{prefix}_r2_both"] - scores[f"{prefix}_r2_time"]
    return scores


def _probe_domain(
    model: MultimodalDecoder,
    datasets: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Score real and deranged text across PCA dimensions on one domain.

    Args:
        model: Decoder used for the unimodal forecast.
        datasets: The 'train', 'val' and 'test' splits.
        args: Parsed arguments carrying the grids and fold count.
        device: Device to run on.

    Returns:
        Per-dimension scores plus the split sizes and residual scale.
    """
    train_text, train_residuals, train_time = _collect(model, datasets["train"], args.batch_size, device)
    val_text, val_residuals, val_time = _collect(model, datasets["val"], args.batch_size, device)
    test_text, test_residuals, test_time = _collect(model, datasets["test"], args.batch_size, device)

    num_text = train_text.shape[1]
    train_features = np.hstack([train_text, train_time])
    val_features = np.hstack([val_text, val_time])
    test_features = np.hstack([test_text, test_time])
    # Deranging the text columns alone leaves the clock aligned, so the null says what a model
    # already holding the timestamp gains from text that belongs to another sample.
    shuffled = np.hstack([train_text[derangement(len(train_text), args.seed)], train_time])
    train_mean = train_residuals.mean(axis=1)
    val_mean = val_residuals.mean(axis=1)
    test_mean = test_residuals.mean(axis=1)

    # PCA cannot ask for more components than the smallest fold provides.
    fold_size = len(train_features) - len(train_features) // args.cv_folds
    limit = min(num_text, fold_size - 1)
    dimensions = sorted(
        {min(k, limit) for k in args.pca_components if k <= limit} | {min(args.pca_components[0], limit)}
    )

    # Independent of the text dimension, so scored once.
    r2_time, corr_time, _ = _cross_validate(train_features, train_mean, "time", 1, args.alphas, args.cv_folds, num_text)

    curve: list[dict[str, float]] = []
    for k in dimensions:

        def cv(
            features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64], kind: str, k: int = k
        ) -> tuple[float, float, float]:
            return _cross_validate(features, targets, kind, k, args.alphas, args.cv_folds, num_text)

        r2_mean, corr_mean, alpha = cv(train_features, train_mean, "text")
        r2_mean_null, _, _ = cv(shuffled, train_mean, "text")
        r2_horizon, _, _ = cv(train_features, train_residuals, "text")
        r2_horizon_null, _, _ = cv(shuffled, train_residuals, "text")
        r2_both, _, _ = cv(train_features, train_mean, "both")
        r2_both_null, _, _ = cv(shuffled, train_mean, "both")
        curve.append(
            {
                "n_components": float(k),
                "cv_r2_mean": r2_mean,
                "cv_r2_mean_null": r2_mean_null,
                "cv_corr_mean": corr_mean,
                "cv_r2_horizon": r2_horizon,
                "cv_r2_horizon_null": r2_horizon_null,
                "cv_r2_both": r2_both,
                "cv_r2_text_increment": r2_both - r2_time,
                "cv_r2_text_increment_null": r2_both_null - r2_time,
                "ridge_alpha": alpha,
                **_transfer(train_features, train_mean, val_features, val_mean, k, args.alphas, num_text, "val"),
                **_transfer(train_features, train_mean, test_features, test_mean, k, args.alphas, num_text, "test"),
            }
        )

    return {
        "curve": curve,
        "cv_r2_time": r2_time,
        "cv_corr_time": corr_time,
        "residual_rms": float(np.sqrt(np.mean(test_residuals**2))),
        "num_train": float(len(train_features)),
        "num_val": float(len(val_features)),
        "num_test": float(len(test_features)),
        "num_text_features": float(num_text),
        "num_time_features": float(train_time.shape[1]),
    }


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

    results: dict[str, dict[str, Any]] = {}
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
                for split in ("train", "val", "test")
            }
        except Exception as e:
            _logger.warning("Skipping %s: %s", domain, e)
            continue

        results[domain] = _probe_domain(model, splits, args, device)
        best = max(results[domain]["curve"], key=lambda row: row["cv_r2_text_increment"])
        _logger.info(
            "%s: time alone %.4f | text %.4f (null %.4f, corr %.3f) | text over time: cv %.4f "
            "(null %.4f), val %.4f, test %.4f — at %d components",
            domain,
            results[domain]["cv_r2_time"],
            best["cv_r2_mean"],
            best["cv_r2_mean_null"],
            best["cv_corr_mean"],
            best["cv_r2_text_increment"],
            best["cv_r2_text_increment_null"],
            best["val_increment"],
            best["test_increment"],
            int(best["n_components"]),
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
