#!/usr/bin/env python3
"""Summarize test MSE across seed repeats, and compare each condition against a reference.

Reads the layout run_germany_wind_seed_repeats.sh writes: <root>/<condition>/seed_<n>/test_metrics.json,
one JSON per trained seed mapping each entity to its test MSE and MAE. A difference between two
conditions is reported against its standard error, so a gap that the seeds alone could produce is
not read as a result.
"""

import argparse
import json
import math
from pathlib import Path

from tsfmx.utils.logging import setup_logger

_logger = setup_logger()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Summarize test MSE across seed repeats.")
    parser.add_argument("root", type=str, help="Directory holding one subdirectory per condition.")
    parser.add_argument(
        "--reference",
        type=str,
        default="adapter",
        help="Condition the others are compared against. Skipped if absent.",
    )
    parser.add_argument("--output", type=str, help="Optional path to write the summary as JSON.")
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return the mean and the sample standard deviation (0 for a single value)."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _load(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Collect test MSE per condition, seed and entity, with the macro over entities added.

    Args:
        root: Directory holding one subdirectory per condition.

    Returns:
        Mapping condition -> seed directory name -> entity (and 'macro') -> MSE.

    Raises:
        ValueError: If two seeds of one condition were evaluated on different entities.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    for condition_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds: dict[str, dict[str, float]] = {}
        for metrics_path in sorted(condition_dir.glob("seed_*/test_metrics.json")):
            metrics = json.loads(metrics_path.read_text())
            mse = {entity: values["mse"] for entity, values in metrics.items()}
            # Macro per seed, so its spread across seeds is the spread of the number reported.
            mse["macro"] = sum(mse.values()) / len(mse)
            seeds[metrics_path.parent.name] = mse
        if not seeds:
            continue
        entity_sets = {tuple(sorted(mse)) for mse in seeds.values()}
        if len(entity_sets) > 1:
            raise ValueError(f"Seeds of {condition_dir.name} were evaluated on different entities: {entity_sets}")
        results[condition_dir.name] = seeds
    return results


def main() -> int:
    """Entry point: print per-condition spreads and the comparison against the reference.

    Returns:
        Exit code — 0 on success, 1 if no condition has any evaluated seed.
    """
    args = _parse_args()
    results = _load(Path(args.root))
    if not results:
        _logger.error("No seed_*/test_metrics.json found under %s", args.root)
        return 1

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for condition, seeds in results.items():
        rows = next(iter(seeds.values())).keys()
        summary[condition] = {}
        for row in rows:
            mean, std = _mean_std([mse[row] for mse in seeds.values()])
            summary[condition][row] = {"mean": mean, "std": std, "n": float(len(seeds))}

    rows = list(next(iter(summary.values())))
    print(
        f"Test MSE, mean ± std over seeds (n per condition: {', '.join(f'{c}={len(s)}' for c, s in results.items())})"
    )
    print(f"{'series':<18}" + "".join(f"{condition:>24}" for condition in summary))
    for row in rows:
        cells = [f"{s[row]['mean']:.4f} ± {s[row]['std']:.4f}" if row in s else "-" for s in summary.values()]
        print(f"{row:<18}" + "".join(f"{cell:>24}" for cell in cells))

    comparisons: dict[str, dict[str, float]] = {}
    reference = summary.get(args.reference)
    if reference is None:
        _logger.warning("Reference condition %r not found; skipping comparisons", args.reference)
    else:
        print(f"\nMacro difference against {args.reference} (negative is better than the reference)")
        ref = reference["macro"]
        for condition, stats in summary.items():
            if condition == args.reference:
                continue
            other = stats["macro"]
            diff = other["mean"] - ref["mean"]
            # Welch-style standard error: the two conditions share no seed-level pairing.
            se = math.sqrt(other["std"] ** 2 / other["n"] + ref["std"] ** 2 / ref["n"])
            ratio = diff / se if se > 0 else float("nan")
            comparisons[condition] = {
                "diff": diff,
                "diff_pct": diff / ref["mean"] * 100,
                "se": se,
                "diff_over_se": ratio,
            }
            print(
                f"  {condition:<22} {diff:+.4f} ({diff / ref['mean'] * 100:+.1f}%), SE {se:.4f}, diff/SE {ratio:+.1f}"
            )
        print("  |diff/SE| below about 2 is within what the seeds alone produce.")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary, "comparisons": comparisons, "seeds": results}, indent=2))
        _logger.info("Summary written to %s", output)
    return 0


if __name__ == "__main__":
    exit(main())
