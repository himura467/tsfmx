#!/usr/bin/env python3
"""Export one finished trial of a W&B sweep as a sweep config that fixes every parameter.

A sweep's best trial is a single draw: the configuration that happened to reach the lowest
validation loss once, under one seed. Re-training it under several seeds measures how much of
that result was the seed; exporting the next-ranked trials as well measures how much was the luck
of which configuration topped the ranking. The written config is a grid over constants, so a
sweep created from it runs exactly one trial and then finishes.
"""

import argparse
from pathlib import Path
from typing import Any

import wandb
import yaml

from tsfmx.utils.logging import setup_logger

_logger = setup_logger()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Write a sweep config that fixes every parameter to the values of one ranked trial.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Sweep path as entity/project/sweep_id, as shown in the W&B sweep URL.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which trial to export, ranked by the sweep's own metric: 1 is the best.",
    )
    parser.add_argument("--output", type=str, required=True, help="Path of the sweep YAML to write.")
    return parser.parse_args()


def _ranked_runs(sweep: Any) -> list[Any]:
    """Order the sweep's finished trials by the metric the sweep optimized.

    Args:
        sweep: W&B public-API sweep.

    Returns:
        Finished runs that logged the metric, best first.

    Raises:
        ValueError: If the sweep config names no metric, or no finished run logged it.
    """
    metric = sweep.config.get("metric")
    if not metric or "name" not in metric:
        raise ValueError(f"Sweep {sweep.id} defines no metric to rank its trials by")
    name = metric["name"]
    runs = [run for run in sweep.runs if run.state == "finished" and name in run.summary]
    if not runs:
        raise ValueError(f"Sweep {sweep.id} has no finished run that logged {name!r}")
    return sorted(runs, key=lambda run: run.summary[name], reverse=metric.get("goal") == "maximize")


def _fixed_config(sweep_config: dict[str, Any], run: Any) -> dict[str, Any]:
    """Build a sweep config whose every parameter is the run's value.

    Only the sweep's own parameters are copied: the rest of a run's config is what the scripts
    derive from their arguments, and restating it would not change what the trial trains.

    Args:
        sweep_config: Config of the sweep the run belongs to.
        run: W&B public-API run to reproduce.

    Returns:
        Sweep config with the source sweep's metric and one constant per parameter.

    Raises:
        ValueError: If the run's config lacks a parameter the sweep defines.
    """
    names = list(sweep_config.get("parameters", {}))
    missing = [name for name in names if name not in run.config]
    if missing:
        raise ValueError(f"Run {run.id} has no value for sweep parameters {missing}")
    return {
        "method": "grid",
        "metric": sweep_config["metric"],
        "parameters": {name: {"value": run.config[name]} for name in names},
    }


def main() -> int:
    """Entry point: rank the sweep's trials and write the chosen one as a fixed config.

    Returns:
        Exit code — 0 on success, 1 if the rank is out of range.
    """
    args = _parse_args()

    sweep = wandb.Api().sweep(args.sweep)
    runs = _ranked_runs(sweep)
    if not 1 <= args.rank <= len(runs):
        _logger.error("Rank %d is out of range: sweep %s has %d ranked trials", args.rank, sweep.id, len(runs))
        return 1
    run = runs[args.rank - 1]
    metric_name = sweep.config["metric"]["name"]

    config = _fixed_config(sweep.config, run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        # Provenance as comments, so the file says which trial it reproduces without W&B at hand.
        f.write(f"# Trial {run.id} ({run.name}) of sweep {args.sweep}, ranked {args.rank} of {len(runs)}\n")
        f.write(f"# {metric_name} = {run.summary[metric_name]}\n")
        yaml.safe_dump(config, f, sort_keys=False)

    _logger.info(
        "Wrote trial %s (rank %d, %s=%s) to %s", run.id, args.rank, metric_name, run.summary[metric_name], output
    )
    return 0


if __name__ == "__main__":
    exit(main())
