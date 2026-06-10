#!/usr/bin/env python3
"""Split Time-MMD numerical data chronologically and duplicate textual data per split."""

import argparse
import shutil
from pathlib import Path
from typing import Literal, get_args

import pandas as pd

from examples.time_mmd.configs.domain_columns import DEFAULT_TIME_MMD_CONFIGS
from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from tsfmx.utils.logging import setup_logger

_logger = setup_logger()

Split = Literal["train", "val", "test"]
_SPLITS: tuple[Split, ...] = get_args(Split)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Split Time-MMD dataset into train / val / test sets.",
    )

    parser.add_argument("--data-path", type=str, default="data/Time-MMD")
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--domains", type=str, nargs="+")
    parser.add_argument("--force-rebuild", action="store_true", help="Overwrite existing split files.")

    return parser.parse_args()


def _split_numerical(
    numerical_dir: Path,
    domain: str,
    train_ratio: float,
    val_ratio: float,
    context_len: int,
    force_rebuild: bool,
) -> None:
    """Split a domain's numerical CSV into train / val / test subsets with context overlap."""
    src = numerical_dir / domain / f"{domain}.csv"
    if not src.exists():
        _logger.warning("Numerical file not found, skipping: %s", src)
        return

    df = pd.read_csv(src)
    date_col = DEFAULT_TIME_MMD_CONFIGS.get_config_for_domain(domain).start_date_col
    if date_col not in df.columns:
        _logger.error("Date column %r not found in %s — cannot split chronologically", date_col, src)
        return
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    slices: dict[Split, pd.DataFrame] = {
        "train": df.iloc[:train_end],
        "val": df.iloc[max(0, train_end - context_len) : val_end],
        "test": df.iloc[max(0, val_end - context_len) :],
    }

    for split in _SPLITS:
        split_domain = f"{domain}_{split}"
        out_dir = numerical_dir / split_domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split_domain}.csv"
        split_df = slices[split]
        if out_path.exists() and not force_rebuild:
            _logger.info("Skip (exists): numerical/%s/%s.csv (%d rows)", split_domain, split_domain, len(split_df))
            continue
        split_df.reset_index(drop=True).to_csv(out_path, index=False)
        _logger.info("Wrote: numerical/%s/%s.csv (%d rows)", split_domain, split_domain, len(split_df))


def _duplicate_textual(
    textual_dir: Path,
    domain: str,
    force_rebuild: bool,
) -> None:
    """Copy a domain's textual CSVs into each split subdirectory with renamed files.

    Args:
        textual_dir: Path to the textual/ directory.
        domain: Domain name.
        force_rebuild: Overwrite existing files when True.
    """
    domain_dir = textual_dir / domain
    if not domain_dir.exists():
        _logger.warning("Textual directory not found, skipping: %s", domain_dir)
        return

    sources = sorted(domain_dir.glob("*.csv"))
    if not sources:
        return

    for split in _SPLITS:
        split_domain = f"{domain}_{split}"
        out_dir = textual_dir / split_domain
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            # Rename {domain}_report.csv -> {domain}_{split}_report.csv
            suffix = src.stem[len(domain) :]
            out_name = f"{split_domain}{suffix}.csv"
            out_path = out_dir / out_name
            if out_path.exists() and not force_rebuild:
                _logger.info("Skip (exists): textual/%s/%s", split_domain, out_name)
                continue
            shutil.copy2(src, out_path)
            _logger.info("Copied: textual/%s/%s", split_domain, out_name)


def main() -> int:
    """Entry point: split numerical and duplicate textual files for all domains.

    Returns:
        Exit code — 0 on success, 1 on error.
    """
    args = _parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        _logger.error("train_ratio + val_ratio must be < 1.0")
        return 1

    data_path = Path(args.data_path)
    if not data_path.exists():
        _logger.error("Dataset directory not found: %s", data_path)
        return 1

    numerical_dir = data_path / "numerical"
    textual_dir = data_path / "textual"

    domains = args.domains or [
        d for d in TimeMmdDataset.get_domains(data_path) if not any(d.endswith(f"_{s}") for s in _SPLITS)
    ]
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    _logger.info("Domains: %s", domains)
    _logger.info("Ratios — train: %s, val: %s, test: %.2f", args.train_ratio, args.val_ratio, test_ratio)
    _logger.info("Context overlap prefix: %d rows", args.context_len)

    for domain in domains:
        _logger.info("Processing domain: %s", domain)
        _split_numerical(numerical_dir, domain, args.train_ratio, args.val_ratio, args.context_len, args.force_rebuild)
        _duplicate_textual(textual_dir, domain, args.force_rebuild)

    _logger.info("All domains processed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
