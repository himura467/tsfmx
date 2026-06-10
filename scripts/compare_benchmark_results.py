#!/usr/bin/env python3
"""Print a comparison table of tsfmx vs MM-TSFlib per-domain MSE/MAE."""

import argparse
import json
import re
from pathlib import Path

_MM_DOMAIN_ALIASES: dict[str, str] = {
    "Algriculture": "Agriculture",
    "Public_Health": "Health_US",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--tsfmx", default="outputs/tsfmx_eval_results.json")
    parser.add_argument("--mm-tsflib", default="third_party/MM-TSFlib/result_longterm_forecast")

    return parser.parse_args()


def _load_tsfmx(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        print(f"[warn] not found: {path}")
        return {}
    return json.loads(path.read_text())


def _load_mm_tsflib(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        print(f"[warn] not found: {path}")
        return {}
    results: dict[str, dict[str, float]] = {}
    setting = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if "mse:" in line and setting:
            parsed = {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.eE+\-]+)", line)}
            if "mse" in parsed and "mae" in parsed:
                m = re.match(r"^(.+?)_\d+_\d+_LLAMA3", setting)
                if m:
                    domain = _MM_DOMAIN_ALIASES.get(m.group(1), m.group(1))
                    results[domain] = {"mse": parsed["mse"], "mae": parsed["mae"]}
            setting = ""
        else:
            setting = line
    return results


def _col(v: float | None) -> str:
    return f"{v:.5f}".rjust(12) if v is not None else "N/A".rjust(12)


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    args = _parse_args()

    tsfmx = _load_tsfmx(Path(args.tsfmx))
    mm = _load_mm_tsflib(Path(args.mm_tsflib))

    domains = sorted(set(tsfmx) | set(mm))
    if not domains:
        print("No results. Run both benchmarks first.")
        return

    header = f"{'Domain':<20}{'tsfmx MSE':>12}{'tsfmx MAE':>12}{'MM-TSF MSE':>12}{'MM-TSF MAE':>12}"
    sep = "-" * len(header)
    print("=" * len(header))
    print("tsfmx vs MM-TSFlib (LLAMA3+Autoformer) | Time-MMD | ctx=32 hz=32 split=70/10/20")
    print("=" * len(header))
    print(header)
    print(sep)

    t_mses, t_maes, m_mses, m_maes = [], [], [], []
    for domain in domains:
        t = tsfmx.get(domain, {})
        m = mm.get(domain, {})
        print(f"{domain:<20}{_col(t.get('mse'))}{_col(t.get('mae'))}{_col(m.get('mse'))}{_col(m.get('mae'))}")
        if "mse" in t:
            t_mses.append(t["mse"])
            t_maes.append(t["mae"])
        if "mse" in m:
            m_mses.append(m["mse"])
            m_maes.append(m["mae"])

    if t_mses or m_mses:
        print(sep)
        print(f"{'Mean':<20}{_col(_avg(t_mses))}{_col(_avg(t_maes))}{_col(_avg(m_mses))}{_col(_avg(m_maes))}")


if __name__ == "__main__":
    main()
