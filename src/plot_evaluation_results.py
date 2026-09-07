#!/usr/bin/env python3
"""Generate benchmark PDF/SVG figures from an existing episodes CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.evaluate_policies import FIELDS, plot_results


ROOT = Path(__file__).resolve().parents[1]


def load_episodes(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No episode rows found in {path}")

    required = set(FIELDS) - {"revisit_rate"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    parsed = []
    for row in rows:
        item = {"policy": row["policy"]}
        for field in FIELDS:
            if field == "policy":
                continue
            value = row.get(field, "")
            try:
                item[field] = float(value)
            except (TypeError, ValueError):
                item[field] = np.nan
        if not np.isfinite(item["revisit_rate"]):
            entries = item["cell_entries"]
            revisits = item["revisits"]
            item["revisit_rate"] = (
                revisits / entries
                if np.isfinite(entries) and np.isfinite(revisits) and entries > 0
                else np.nan
            )
        parsed.append(item)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-csv",
        type=Path,
        default=ROOT / "evaluation_results" / "episodes.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation_results" / "policy_benchmark",
        help="output path without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_episodes(args.episodes_csv)
    plot_results(rows, args.output)
    print(f"Saved {args.output.with_suffix('.pdf')} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
