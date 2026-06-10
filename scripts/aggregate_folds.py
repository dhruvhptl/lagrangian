"""Aggregate metrics.json files from results/folds/fold_*/ into a single CSV.

Usage:
    python scripts/aggregate_folds.py
    python scripts/aggregate_folds.py --folds-dir results/folds --out results/all_folds_metrics.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-dir", default="results/folds", help="Directory containing fold_*/ subdirs")
    parser.add_argument("--out", default="results/all_folds_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    folds_dir = Path(args.folds_dir)
    if not folds_dir.exists():
        raise SystemExit(f"Folds directory not found: {folds_dir}")

    records = []
    for metrics_file in sorted(folds_dir.glob("fold_*/metrics.json")):
        with open(metrics_file) as f:
            records.append(json.load(f))

    if not records:
        raise SystemExit(f"No metrics.json files found under {folds_dir}")

    df = pd.DataFrame(records).sort_values("fold_id").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} folds to {out_path}\n")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "fold_id"]

    summary = df[numeric_cols].agg(["mean", "std", "min", "max"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
