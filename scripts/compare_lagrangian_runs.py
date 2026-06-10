"""Compare Lagrangian walk-forward runs and print a summary table.

Usage:
    python scripts/compare_lagrangian_runs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

RUNS: dict[str, Path] = {
    "lagrangian_v1 (8d, 4s, lr=1e-3, p=15)": ROOT / "walk_forward_summary_lagrangian_v1.csv",
    "lagrangian_v2 (8d, 4s, lr=5e-4, p=30)": ROOT / "walk_forward_summary_lagrangian_v2.csv",
    "lagrangian_v3 (16d, 8s, lr=5e-4, p=30)": ROOT / "walk_forward_summary_lagrangian_v3.csv",
    "lagrangian_v4 (32d, 8s, lr=5e-4, p=30)": ROOT / "walk_forward_summary_lagrangian_v4.csv",
}

# Baselines for context
BASELINES: dict[str, tuple[float, float, float]] = {
    "LSTM  (hidden=64)": (0.4062, 0.6152, 0.1477),
    "GRU   (hidden=64)": (0.4008, 0.6071, 0.1477),
    "NODE  (hidden=64)": (0.3850, float("nan"), float("nan")),
}


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  [missing] {path.name}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} folds from {path.name}")
    return df


def summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "mean_macro_f1": df["macro_f1"].mean(),
        "std_macro_f1": df["macro_f1"].std(),
        "mean_brier_score": df["brier_score"].mean(),
        "mean_ece": df["ece"].mean(),
        "n_folds": len(df),
    }


def fold_delta(base: pd.DataFrame, new: pd.DataFrame, label: str) -> None:
    if base.empty or new.empty or len(base) != len(new):
        return
    delta = new["macro_f1"].values - base["macro_f1"].values
    print(f"\n  Fold-wise F1 delta ({label} vs v2):")
    print(f"    Mean   : {delta.mean():+.4f}")
    print(f"    Median : {float(pd.Series(delta).median()):+.4f}")
    print(f"    Better : {(delta > 0).sum()} / {len(delta)} folds")
    print(f"    Early folds  0-9  : {delta[:10].mean():+.4f}")
    print(f"    Later folds 50-70 : {delta[50:].mean():+.4f}")


def main() -> None:
    print("\nLoading results...")
    dfs = {label: load(path) for label, path in RUNS.items()}

    rows = []
    for label, df in dfs.items():
        s = summarise(df)
        if s:
            rows.append({"Config": label, **s})

    print("\n=== Lagrangian Run Comparison ===\n")
    if rows:
        summary = pd.DataFrame(rows).set_index("Config")
        display = summary.rename(columns={
            "mean_macro_f1": "Mean F1",
            "std_macro_f1": "Std F1",
            "mean_brier_score": "Mean Brier",
            "mean_ece": "Mean ECE",
            "n_folds": "N Folds",
        })
        print(display.to_string(float_format="{:.4f}".format))

    print("\n=== Baselines (for context) ===\n")
    baseline_rows = [
        {"Config": k, "Mean F1": f1, "Mean Brier": b, "Mean ECE": e}
        for k, (f1, b, e) in BASELINES.items()
    ]
    print(pd.DataFrame(baseline_rows).set_index("Config").to_string(float_format="{:.4f}".format))

    # Fold-wise deltas vs v2
    v2_df = dfs.get("lagrangian_v2 (8d, 4s, lr=5e-4, p=30)", pd.DataFrame())
    v3_df = dfs.get("lagrangian_v3 (16d, 8s, lr=5e-4, p=30)", pd.DataFrame())
    v4_df = dfs.get("lagrangian_v4 (32d, 8s, lr=5e-4, p=30)", pd.DataFrame())

    if not v2_df.empty:
        print("\n=== Fold-wise deltas vs v2 ===")
        fold_delta(v2_df, v3_df, "v3")
        fold_delta(v2_df, v4_df, "v4")


if __name__ == "__main__":
    main()
