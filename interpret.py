#!/usr/bin/env python3
"""
Compute permutation feature importances for the trained model and save
a bar chart.

Permutation importance assesses the contribution of each feature to
model performance by shuffling the feature and measuring the drop in
performance.  This script reports the mean importance across a small
number of shuffles and writes a PNG plot to the results directory.

Usage:
    python interpret.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permutation importance analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of permutations to average over",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    model_path = results_dir / "model.joblib"
    plot_path = results_dir / "feature_importance.png"
    # Load data and model
    df = pd.read_csv(processed_path)
    target_col = config["task"]["target"]
    if "Sequence_Name" in df.columns:
        X = df.drop(columns=[target_col, "Sequence_Name"])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]
    model = joblib.load(model_path)
    # Compute permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=args.n_repeats, random_state=config.get("seed", 42), n_jobs=1
    )
    importances = result.importances_mean
    # Sort importances descending
    indices = np.argsort(importances)[::-1]
    features = X.columns[indices]
    sorted_importances = importances[indices]
    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(features)), sorted_importances, color="skyblue")
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")


if __name__ == "__main__":
    main()