#!/usr/bin/env python3
"""
Perform univariate feature selection on the processed dataset.

This script selects the top K features based on univariate statistical
tests (ANOVA F‑test) and visualizes their importance scores.  It can be
used as a quick diagnostic to identify the most informative features
before running more complex models.

Example usage:

    python src/feature_selection.py --config configs/base.yaml --k 8

The script prints the top features and saves a bar chart to
``results/feature_selection.png``.
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Univariate feature selection")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of top features to select"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    processed_path = config["paths"]["processed"]
    # Determine processed file path (similar logic as in train_experiment.py)
    if os.path.isdir(processed_path):
        processed_file = os.path.join(processed_path, "processed.csv")
    else:
        processed_file = processed_path
    target_col = config["task"]["target"]
    df = pd.read_csv(processed_file)
    # Drop non‑numeric features
    if "Sequence_Name" in df.columns:
        X = df.drop(columns=[target_col, "Sequence_Name"])
    else:
        X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    # Perform feature selection
    k = min(args.k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    # Get top feature names and scores
    feature_scores = list(zip(X.columns, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_scores[:k]
    print("Top features (ANOVA F‑scores):")
    for feat, score in top_features:
        print(f"{feat}: {score:.3f}")
    # Plot bar chart
    names = [f[0] for f in top_features]
    vals = [f[1] for f in top_features]
    plt.figure(figsize=(10, 4))
    plt.bar(range(k), vals, color="salmon")
    plt.xticks(range(k), names, rotation=45, ha="right")
    plt.ylabel("F‑score")
    plt.title(f"Top {k} Features by ANOVA F‑score")
    plt.tight_layout()
    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "feature_selection.png")
    plt.savefig(out_path)
    print(f"Feature selection plot saved to {out_path}")


if __name__ == "__main__":
    main()