#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for processed datasets.

This script performs a simple yet informative exploratory analysis of
any processed CSV produced by ``prepare_data.py``.  It computes
summary statistics of all numeric features, visualises pairwise
correlations and generates a two‑dimensional embedding of the data
for intuitive inspection.  The goal is to help users understand
their datasets before training a predictive model — a crucial step
in bioinformatics to uncover batch effects, class imbalance and
feature distributions.

Outputs are written into the directory specified by
``paths.results`` in the YAML configuration:

* ``eda_stats.json`` – a JSON file containing per‑feature summary
  statistics (mean, std, min, 25th percentile, median, 75th
  percentile, max).
* ``eda_correlation.png`` – a heatmap visualising the Pearson
  correlation matrix of numeric features.
* ``eda_embedding.png`` – a scatter plot of a two‑dimensional
  embedding computed via UMAP if available, otherwise t‑SNE.  Points
  are coloured by the target variable.

Example usage:

    python src/eda.py --config configs/base.yaml

If the optional package ``umap-learn`` is installed, the script will
use UMAP to compute the embedding; otherwise it falls back to t‑SNE
from scikit‑learn.  Both methods can fail on very small or singular
datasets; in such cases the embedding step is skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # use non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap  # type: ignore
except ImportError:
    umap = None  # type: ignore

from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory data analysis for processed datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(processed_path)
    target_col = config["task"]["target"]
    # Select numeric feature columns; drop the target and any non‑numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    X = df[numeric_cols]
    y = df[target_col]

    # Compute summary statistics
    stats = X.describe().T.to_dict()
    stats_path = results_dir / "eda_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics written to {stats_path}")

    # Plot correlation heatmap
    if X.shape[1] >= 2:
        corr = X.corr().astype(float)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        heatmap_path = results_dir / "eda_correlation.png"
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Correlation heatmap saved to {heatmap_path}")
    else:
        print("Not enough numeric features to compute correlation heatmap.")

    # Compute embedding for visual inspection
    embedding = None
    embedding_method = ""
    try:
        if umap is not None and X.shape[1] > 2:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=config.get("seed", 42))
            embedding = reducer.fit_transform(X.values)
            embedding_method = "UMAP"
        elif X.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=config.get("seed", 42))
            embedding = tsne.fit_transform(X.values)
            embedding_method = "t-SNE"
    except Exception as exc:
        print(f"Embedding computation failed: {exc}")
        embedding = None

    if embedding is not None:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="tab20", alpha=0.8, s=30)
        plt.colorbar(scatter, label=target_col)
        plt.title(f"{embedding_method} embedding of samples")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        embed_path = results_dir / "eda_embedding.png"
        plt.savefig(embed_path)
        plt.close()
        print(f"Embedding plot saved to {embed_path}")
    else:
        print("Embedding not computed due to insufficient features or errors.")


if __name__ == "__main__":
    main()