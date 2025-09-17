#!/usr/bin/env python3
"""
interpret_shap.py
------------------

This script provides model interpretability using SHAP (SHapley
Additive exPlanations).  Unlike permutation importance, SHAP values
attribute the contribution of each feature to individual predictions,
offering both global and local insights into the model’s behaviour.

Given a trained classifier saved in the results directory and a
processed dataset specified in the configuration, the script
computes SHAP values on a representative subset of the data and
produces a summary bar plot.  If the optional ``shap`` library is
installed, tree‑based models are handled using ``TreeExplainer``; for
non‑tree models, it attempts to use ``KernelExplainer``.  When
dependencies are missing, the script prints a message and exits.

Outputs:
  * ``shap_summary.png`` – bar chart of mean absolute SHAP values per
    feature, saved into the results directory specified in the config.

Note that computing SHAP values can be computationally expensive for
large datasets.  To mitigate this, the script samples up to 200
instances randomly.  Adjust the ``--max-samples`` argument to change
the number of instances used for estimation.

Usage:

    python src/interpret_shap.py --config configs/base.yaml [--max-samples 200]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Attempt to import shap; this package may not be installed in all
# environments.  SHAP is optional and must be installed separately.
try:
    import shap  # type: ignore
except ImportError:
    shap = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for a trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to use for SHAP value estimation",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    if shap is None:
        print(
            "The 'shap' package is not installed. Please install it with 'pip install shap' "
            "to use this script."
        )
        return
    config = load_config(args.config)
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    target_col = config["task"]["target"]

    # Load data and model
    df = pd.read_csv(processed_path)
    # Drop non‑numeric and target columns for features
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Remove non‑numeric features (e.g., Sequence_Name)
    X = X.select_dtypes(include=[np.number])
    model_path_candidates = [results_dir / "best_model.joblib", results_dir / "stacking_model.joblib", results_dir / "model.joblib"]
    model_path = None
    for cand in model_path_candidates:
        if cand.exists():
            model_path = cand
            break
    if model_path is None:
        print("No trained model found in results directory; please train a model first.")
        return
    model = joblib.load(model_path)

    # Sample data to speed up SHAP
    if args.max_samples < len(X):
        sample_indices = np.random.RandomState(config.get("seed", 42)).choice(len(X), size=args.max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X

    # Determine appropriate explainer
    explainer = None
    shap_values = None
    try:
        # Tree‑based models
        if hasattr(model, "predict_proba") and (
            "xgb" in model.__class__.__module__
            or "sklearn.ensemble" in model.__class__.__module__
            or "forest" in model.__class__.__name__.lower()
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Fallback to KernelExplainer
            # Use a small background dataset for KernelExplainer
            background = shap.sample(X_sample, min(50, len(X_sample)), random_state=config.get("seed", 42))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        print(f"Failed to compute SHAP values: {exc}")
        return

    # shap_values may be a list for multi‑class problems; compute mean absolute values across classes
    if isinstance(shap_values, list):
        # For multi‑class, sum absolute contributions across classes
        abs_vals = np.mean([np.abs(vals) for vals in shap_values], axis=0)
    else:
        abs_vals = np.abs(shap_values)
    # Compute mean absolute SHAP value per feature
    mean_abs = abs_vals.mean(axis=0)
    feature_names = X_sample.columns
    # Sort features by importance
    sorted_indices = np.argsort(mean_abs)[::-1]
    sorted_features = feature_names[sorted_indices]
    sorted_importances = mean_abs[sorted_indices]

    # Plot bar chart
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_features)), sorted_importances, color="orchid")
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    out_path = results_dir / "shap_summary.png"
    plt.savefig(out_path)
    plt.close()
    print(f"SHAP summary plot saved to {out_path}")


if __name__ == "__main__":
    main()