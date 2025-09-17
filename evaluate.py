#!/usr/bin/env python3
"""
Evaluate a previously trained model on the full processed dataset.

This script reloads the trained pipeline and the processed dataset,
computes metrics for the entire dataset and writes them into a JSON
file within the results directory.  If you want to compute metrics
specific to test or validation splits, use `train.py` which records
those during training.

Usage:
    python evaluate.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def compute_metrics(model, X, y):
    probs = model.predict_proba(X)
    preds = model.predict(X)
    classes = sorted(set(y))
    y_bin = label_binarize(y, classes=classes)
    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = average_precision_score(y_bin, probs, average="macro")
    except Exception:
        metrics["auprc"] = None
    metrics["accuracy"] = accuracy_score(y, preds)
    try:
        metrics["brier"] = ((probs - y_bin) ** 2).sum(axis=1).mean()
    except Exception:
        metrics["brier"] = None
    return metrics


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    model_path = results_dir / "model.joblib"
    metrics_path = results_dir / "metrics_full.json"

    df = pd.read_csv(processed_path)
    target_col = config["task"]["target"]
    if "Sequence_Name" in df.columns:
        X = df.drop(columns=[target_col, "Sequence_Name"])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]

    model = joblib.load(model_path)
    metrics = compute_metrics(model, X, y)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Full‑dataset metrics written to {metrics_path}")


if __name__ == "__main__":
    main()