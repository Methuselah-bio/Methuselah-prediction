#!/usr/bin/env python3
"""
Train a baseline classifier on the processed yeast dataset.

This script reads configuration from a YAML file, loads the
preprocessed dataset produced by `prepare_data.py`, splits the data
into training/validation/test sets and trains a model specified in the
configuration.  Metrics are computed on the test (and optionally
validation) sets and persisted to JSON.  The trained model is saved
with joblib for later interpretation or inference.

Supported models: logistic regression ('logreg'), random forest ('rf'),
and gradient boosting using XGBoost ('xgboost').  If the name in the
configuration is unknown the script falls back to a logistic
regression.

Usage:
    python train.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # XGBoost not installed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier on yeast data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name: str, params: dict, seed: int):
    """Return an instantiated classifier according to the given name."""
    name_lower = (name or "").lower()
    if name_lower == "rf":
        model = RandomForestClassifier(random_state=seed, **(params or {}))
    elif name_lower == "xgboost" and xgb is not None:
        model = xgb.XGBClassifier(
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            **(params or {})
        )
    else:
        # Default to logistic regression; multi_class='auto' handles multi‑class
        model = LogisticRegression(max_iter=500, multi_class="auto", **(params or {}))
    # Wrap the model with standardisation of features
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model),
    ])
    return pipeline


def compute_metrics(model, X, y):
    """Compute a set of metrics for classification."""
    # Predictions and probabilities
    probs = model.predict_proba(X)
    preds = model.predict(X)
    # Binarise labels for multi‑class metrics
    classes = sorted(set(y))
    y_bin = label_binarize(y, classes=classes)
    # Metric calculations; exceptions handled for cases where metric isn't defined
    metrics = {}
    try:
        auroc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = None
    try:
        auprc = average_precision_score(y_bin, probs, average="macro")
    except Exception:
        auprc = None
    metrics["auroc"] = auroc
    metrics["auprc"] = auprc
    metrics["accuracy"] = accuracy_score(y, preds)
    # Brier score: mean squared difference between predicted probabilities and the one‑hot labels
    try:
        brier = ((probs - y_bin) ** 2).sum(axis=1).mean()
    except Exception:
        brier = None
    metrics["brier"] = brier
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    data_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    target_col = config["task"]["target"]

    df = pd.read_csv(data_path)
    # Drop the sequence name but keep all numeric features and target
    if "Sequence_Name" in df.columns:
        X = df.drop(columns=[target_col, "Sequence_Name"])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]
    # Train/val/test split
    test_size = config["split"].get("test_size", 0.2)
    val_size = config["split"].get("val_size", 0.2)
    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Then optionally split train/val
    if val_size > 0 and val_size < 1.0:
        # Compute relative validation fraction w.r.t. remaining data
        val_frac = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_frac,
            random_state=seed,
            stratify=y_train_val,
        )
    else:
        X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None

    # Instantiate model
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "logreg")
    params = model_cfg.get("params", {})
    model = build_model(model_name, params, seed=seed)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    metrics = {}
    metrics["test"] = compute_metrics(model, X_test, y_test)
    if X_val is not None:
        metrics["val"] = compute_metrics(model, X_val, y_val)

    # Persist results
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Save model pipeline
    model_path = results_dir / "model.joblib"
    joblib.dump(model, model_path)

    print(f"Training complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()