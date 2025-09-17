#!/usr/bin/env python3
"""
stacking_ensemble.py
---------------------

This script trains a stacking ensemble classifier to improve predictive
performance by combining the strengths of several base learners.  The
approach follows best practices from aging research by evaluating
multiple models and integrating them into a single meta‑learner.  Base
models include elastic‑net logistic regression, random forest,
XGBoost and multi‑layer perceptron.  A logistic regression model
serves as the meta‑classifier.  Cross‑validated metrics (AUROC, AUPRC,
accuracy and Brier score) are reported, and the trained ensemble is
saved for future use.  Results are written to ``results/ensemble_results.json``.

Usage:
    python src/stacking_ensemble.py --config configs/base.yaml

The configuration file supplies the processed dataset path and the target
column name.  The script does not rely on the ``experiment`` section of the
configuration but uses fixed hyperparameters for simplicity.  You can
modify the base learners or meta‑learner parameters in the code below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(processed_path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed CSV and return features and target."""
    df = pd.read_csv(processed_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in processed dataset.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Keep numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    return X[numeric_cols], y


def build_stacking_classifier(seed: int = 42) -> StackingClassifier:
    """Construct the stacking ensemble with default hyperparameters."""
    # Base estimators with preprocessing where appropriate
    # Elastic‑net logistic regression
    lr_base = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=3000)
    )
    # Random forest
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=None, max_features="sqrt", random_state=seed)
    # XGBoost (no scaling needed)
    xgb_base = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=1,
        random_state=seed,
    )
    # Multi‑layer perceptron
    mlp_base = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=300, random_state=seed)
    )
    estimators = [
        ("elastic_net_logreg", lr_base),
        ("random_forest", rf_base),
        ("xgboost", xgb_base),
        ("mlp", mlp_base),
    ]
    # Meta‑learner: logistic regression with regularization
    meta_clf = LogisticRegression(max_iter=5000)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=1,
        passthrough=False,
    )


def compute_metrics(model: StackingClassifier, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Dict[str, float]:
    """Compute cross‑validated metrics for the ensemble."""
    # Define scoring functions for multi‑class
    def multiclass_roc_auc(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")

    def multiclass_auprc(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return average_precision_score(y_bin, y_proba, average="macro")

    def multiclass_brier(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return np.mean(np.sum((y_proba - y_bin) ** 2, axis=1))

    scoring = {
        "auroc": lambda est, X_fold, y_fold: multiclass_roc_auc(y_fold, est.predict_proba(X_fold)),
        "auprc": lambda est, X_fold, y_fold: multiclass_auprc(y_fold, est.predict_proba(X_fold)),
        "accuracy": lambda est, X_fold, y_fold: accuracy_score(y_fold, est.predict(X_fold)),
        "brier": lambda est, X_fold, y_fold: multiclass_brier(y_fold, est.predict_proba(X_fold)),
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    # Aggregate mean scores
    return {
        "auroc": float(np.mean(cv_results["test_auroc"])),
        "auprc": float(np.mean(cv_results["test_auprc"])),
        "accuracy": float(np.mean(cv_results["test_accuracy"])),
        "brier": float(np.mean(cv_results["test_brier"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stacking ensemble classifier")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    # Load configuration and dataset
    config = load_config(args.config)
    processed_path = Path(config["paths"]["processed"])
    target_col = config["task"]["target"]
    X, y = load_dataset(processed_path, target_col)
    # Build model
    seed = int(config.get("seed", 42))
    ensemble = build_stacking_classifier(seed=seed)
    # Cross‑validation
    cv = StratifiedKFold(n_splits=int(config.get("experiment", {}).get("cv_folds", 5)), shuffle=True, random_state=seed)
    print("[Ensemble] Performing cross‑validated evaluation...")
    metrics = compute_metrics(ensemble, X, y, cv)
    print("[Ensemble] Cross‑validated metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    # Fit on full data and save model
    print("[Ensemble] Training on full dataset and saving model...")
    ensemble.fit(X, y)
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "stacking_model.joblib"
    joblib.dump(ensemble, model_path)
    # Write metrics to JSON
    results_json = results_dir / "ensemble_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump({"stacking_ensemble": metrics}, f, indent=2)
    print(f"[Ensemble] Metrics saved to {results_json}")
    print(f"[Ensemble] Model saved to {model_path}")


if __name__ == "__main__":
    main()
