"""
automl.py
---------

This script demonstrates how to leverage AutoML tools to search for
well‑performing models with minimal manual tuning.  It uses FLAML
(Fast Lightweight AutoML) when available to automatically explore
model types and hyperparameters within a time budget.  If FLAML is not
installed, the script falls back to a simple baseline model.

Usage:

```bash
python src/automl.py --config configs/base.yaml --time-budget 60
```

This will run an AutoML search for 60 seconds using the processed
dataset specified in the configuration and report the best model
found.  The final model and its configuration are saved to
``results/automl_model.joblib``.
"""

import argparse
import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from flaml import AutoML  # type: ignore
except ImportError:
    AutoML = None  # type: ignore


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(processed_path: str, target_column: str):
    data = pd.read_csv(processed_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    # Drop the target and retain only numeric feature columns.  AutoML
    # frameworks generally expect numeric inputs.  Non‑numeric columns
    # (e.g., identifiers) are discarded.
    X = data.drop(columns=[target_column])
    y = data[target_column]
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    return X, y


def main(config_path: str, time_budget: int = 60):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    config = load_config(config_path)
    processed_path = config['paths']['processed']
    if os.path.isdir(processed_path):
        processed_file = os.path.join(processed_path, 'processed.csv')
    else:
        processed_file = processed_path
    target_column = config['task']['target']
    X, y = load_dataset(processed_file, target_column)
    # Split into train/test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.get('seed', 42))
    if AutoML is None:
        logging.info("FLAML is not available; falling back to a random forest baseline.")
        model = RandomForestClassifier(random_state=config.get('seed', 42))
        model.fit(X_train, y_train)
        # Compute macro‑averaged one‑vs‑rest ROC AUC for multi‑class
        # problems.  For binary classification, scikit‑learn returns
        # the appropriate scalar.  When probabilities are available,
        # use them; otherwise fall back to class predictions.
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        else:
            # Convert hard predictions into one‑hot probabilities
            preds = model.predict(X_test)
            classes = np.unique(y_train)
            y_prob = np.zeros((len(preds), len(classes)))
            for idx, cls in enumerate(classes):
                y_prob[:, idx] = (preds == cls).astype(float)
        # Binarize labels for multi‑class AUC
        from sklearn.preprocessing import label_binarize  # type: ignore
        y_bin = label_binarize(y_test, classes=np.unique(y_train))
        auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
        logging.info(f"Random forest AUROC (macro OVR): {auc:.3f}")
        best_model = model
        best_params: dict[str, Any] = {}
    else:
        # Run AutoML
        automl = AutoML()
        automl_settings = {
            'time_budget': time_budget,  # seconds
            'metric': 'roc_auc',
            'task': 'classification',
            'log_file_name': 'automl.log',
        }
        logging.info(f"Starting AutoML with time_budget={time_budget}s")
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        y_prob = automl.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.unique(y_train))
        auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
        logging.info(f"AutoML best AUROC (macro OVR): {auc:.3f}; estimator: {automl.best_estimator}")
        best_model = automl.model
        best_params = automl.best_config
    # Save best model and config
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'automl_model.joblib')
    joblib.dump(best_model, model_path)
    params_path = os.path.join(results_dir, 'automl_best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logging.info(f"Best AutoML model saved to {model_path}; parameters saved to {params_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AutoML for classification tasks.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    parser.add_argument('--time-budget', type=int, default=60, help='Time budget in seconds for AutoML search.')
    args = parser.parse_args()
    main(args.config, args.time_budget)