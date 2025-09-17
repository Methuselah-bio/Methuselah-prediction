"""
train_nested_cv.py
-------------------

This script implements a nested cross‑validation training routine for the
Methuselah‑prediction project.  Nested cross‑validation provides an
unbiased estimate of generalisation performance by separating the
hyperparameter selection process (inner loop) from the model evaluation
(outer loop).  It is particularly useful when tuning many
hyperparameters or comparing several algorithms, as it prevents
information leakage from the validation set into the test set.

Inspired by best practices in machine learning and bioinformatics,
`train_nested_cv.py` performs the following steps:

1. Loads a processed dataset from the path specified in the YAML
   configuration file.
2. Builds model pipelines and hyperparameter grids as in
   ``train_experiment.py``.  Oversampling is supported when
   ``experiment.use_oversampling`` is enabled and the `imbalanced‑learn`
   package is installed.
3. For each algorithm listed in ``experiment.algorithms``, runs a
   nested cross‑validation loop:
     - The outer loop uses ``nested_cv.outer_folds`` stratified folds.
     - For each outer fold, a grid search is performed on the training
       portion using ``nested_cv.inner_folds`` folds to select the
       best hyperparameters based on macro‑averaged ROC AUC.
     - The best estimator from the inner loop is then evaluated on
       the corresponding outer test fold using several metrics
       (AUROC, AUPRC, accuracy and Brier score).
4. Averages metrics across outer folds and records the selected
   hyperparameters.  The results for all algorithms are written to
   ``results/nested_cv_results.json``.
5. Trains the best performing model on the full dataset with its
   optimal hyperparameters and saves it as ``results/best_model_nested.joblib``.

To use this script, ensure the ``nested_cv`` section of your
configuration file has ``enabled: true`` and specify ``outer_folds`` and
``inner_folds``.  Run:

```
python src/train_nested_cv.py --config configs/base.yaml
```

"""

import argparse
import json
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    make_scorer,
)
from sklearn.preprocessing import label_binarize

# Reuse the model construction logic from train_experiment.py.  We import
# ``build_models`` rather than duplicate its implementation here.  If you
# modify the models in train_experiment.py, the nested CV script will
# automatically reflect those changes.
# We intentionally do not import ``train_experiment`` here because the
# ``src`` directory is not a Python package (it lacks an ``__init__.py``).
# Instead, we replicate the minimal functions needed from
# ``train_experiment.py`` to make this script self‑contained.  These
# functions are copied verbatim (with minor adjustments) to avoid
# import errors when running the script directly.

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(processed_path: str, target_column: str) -> (pd.DataFrame, pd.Series):
    """Load a processed CSV and split it into features and target.

    If the target column is not present, a ValueError is raised.  All
    non‑numeric feature columns are dropped, as the models expect
    numeric inputs.  This behaviour mirrors the corresponding function
    in ``train_experiment.py``.
    """
    data = pd.read_csv(processed_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in processed dataset.")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    return X, y


def build_models(config: Dict[str, Any]):
    """
    Construct model pipelines and hyperparameter grids based on the
    configuration.  This function is adapted from
    ``train_experiment.build_models``.  It returns a dictionary mapping
    algorithm names to tuples of (pipeline, param_grid).  Oversampling
    is handled when ``experiment.use_oversampling`` is set to true and
    `imbalanced‑learn` is available.
    """
    from sklearn.compose import ColumnTransformer  # unused but maintained for parity
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    # Optional oversampling imports
    try:
        from imblearn.over_sampling import RandomOverSampler  # type: ignore
        from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    except ImportError:
        RandomOverSampler = None  # type: ignore
        ImbPipeline = None  # type: ignore
    # Default parameter grids
    default_param_grids = {
        'elastic_net_logreg': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__l1_ratio': [0.0, 0.5, 1.0],
        },
        'svm': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__gamma': ['scale', 'auto'],
        },
        'xgboost': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.1, 0.01],
        },
        'mlp': {
            'clf__hidden_layer_sizes': [(50,), (50, 50), (100,), (100, 50)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [1e-4, 1e-3],
            'clf__learning_rate_init': [1e-3, 1e-2],
        },
    }
    user_param_grids = config.get('experiment', {}).get('param_grids', {})
    models: Dict[str, Any] = {}
    use_oversampling = bool(config.get('experiment', {}).get('use_oversampling', False))

    def numeric_pipeline(clf):
        """Create a pipeline optionally including oversampling and scaling."""
        steps = []
        if use_oversampling and RandomOverSampler is not None:
            steps.append(('oversampler', RandomOverSampler()))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', clf))
        if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
            return ImbPipeline(steps)
        else:
            from sklearn.pipeline import Pipeline
            return Pipeline(steps)

    # Elastic‑net logistic regression
    lr_pipeline = numeric_pipeline(LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000))
    lr_param_grid = {**default_param_grids['elastic_net_logreg'], **user_param_grids.get('elastic_net_logreg', {})}
    models['elastic_net_logreg'] = (lr_pipeline, lr_param_grid)

    # SVM with RBF kernel
    svm_pipeline = numeric_pipeline(SVC(kernel='rbf', probability=True))
    svm_param_grid = {**default_param_grids['svm'], **user_param_grids.get('svm', {})}
    models['svm'] = (svm_pipeline, svm_param_grid)

    # XGBoost (no scaling by default)
    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_jobs=1)
    xgb_steps = []
    if use_oversampling and RandomOverSampler is not None:
        xgb_steps.append(('oversampler', RandomOverSampler()))
    xgb_steps.append(('clf', xgb_clf))
    if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline2  # alias to avoid confusion
        xgb_pipeline = ImbPipeline2(xgb_steps)
    else:
        from sklearn.pipeline import Pipeline
        xgb_pipeline = Pipeline(xgb_steps)
    xgb_param_grid = {**default_param_grids['xgboost'], **user_param_grids.get('xgboost', {})}
    models['xgboost'] = (xgb_pipeline, xgb_param_grid)

    # Random forest
    rf_clf = RandomForestClassifier(random_state=config.get('seed', 42))
    rf_steps = []
    if use_oversampling and RandomOverSampler is not None:
        rf_steps.append(('oversampler', RandomOverSampler()))
    rf_steps.append(('clf', rf_clf))
    if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
        rf_pipeline = ImbPipeline(rf_steps)
    else:
        from sklearn.pipeline import Pipeline
        rf_pipeline = Pipeline(rf_steps)
    default_rf_grid = {
        'clf__n_estimators': [200],
        'clf__max_depth': [None],
        'clf__max_features': ['sqrt'],
    }
    rf_param_grid = {**default_rf_grid, **user_param_grids.get('random_forest', {})}
    models['random_forest'] = (rf_pipeline, rf_param_grid)

    # MLP classifier
    mlp_clf = MLPClassifier(max_iter=300, random_state=config.get('seed', 42))
    mlp_pipeline = numeric_pipeline(mlp_clf)
    default_mlp_grid = default_param_grids['mlp']
    mlp_param_grid = {**default_mlp_grid, **user_param_grids.get('mlp', {})}
    models['mlp'] = (mlp_pipeline, mlp_param_grid)
    return models


def multiclass_roc_auc(y_true, y_proba):
    """
    Compute macro‑averaged one‑vs‑rest ROC AUC for multi‑class targets.
    Binarises the labels and calls ``roc_auc_score`` with ``average='macro'``.
    """
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')


def multiclass_auprc(y_true, y_proba):
    """
    Compute macro‑averaged area under the precision‑recall curve for
    multi‑class targets.
    """
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    return average_precision_score(y_bin, y_proba, average='macro')


def multiclass_brier(y_true, y_proba):
    """
    Compute the multi‑class Brier score.  This generalises the binary
    Brier score by summing squared differences between predicted class
    probabilities and the one‑hot encoded ground truth, then averaging
    across samples.
    """
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    return np.mean(np.sum((y_proba - y_bin) ** 2, axis=1))


def run_nested_cv(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute nested cross‑validation for each algorithm defined in the
    configuration.  Returns a dictionary mapping algorithm names to a
    nested structure containing averaged metrics and the best
    hyperparameters selected on the full dataset.
    """
    # Extract nested CV settings
    nested_settings = config.get('nested_cv', {})
    outer_folds = int(nested_settings.get('outer_folds', 5))
    inner_folds = int(nested_settings.get('inner_folds', 3))
    # Outer and inner cross‑validation objects
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=config.get('seed', 42))
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=config.get('seed', 42))
    # Build models and parameter grids using existing utility
    models = build_models(config)
    algorithms = config.get('experiment', {}).get('algorithms', [])
    results: Dict[str, Any] = {}
    best_model_name: str | None = None
    best_auc = -np.inf
    best_estimator = None
    # Custom scorer for grid search: macro‑averaged ROC AUC
    scorer_grid = make_scorer(multiclass_roc_auc, needs_proba=True)
    # Scoring dictionary for outer evaluation
    scoring_outer = {
        'auroc': make_scorer(multiclass_roc_auc, needs_proba=True),
        'auprc': make_scorer(multiclass_auprc, needs_proba=True),
        'accuracy': 'accuracy',
        'brier': make_scorer(multiclass_brier, greater_is_better=False, needs_proba=True),
    }
    for name, (pipeline, param_grid) in models.items():
        if name not in algorithms:
            continue
        # Inner grid search over hyperparameters
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scorer_grid,
            n_jobs=-1,
        )
        # Perform nested cross‑validation: cross_validate will train the
        # GridSearchCV instance on each outer fold.  For each fold it
        # refits the estimator using the best hyperparameters found in
        # the inner loop and evaluates it on the held‑out test portion.
        cv_results = cross_validate(
            grid_search,
            X,
            y,
            cv=outer_cv,
            scoring=scoring_outer,
            return_estimator=False,
            n_jobs=-1,
        )
        # Compute mean metrics across outer folds
        metrics = {metric: float(np.mean(cv_results[f'test_{metric}'])) for metric in scoring_outer.keys()}
        # Fit grid search on full dataset to retrieve best hyperparameters
        grid_search.fit(X, y)
        best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}
        results[name] = {
            'best_params': best_params,
            'metrics': metrics,
        }
        # Track best model by AUROC
        if metrics['auroc'] > best_auc:
            best_auc = metrics['auroc']
            best_model_name = name
            best_estimator = grid_search.best_estimator_
    # Save best estimator
    return results, best_model_name, best_estimator, best_auc


def main(config_path: str) -> None:
    config = load_config(config_path)
    # Check if nested CV is enabled; if not, exit gracefully
    nested_settings = config.get('nested_cv', {})
    if not nested_settings.get('enabled', False):
        print("Nested cross‑validation is disabled in the configuration.  Set nested_cv.enabled: true to run.")
        return
    # Determine processed file path
    processed_path = config['paths']['processed']
    if os.path.isdir(processed_path):
        processed_file = os.path.join(processed_path, 'processed.csv')
    else:
        processed_file = processed_path
    target_column = config['task']['target']
    X, y = load_dataset(processed_file, target_column)
    # Run nested CV
    results, best_model_name, best_estimator, best_auc = run_nested_cv(X, y, config)
    # Write results to JSON
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'nested_cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    # Save best model
    if best_estimator is not None and best_model_name is not None:
        model_path = os.path.join(results_dir, 'best_model_nested.joblib')
        joblib.dump(best_estimator, model_path)
        print(f"Best algorithm: {best_model_name} (AUROC={best_auc:.3f}). Saved to {model_path}")
        print(f"Nested cross‑validation results saved to {out_path}")
    else:
        print("No models were trained.  Please check the configuration.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nested cross‑validation experiments for aging prediction models.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()
    main(args.config)