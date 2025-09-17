"""
train_experiment.py
--------------------

This module implements an experimental training routine inspired by recent
advances in aging research.  It performs stratified cross‑validation over
several candidate machine‑learning algorithms, selecting hyperparameters via
grid search and reporting the average predictive performance on each fold.

The approach draws on recent recommendations in the aging literature: using
elastic‑net penalized logistic regression and other nonlinear models, tuning
hyperparameters with cross‑validation, and evaluating performance using AUROC
and AUPRC rather than relying solely on accuracy【348516617573011†L170-L198】【348516617573011†L948-L959】.  Feature
selection and proper model evaluation on held‑out data have been shown to
improve predictive performance【257543399768406†L260-L329】.  While the default configuration
uses a modest hyperparameter grid for demonstration purposes, you can
extend the parameter grids in the configuration file.

Usage:
    python train_experiment.py --config configs/base.yaml

The script writes a JSON file `results/experiment_results.json` with the
cross‑validated metrics and selected hyperparameters for each algorithm.
It also trains the best‑performing model on the full dataset and stores
it as `results/best_model.joblib`.
"""

import argparse
import json
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    make_scorer,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# Optional: oversampling support for imbalanced datasets.  When enabled in
# the configuration (experiment.use_oversampling: true), training
# pipelines will incorporate a RandomOverSampler step to balance class
# frequencies.  If imbalanced‑learn is not installed, the oversampling
# step is silently skipped.
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover
    RandomOverSampler = None  # type: ignore
    ImbPipeline = None  # type: ignore


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(processed_path: str, target_column: str) -> (pd.DataFrame, pd.Series):
    data = pd.read_csv(processed_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in processed dataset.")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    # Drop non‑numeric columns.  The processed yeast dataset retains
    # an identifier column (Sequence_Name) that contains strings.
    # Machine‑learning algorithms used in this experiment expect numeric
    # feature arrays.  Remove any columns with non‑numeric dtypes.
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    return X, y


def build_models(config: Dict[str, Any]):
    """Create a dictionary of model pipelines and parameter grids.

    Models include elastic‑net logistic regression, SVM with RBF kernel and
    XGBoost.  Parameter grids can be overridden by values in the config file.
    """
    # Default parameter grids
    default_param_grids = {
        'elastic_net_logreg': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__l1_ratio': [0.0, 0.5, 1.0]
        },
        'svm': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__gamma': ['scale', 'auto']
        },
        'xgboost': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.1, 0.01]
        },
        # Multi‑layer perceptron classifier.  We explore a small set of
        # architectures, activation functions and regularization values.
        'mlp': {
            'clf__hidden_layer_sizes': [(50,), (50, 50), (100,), (100, 50)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [1e-4, 1e-3],
            'clf__learning_rate_init': [1e-3, 1e-2]
        }
    }
    # Override with user‑specified grids
    user_param_grids = config.get('experiment', {}).get('param_grids', {})
    # Build model pipelines
    models = {}
    # Common numeric preprocessing: standardize continuous features
    use_oversampling = bool(config.get('experiment', {}).get('use_oversampling', False))

    def numeric_pipeline(clf):
        """
        Construct a preprocessing and classifier pipeline.  If
        ``use_oversampling`` is enabled and imbalanced‑learn is available,
        prepend a RandomOverSampler step to rebalance the classes on
        each cross‑validation fold.  StandardScaler is applied to
        continuous features prior to classification.
        """
        steps = []
        # Insert oversampler first so that scaling and modeling operate on
        # the resampled dataset.  This step is skipped if imbalanced‑learn
        # is unavailable or oversampling is disabled.
        if use_oversampling and RandomOverSampler is not None:
            steps.append(('oversampler', RandomOverSampler()))
        steps.append(('scaler', StandardScaler()))
        steps.append(('clf', clf))
        # Choose appropriate pipeline class: use imblearn.pipeline if
        # oversampling is required, otherwise fall back to sklearn.pipeline.
        if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
            return ImbPipeline(steps)
        else:
            from sklearn.pipeline import Pipeline  # Local import to avoid circular import
            return Pipeline(steps)

    # Elastic‑net logistic regression (saga solver supports elastic net)
    lr_pipeline = numeric_pipeline(
        LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000)
    )
    lr_param_grid = {**default_param_grids['elastic_net_logreg'], **user_param_grids.get('elastic_net_logreg', {})}
    models['elastic_net_logreg'] = (lr_pipeline, lr_param_grid)

    # SVM with RBF kernel
    svm_pipeline = numeric_pipeline(
        SVC(kernel='rbf', probability=True)
    )
    svm_param_grid = {**default_param_grids['svm'], **user_param_grids.get('svm', {})}
    models['svm'] = (svm_pipeline, svm_param_grid)

    # XGBoost
    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_jobs=1)
    # XGBoost often performs well without scaling
    # XGBoost pipeline.  Scaling is not required; oversampling can be
    # inserted when requested.
    xgb_steps = []
    if use_oversampling and RandomOverSampler is not None:
        xgb_steps.append(('oversampler', RandomOverSampler()))
    xgb_steps.append(('clf', xgb_clf))
    if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
        xgb_pipeline = ImbPipeline(xgb_steps)
    else:
        from sklearn.pipeline import Pipeline
        xgb_pipeline = Pipeline(xgb_steps)
    xgb_param_grid = {**default_param_grids['xgboost'], **user_param_grids.get('xgboost', {})}
    models['xgboost'] = (xgb_pipeline, xgb_param_grid)

    # Random Forest
    # The random forest classifier is robust to feature scaling, so we do not
    # apply StandardScaler here.  Use the parameter grid defined in the
    # configuration file or fall back to reasonable defaults.  Note: Random
    # forests can handle categorical features encoded as integers but may
    # perform poorly if the classes are not encoded ordinally.  Ensure that
    # categorical variables are appropriately encoded in the processed dataset.
    rf_clf = RandomForestClassifier(random_state=config.get('seed', 42))
    # Random forest pipeline.  Include oversampler when enabled.  Scaling is
    # generally unnecessary for tree‑based models.
    rf_steps = []
    if use_oversampling and RandomOverSampler is not None:
        rf_steps.append(('oversampler', RandomOverSampler()))
    rf_steps.append(('clf', rf_clf))
    if use_oversampling and RandomOverSampler is not None and ImbPipeline is not None:
        rf_pipeline = ImbPipeline(rf_steps)
    else:
        from sklearn.pipeline import Pipeline  # Local import
        rf_pipeline = Pipeline(rf_steps)
    default_rf_grid = {
        'clf__n_estimators': [200],
        'clf__max_depth': [None],
        'clf__max_features': ['sqrt']
    }
    rf_param_grid = {**default_rf_grid, **user_param_grids.get('random_forest', {})}
    models['random_forest'] = (rf_pipeline, rf_param_grid)

    # Multi‑layer Perceptron (MLP) classifier
    # The MLP classifier benefits from feature scaling; we wrap it in the
    # numeric pipeline.  Use the parameter grid defined above or override via
    # the configuration file under ``param_grids.mlp``.  Note that MLP
    # training can be sensitive to the ``max_iter`` parameter; we set a
    # relatively high number of iterations to encourage convergence.
    mlp_clf = MLPClassifier(max_iter=300, random_state=config.get('seed', 42))
    mlp_pipeline = numeric_pipeline(mlp_clf)
    default_mlp_grid = default_param_grids['mlp']
    mlp_param_grid = {**default_mlp_grid, **user_param_grids.get('mlp', {})}
    models['mlp'] = (mlp_pipeline, mlp_param_grid)

    return models


def compute_metrics(estimator, X, y, cv):
    """
    Compute cross‑validated metrics: AUROC, AUPRC, accuracy and Brier score.
    Returns a dictionary with the mean score across folds for each metric.
    """
    # Define custom scoring functions that handle multi‑class targets.  We
    # compute macro‑averaged one‑vs‑rest ROC AUC and average precision
    # scores by binarizing the labels.  The Brier score is generalized
    # to multi‑class by summing squared errors across all classes.  Each
    # function expects ``y_true`` and the probability estimates as
    # provided by the estimator's ``predict_proba`` method.
    def multiclass_roc_auc(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')

    def multiclass_auprc(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return average_precision_score(y_bin, y_proba, average='macro')

    def multiclass_brier(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return np.mean(np.sum((y_proba - y_bin) ** 2, axis=1))

    scoring = {
        'auroc': make_scorer(multiclass_roc_auc, needs_proba=True),
        'auprc': make_scorer(multiclass_auprc, needs_proba=True),
        'accuracy': 'accuracy',
        'brier': make_scorer(multiclass_brier, greater_is_better=False, needs_proba=True),
    }
    cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    metrics = {metric: float(np.mean(cv_results[f'test_{metric}'])) for metric in scoring.keys()}
    return metrics


def main(config_path: str):
    config = load_config(config_path)
    processed_path = config['paths']['processed']
    # Determine if processed path is a directory or file
    if os.path.isdir(processed_path):
        # If directory, assume there is a file named processed.csv
        processed_file = os.path.join(processed_path, 'processed.csv')
    else:
        processed_file = processed_path
    target_column = config['task']['target']
    X, y = load_dataset(processed_file, target_column)
    cv_folds = config.get('experiment', {}).get('cv_folds', 5)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.get('seed', 42))
    models = build_models(config)

    results = {}
    best_model_name = None
    best_auc = -np.inf
    best_estimator = None
    # Define a custom scoring function for grid search that handles multi‑class
    # ROC AUC using a macro‑averaged one‑vs‑rest strategy.  Without this,
    # scikit‑learn's default 'roc_auc' scorer only supports binary targets.
    def multiclass_roc_auc(y_true, y_proba):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        return roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')

    scorer_grid = make_scorer(multiclass_roc_auc, needs_proba=True)

    for name, (pipeline, param_grid) in models.items():
        if name not in config.get('experiment', {}).get('algorithms', []):
            continue  # skip algorithms not specified
        # Perform grid search to select hyperparameters based on ROC AUC
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer_grid,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        # Compute metrics with best estimator
        metrics = compute_metrics(grid_search.best_estimator_, X, y, cv)
        results[name] = {
            'best_params': {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()},
            'metrics': metrics
        }
        # Update best model
        if metrics['auroc'] > best_auc:
            best_auc = metrics['auroc']
            best_model_name = name
            best_estimator = grid_search.best_estimator_
    # Save results to JSON
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    # Train best model on the full dataset and save
    if best_estimator is not None:
        model_path = os.path.join(results_dir, 'best_model.joblib')
        joblib.dump(best_estimator, model_path)
        print(f"Best model: {best_model_name} (AUROC={best_auc:.3f}). Saved to {model_path}")
        print(f"Cross‑validated metrics saved to {results_path}")
    else:
        print("No models were trained. Please check the 'algorithms' section of your config.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross‑validated experiments for aging prediction models.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()
    main(args.config)