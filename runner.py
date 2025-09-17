#!/usr/bin/env python3
"""
runner.py
---------

This script orchestrates the full pipeline for the Methuselah‑prediction
project.  When executed, it sequentially preprocesses raw yeast data,
merges any processed datasets, trains models via cross‑validated
experiments, performs feature selection and visualizes results.  The
runner is intended to automate workflow when new datasets are added to
the ``data/raw`` directory.  By reusing the existing modular
components, the pipeline can recursively improve model performance on
larger combined datasets.

Usage:
    python src/runner.py --config configs/base.yaml

The script assumes the configuration file defines ``paths.raw_dir``,
``paths.processed`` and ``paths.results``.  It respects the
``experiment`` section for algorithm selection and hyperparameters.

The runner performs the following steps:

1. **Preparation**: invokes ``prepare_data.py`` to convert raw data in
   ``data/raw`` into a processed CSV.
2. **Merging**: uses ``merge_datasets.py`` to combine all processed CSVs
   in ``data/processed`` into a single ``processed.csv`` (if more than one
   is present).
3. **Training**: runs ``train_experiment.py`` to evaluate algorithms
   specified in the configuration via stratified cross‑validation and
   hyperparameter grid search.  Results are written to
   ``results/experiment_results.json`` and the best model is saved to
   ``results/best_model.joblib``.
4. **Feature Selection**: executes ``feature_selection.py`` to rank
   features using an ANOVA F‑test and generate a bar chart.
5. **Visualization**: calls ``visualize_experiment.py`` to produce a
   grouped bar chart summarizing cross‑validated metrics across
   algorithms.

Each step logs progress to standard output.  If any subprocess returns
a non‑zero exit code, execution stops and the error is reported.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def run_command(cmd: list[str], description: str) -> None:
    """Execute a subprocess command and handle errors."""
    print(f"\n[Runner] {description}...", flush=True)
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({description}): {cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Methuselah pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    # Validate configuration path
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # Load config to retrieve paths
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    # Step 1: preprocess raw data
    # Always run prepare_data.py to ensure the processed CSV is up to date.
    run_command([
        sys.executable,
        str(Path(__file__).resolve().parent / "prepare_data.py"),
        "--config",
        str(config_path),
    ], description="Preparing data")
    # Step 2: merge processed datasets if multiple files present
    processed_dir = processed_path.parent if processed_path.is_file() else processed_path
    csv_files = list(processed_dir.glob("*.csv"))
    if len(csv_files) > 1:
        # Use merge_datasets to combine all CSVs into processed.csv
        merge_input_dir = processed_dir
        merge_output_file = processed_dir / "processed.csv"
        run_command(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "merge_datasets.py"),
                "--input-dir",
                str(merge_input_dir),
                "--output-file",
                str(merge_output_file),
                "--target",
                config["task"]["target"],
            ],
            description="Merging processed datasets",
        )
    # Step 3: run cross‑validated experiment
    run_command(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_experiment.py"),
            "--config",
            str(config_path),
        ],
        description="Training cross‑validated models",
    )
    # Step 4: feature selection
    run_command(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "feature_selection.py"),
            "--config",
            str(config_path),
            "--k",
            "10",
        ],
        description="Running feature selection",
    )
    # Step 5: visualize experiment results
    experiment_results = results_dir / "experiment_results.json"
    if experiment_results.exists():
        run_command(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "visualize_experiment.py"),
                "--results-file",
                str(experiment_results),
                "--output-file",
                str(results_dir / "experiment_summary.png"),
            ],
            description="Visualizing experiment results",
        )
    else:
        print(
            f"Experiment results file not found at {experiment_results}. Skipping visualization.",
            file=sys.stderr,
        )
    print("\n[Runner] Pipeline complete. Check the 'results' directory for outputs.")


if __name__ == "__main__":
    main()