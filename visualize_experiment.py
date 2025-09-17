#!/usr/bin/env python3
"""
Visualize the results of a cross‑validated experiment.

This script reads the JSON results produced by ``train_experiment.py``
(`results/experiment_results.json` by default) and generates a bar chart
showing the performance metrics of each algorithm.  It saves the plot as
``results/experiment_summary.png``.

Usage:

    python src/visualize_experiment.py --results-file results/experiment_results.json

You can optionally specify an output path for the plot via ``--output-file``.
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment metrics across algorithms")
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/experiment_results.json",
        help="Path to JSON file containing cross‑validated metrics",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/experiment_summary.png",
        help="Path for saving the resulting plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load results
    with open(args.results_file, "r") as f:
        results = json.load(f)
    algorithms = list(results.keys())
    # Extract metric names from the metrics subdict of the first algorithm
    first_algo = next(iter(results.values()))
    if isinstance(first_algo, dict) and "metrics" in first_algo:
        metrics = list(first_algo["metrics"].keys())
    else:
        raise ValueError("JSON structure does not contain a 'metrics' key for each algorithm")
    # Create grouped bar chart: each group corresponds to an algorithm and contains bars for each metric
    n_algos = len(algorithms)
    n_metrics = len(metrics)
    x = np.arange(n_algos)
    width = 0.8 / n_metrics
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [results[algo]["metrics"][metric] for algo in algorithms]
        plt.bar(x + i * width, values, width=width, label=metric)
    plt.xticks(x + width * (n_metrics - 1) / 2, algorithms, rotation=45, ha="right")
    plt.ylabel("Metric score")
    plt.title("Cross‑validated experiment performance across algorithms")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    plt.savefig(args.output_file)
    print(f"Experiment summary plot saved to {args.output_file}")


if __name__ == "__main__":
    main()