#!/usr/bin/env python3
"""
Prepare the yeast dataset for modelling.

This script reads raw data files from the path specified in a YAML
configuration and produces a processed CSV.  The UCI Yeast dataset
contains a first column with a sequence identifier followed by eight
continuous features and a categorical class label.  To simplify
analysis, this script encodes the categorical target using integer
codes and drops the original label column.  The processed dataset
retains the sequence identifier for reference, the eight numeric
features and the new target column defined by the configuration.

Usage:
    python prepare_data.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare yeast dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    raw_dir = Path(config["paths"]["raw_dir"])
    processed_path = Path(config["paths"]["processed"])

    # Create processed directory if necessary
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Data file names in the raw directory; the UCI dataset ships with
    # `yeast.data` and `yeast.names`.  We care only about `yeast.data`.
    data_file = raw_dir / "yeast.data"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Expected raw data file at {data_file} but it was not found."
        )

    # Column names according to the UCI description
    columns = [
        "Sequence_Name",
        "mcg",
        "gvh",
        "alm",
        "mit",
        "erl",
        "pox",
        "vac",
        "nuc",
        "localization_site",
    ]

    # Read the dataset using whitespace as the delimiter
    df = pd.read_csv(data_file, sep="\s+", header=None, names=columns)

    # Encode the categorical target into integer codes under the name
    # specified in the configuration.  This makes downstream models
    # agnostic to the original string labels.
    target_col = config["task"]["target"]
    df[target_col] = df["localization_site"].astype("category").cat.codes

    # Drop the original string target column
    df = df.drop(columns=["localization_site"])

    # Save processed data
    df.to_csv(processed_path, index=False)
    print(f"Processed data written to {processed_path}")


if __name__ == "__main__":
    main()