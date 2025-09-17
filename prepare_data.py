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
import logging
from typing import List

# Optional: provide additional functionality for FASTA/GEO parsing.  These
# imports are wrapped in a try/except block so that the script can run
# even if Biopython is not installed.  If you intend to process
# sequence or GEO datasets, install ``biopython``.
try:
    from Bio import SeqIO  # type: ignore
except ImportError:
    SeqIO = None  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare dataset for modelling")
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

    # Configure basic logging.  This allows debug information and
    # warnings to be emitted to the console.  Logging level can be
    # adjusted via the YAML configuration (not currently exposed).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create processed directory if necessary
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which raw data file to process.  By default we look
    # for the UCI yeast dataset (yeast.data).  If other files are
    # present in the raw directory (e.g., FASTA sequences, GEO
    # expression matrices), those can be specified via the
    # configuration under ``paths.raw_file``.  Otherwise the script
    # attempts to infer the file based on its extension.
    raw_file = config.get("paths", {}).get("raw_file")
    if raw_file:
        data_file = raw_dir / raw_file
    else:
        # If raw_file isn't specified, search for known file types.  We
        # prioritise yeast.data for backward compatibility.
        if (raw_dir / "yeast.data").exists():
            data_file = raw_dir / "yeast.data"
        else:
            # Take the first file in the raw directory
            files = list(raw_dir.glob("*"))
            if not files:
                raise FileNotFoundError(f"No raw data files found in {raw_dir}")
            data_file = files[0]
    if not data_file.exists():
        raise FileNotFoundError(f"Expected raw data file at {data_file} but it was not found.")

    suffix = data_file.suffix.lower()
    logging.info(f"Preparing dataset from {data_file} (type: {suffix})")
    target_col = config["task"]["target"]
    df: pd.DataFrame

    try:
        if suffix in {".data", "", ".txt"}:
            # Assume UCI yeast format: whitespace‑separated with 8 features and a class label
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
            df = pd.read_csv(data_file, sep="\s+", header=None, names=columns)
            # Encode categorical target
            df[target_col] = df["localization_site"].astype("category").cat.codes
            df = df.drop(columns=["localization_site"])
        elif suffix in {".csv", ".tsv", ".tab"}:
            # Generic tabular dataset.  Expect a column matching
            # ``target_col`` for the labels.
            sep = "," if suffix == ".csv" else "\t"
            df = pd.read_csv(data_file, sep=sep)
            if target_col not in df.columns:
                raise KeyError(
                    f"Target column '{target_col}' not found in {data_file.name}. "
                    "Please ensure your CSV/TSV has this column or update the config."
                )
        elif suffix in {".fasta", ".fa", ".fna", ".faa"}:
            # Sequence file: compute simple features from FASTA sequences.
            if SeqIO is None:
                raise ImportError("Biopython is required to process FASTA files. Please install it.")
            records = list(SeqIO.parse(str(data_file), "fasta"))
            if not records:
                raise ValueError(f"No sequences found in FASTA file {data_file}")
            features: List[dict] = []
            for rec in records:
                seq = rec.seq.upper()
                length = len(seq)
                gc_count = seq.count("G") + seq.count("C")
                gc_content = gc_count / length if length > 0 else 0
                features.append({
                    "id": rec.id,
                    "seq_length": length,
                    "gc_content": gc_content,
                })
            df = pd.DataFrame(features)
            # For FASTA data there may be no label; create a dummy target of zeros
            df[target_col] = 0
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Please provide a .data, .csv, .tsv or .fasta file.")
    except Exception as exc:
        logging.error(f"Failed to process {data_file}: {exc}")
        raise

    # Save processed data
    df.to_csv(processed_path, index=False)
    logging.info(f"Processed data written to {processed_path}")


if __name__ == "__main__":
    main()