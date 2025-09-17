#!/usr/bin/env python3
"""
Merge multiple processed CSV datasets into a single dataset.

This utility is useful when you have collected several processed data tables
from different experiments or sources and wish to train a single model on
the combined set.  It assumes that each input CSV shares the same target
column but may differ in feature columns.  When merging, the union of all
feature columns is taken; missing values are filled with NaN.

Example usage:

    python src/merge_datasets.py --input-dir data/processed_multiple --output-file data/processed/merged.csv --target survival_label

Arguments:
  --input-dir: Directory containing one or more processed CSV files to merge.
  --output-file: Destination path for the combined CSV (default:
                 data/processed/merged.csv).
  --target: Name of the target column.  Rows missing this column will be
            dropped with a warning.

The script prints a summary of the merged dataset, including the total
number of rows and columns, and writes the combined CSV to the specified
output file.
"""
import argparse
import glob
import os
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple processed CSV datasets into a single dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing processed CSV files to merge",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/merged.csv",
        help="Path of the output merged CSV file",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target column present in all input files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    target_col = args.target
    # Find all CSV files in the input directory
    pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No CSV files found in directory {input_dir}")
    datasets = []
    for fp in files:
        df = pd.read_csv(fp)
        if target_col not in df.columns:
            print(f"Warning: target column '{target_col}' not found in {fp}; skipping this file.")
            continue
        datasets.append(df)
    if not datasets:
        raise ValueError("No valid datasets with the target column were found.")
    # Combine all datasets by taking the union of columns and aligning on column names
    merged_df = pd.concat(datasets, axis=0, join="outer", ignore_index=True, sort=False)
    # Drop rows missing the target
    missing_target = merged_df[target_col].isna().sum()
    if missing_target > 0:
        print(f"Dropping {missing_target} rows lacking target column values.")
        merged_df = merged_df.dropna(subset=[target_col])
    # Write merged dataset
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    print(f"Merged dataset shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")


if __name__ == "__main__":
    main()