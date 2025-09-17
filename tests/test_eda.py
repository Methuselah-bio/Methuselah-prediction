"""Tests for the eda script.

This test creates a small synthetic dataset, writes a temporary
configuration and invokes the EDA script.  It then verifies that
expected output files are generated in the results directory.
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import yaml
import pandas as pd


def test_eda_creates_outputs(tmp_path: Path) -> None:
    """The EDA script should write statistics and at least one plot."""
    # Create a simple numeric dataset with a binary target
    df = pd.DataFrame({
        'feat1': [1.0, 2.0, 3.0, 4.0],
        'feat2': [0.5, 0.6, 0.7, 0.8],
        'survival_label': [0, 1, 0, 1],
    })
    processed = tmp_path / 'processed.csv'
    df.to_csv(processed, index=False)
    # Write a minimal config pointing to the temporary files
    config = {
        'paths': {
            'processed': str(processed),
            'results': str(tmp_path),
        },
        'task': {
            'target': 'survival_label',
        },
    }
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    # Invoke the eda script via subprocess.  Use sys.executable to ensure
    # the same interpreter is used as pytest.
    script_path = Path(__file__).resolve().parents[1] / 'src' / 'eda.py'
    subprocess.run([sys.executable, str(script_path), '--config', str(config_path)], check=True)
    # Check that the stats JSON was created
    assert (tmp_path / 'eda_stats.json').exists()
    # Heatmap and embedding may or may not be created depending on features; at
    # least one of them should exist when there are ≥2 features.
    correlation_exists = (tmp_path / 'eda_correlation.png').exists()
    embedding_exists = (tmp_path / 'eda_embedding.png').exists()
    assert correlation_exists or embedding_exists