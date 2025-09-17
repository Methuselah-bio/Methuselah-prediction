"""Tests for the interpret_shap script.

The SHAP interpretability script should exit gracefully when the
`shap` package is not installed.  This test simply runs the script
with a small dataset and ensures it does not raise an exception.
"""

import subprocess
import sys
from pathlib import Path
import yaml
import pandas as pd


def test_interpret_shap_runs_without_shap(tmp_path: Path) -> None:
    """interpret_shap.py should exit cleanly even if shap is unavailable."""
    # Create a simple dataset
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [3, 2, 1],
        'survival_label': [0, 1, 0],
    })
    processed = tmp_path / 'processed.csv'
    df.to_csv(processed, index=False)
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
    script_path = Path(__file__).resolve().parents[1] / 'src' / 'interpret_shap.py'
    # Run the script; it should exit with code 0 even if shap is missing
    result = subprocess.run([sys.executable, str(script_path), '--config', str(config_path), '--max-samples', '2'])
    assert result.returncode == 0