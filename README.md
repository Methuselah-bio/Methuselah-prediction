# Methuselah-prediction
Predicting *S. cerevisiae* survival under nutrient perturbations (TOR/SNF1 themes) with simple, reproducible ML baselines.

## Why
A small, admissions-ready demo that bridges my genetics/aging background with applied ML.

## Quick start (placeholder)
> Code coming soon. This repo will include:
> - `src/prepare_data.py` — fetch + preprocess public yeast data
> - `src/train.py` — train baseline models (logreg / RF / XGBoost)
> - `src/evaluate.py` — metrics (AUROC, AUPRC, accuracy, Brier)
> - `src/interpret.py` — permutation feature importance + notes

### Planned commands
```bash
# environment (to be added)
# conda env create -f environment.yml && conda activate methuselah
# or: pip install -r requirements.txt

# preprocess -> data/processed/processed.csv
# python src/prepare_data.py --config configs/base.yaml

# train -> results/metrics.json, auroc.png, calibration.png
# python src/train.py --config configs/base.yaml
# python src/evaluate.py --config configs/base.yaml
