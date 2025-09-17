import argparse, yaml, pandas as pd, numpy as np
from pathlib import Path

def prepare(cfg):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_csv = Path(cfg["paths"]["processed"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # TODO: replace this with real data loading / fetching.
    # For now, make a tiny toy frame you can run end-to-end.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    y = (X[:, 0] + 0.5*X[:, 1] - 0.2*X[:, 2] > 0).astype(int)
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["survival_label"] = y

    # simple z-score
    df[cols] = (df[cols] - df[cols].mean())/df[cols].std(ddof=0)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with shape {df.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    prepare(cfg)
