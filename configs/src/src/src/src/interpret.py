import argparse, yaml, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def interpret(cfg):
    df = pd.read_csv(cfg["paths"]["processed"])
    y = df[cfg["task"]["target"]].values
    X = df.drop(columns=[cfg["task"]["target"]])
    feature_names = X.columns.tolist()
    model = joblib.load(cfg["paths"]["results"] + "/model.pkl")
    r = permutation_importance(model, X.values, y, n_repeats=20, random_state=0)
    idx = np.argsort(r.importances_mean)[::-1][:20]
    plt.barh([feature_names[i] for i in idx][::-1], r.importances_mean[idx][::-1])
    plt.tight_layout(); plt.savefig(cfg["paths"]["results"] + "/feature_importance.png", dpi=150)
    print("Saved feature importance.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    interpret(cfg)
