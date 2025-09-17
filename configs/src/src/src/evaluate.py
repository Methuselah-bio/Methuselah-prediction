import argparse, yaml, json, joblib, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss
def main(cfg):
    df = pandas.read_csv(cfg["paths"]["processed"])
    y = df[cfg["task"]["target"]].values
    X = df.drop(columns=[cfg["task"]["target"]]).values
    model = joblib.load(cfg["paths"]["results"] + "/model.pkl")
    proba = model.predict_proba(X)[:,1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, proba))
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
