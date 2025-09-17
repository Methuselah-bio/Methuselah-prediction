import argparse, yaml, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss
import matplotlib.pyplot as plt

def get_model(name, params):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "rf":
        return RandomForestClassifier(**{k:v for k,v in params.items() if k in {"n_estimators","max_depth"}})
    if name == "xgboost":
        return XGBClassifier(eval_metric="logloss", use_label_encoder=False, **params)
    raise ValueError(f"Unknown model {name}")

def train_eval(cfg):
    rng = np.random.RandomState(cfg["seed"])
    df = pd.read_csv(cfg["paths"]["processed"])
    y = df[cfg["task"]["target"]].values
    X = df.drop(columns=[cfg["task"]["target"]]).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=rng)
    model = get_model(cfg["model"]["name"], cfg["model"]["params"])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y_te, proba)),
        "auprc": float(average_precision_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "brier": float(brier_score_loss(y_te, proba))
    }

    outdir = Path(cfg["paths"]["results"])
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    joblib.dump(model, outdir/"model.pkl")

    # simple ROC plot
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_te, proba)
    plt.tight_layout(); plt.savefig(outdir/"auroc.png", dpi=150); plt.close()

    # simple calibration plot
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("Mean predicted"); plt.ylabel("Fraction positive"); plt.tight_layout()
    plt.savefig(outdir/"calibration.png", dpi=150); plt.close()
    print("Saved metrics and plots to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    train_eval(cfg)
