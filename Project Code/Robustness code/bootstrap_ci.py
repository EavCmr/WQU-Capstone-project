import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = "../data"
RATIO_FEATURES = ["leverage","equity_ratio","debt_to_equity","net_margin"]
OP = RATIO_FEATURES + ["delivery_gwh","delivery_yoy_change","delivery_volatility_3yr"]
train = pd.read_csv(DATA_DIR + "/train_us_panel.csv")
op_rows = train.dropna(subset=["delivery_gwh"]).reset_index(drop=True)

def loco_cv(df, feats):
    X = df[feats].fillna(df[feats].median()).values
    y = df["distress_label"].values
    groups = df["company"].values
    preds = np.zeros(len(df))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])[:, 1]
    return preds, y

def boot_ci(y, preds, label, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    fpr, tpr, thr = roc_curve(y, preds)
    cutoff = thr[np.argmax(tpr - fpr)]
    n = len(y)
    aucs, f1s, precs, recs = [], [], [], []
    idx_all = np.arange(n)
    tries = 0
    while len(aucs) < n_boot and tries < n_boot * 5:
        tries += 1
        idx = rng.choice(idx_all, size=n, replace=True)
        yb, pb = y[idx], preds[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
        yhat = (pb >= cutoff).astype(int)
        f1s.append(f1_score(yb, yhat, zero_division=0))
        precs.append(precision_score(yb, yhat, zero_division=0))
        recs.append(recall_score(yb, yhat, zero_division=0))
    def ci(a):
        a = np.array(a)
        return np.percentile(a, 2.5), np.percentile(a, 97.5)
    print(f"{label}: AUC 95%CI=({ci(aucs)[0]:.3f},{ci(aucs)[1]:.3f})  "
          f"F1(Youden@{cutoff:.3f}) 95%CI=({ci(f1s)[0]:.3f},{ci(f1s)[1]:.3f})  "
          f"Prec 95%CI=({ci(precs)[0]:.3f},{ci(precs)[1]:.3f})  Rec 95%CI=({ci(recs)[0]:.3f},{ci(recs)[1]:.3f})")

predsA_full, yA_full = loco_cv(train, RATIO_FEATURES)
boot_ci(yA_full, predsA_full, "Model A, full panel (n=87)")

predsA_sub, yA_sub = loco_cv(op_rows, RATIO_FEATURES)
boot_ci(yA_sub, predsA_sub, "Model A, subset (n=48)")

predsB, yB = loco_cv(op_rows, OP)
boot_ci(yB, predsB, "Model B, subset (n=48)")
print("Done.")
