import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys, time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

LOG = open("/tmp/work/perm_test_output.log", "w", buffering=1)
def log(msg):
    print(msg); LOG.write(msg+"\n"); LOG.flush()

DATA_DIR = "../data"
RATIO_FEATURES = ["leverage","equity_ratio","debt_to_equity","net_margin"]
OP = RATIO_FEATURES + ["delivery_gwh","delivery_yoy_change","delivery_volatility_3yr"]
train = pd.read_csv(DATA_DIR + "/train_us_panel.csv")
op_rows = train.dropna(subset=["delivery_gwh"]).reset_index(drop=True)

def loco_cv(df, feats, kind, n_est=300):
    X = df[feats].fillna(df[feats].median()).values
    y = df["distress_label"].values
    groups = df["company"].values
    preds = np.zeros(len(df))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if kind == "lr":
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
        else:
            m = RandomForestClassifier(n_estimators=n_est, max_depth=3, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])[:, 1]
    return preds, y

def shuffled_auc(df, feats, kind, rng, n_est=300):
    df2 = df.copy()
    df2["distress_label"] = rng.permutation(df2["distress_label"].values)
    try:
        preds, y = loco_cv(df2, feats, kind, n_est=n_est)
        return roc_auc_score(y, preds) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        return np.nan

rng = np.random.RandomState(123)
configs = [
    ("Model A, full panel (LR)", train, RATIO_FEATURES, "lr", 250, 300),
    ("Model A, subset (LR)", op_rows, RATIO_FEATURES, "lr", 250, 300),
    ("Model B, subset (LR)", op_rows, OP, "lr", 250, 300),
    ("Model B/RF, subset (RF)", op_rows, OP, "rf", 60, 60),
]
log("PERMUTATION TEST RESULTS")
for label, df, feats, kind, n_perm, n_est in configs:
    t0 = time.time()
    preds_obs, y_obs = loco_cv(df, feats, kind, n_est=300)
    obs_auc = roc_auc_score(y_obs, preds_obs)
    null_list = []
    tries = 0
    while len(null_list) < n_perm and tries < n_perm * 3:
        v = shuffled_auc(df, feats, kind, rng, n_est=n_est)
        tries += 1
        if not np.isnan(v):
            null_list.append(v)
    null_aucs = np.array(null_list)
    p = (1 + np.sum(null_aucs >= obs_auc)) / (1 + len(null_aucs))
    log(f"{label:28s} n_perm={len(null_aucs):4d}  obs_AUC={obs_auc:.3f}  null_mean={null_aucs.mean():.3f} "
        f"null_sd={null_aucs.std():.3f}  p={p:.4f}  [{time.time()-t0:.1f}s]")
log("Done.")
LOG.close()
