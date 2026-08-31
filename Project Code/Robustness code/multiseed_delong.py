import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR = "../data"
RATIO_FEATURES = ["leverage","equity_ratio","debt_to_equity","net_margin"]
OP = RATIO_FEATURES + ["delivery_gwh","delivery_yoy_change","delivery_volatility_3yr"]
train = pd.read_csv(DATA_DIR + "/train_us_panel.csv")
op_rows = train.dropna(subset=["delivery_gwh"]).reset_index(drop=True)

def loco_cv(df, feats, kind, seed=42):
    X = df[feats].fillna(df[feats].median()).values
    y = df["distress_label"].values
    groups = df["company"].values
    preds = np.zeros(len(df))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if kind == "lr":
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
            m.fit(X[tr], y[tr])
        elif kind == "rf":
            m = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=2,
                                        class_weight="balanced", random_state=seed)
            m.fit(X[tr], y[tr])
        elif kind == "gb":
            m = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed)
            sw = compute_sample_weight("balanced", y[tr])
            m.fit(X[tr], y[tr], sample_weight=sw)
        preds[te] = m.predict_proba(X[te])[:, 1]
    return preds, y

print("MULTI-SEED ROBUSTNESS (10 seeds, 48-row operational subset)")
t0 = time.time()
for kind in ["rf", "gb"]:
    aucs, f1s = [], []
    for s in range(10):
        preds, y = loco_cv(op_rows, OP, kind, seed=s)
        aucs.append(roc_auc_score(y, preds))
        fpr, tpr, thr = roc_curve(y, preds)
        cutoff = thr[np.argmax(tpr - fpr)]
        f1s.append(f1_score(y, (preds >= cutoff).astype(int), zero_division=0))
    aucs, f1s = np.array(aucs), np.array(f1s)
    print(f"{kind.upper():4s} AUC mean={aucs.mean():.3f} sd={aucs.std():.3f} range=[{aucs.min():.3f},{aucs.max():.3f}] "
          f"| F1 mean={f1s.mean():.3f} sd={f1s.std():.3f}  seeds={list(np.round(aucs,3))}")
print(f"[{time.time()-t0:.1f}s]")

print("\nDELONG'S TEST")
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def fastDeLong(preds_list, y_true):
    order = (-y_true).argsort()
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted)); n = len(y_sorted) - m
    k = len(preds_list)
    ps = np.vstack([p[order] for p in preds_list])
    pos, neg = ps[:, :m], ps[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = compute_midrank(pos[r]); ty[r] = compute_midrank(neg[r]); tz[r] = compute_midrank(ps[r])
    aucs = tz[:, :m].sum(axis=1)/(m*n) - (m+1.0)/(2.0*n)
    v01 = (tz[:, :m] - tx)/n
    v10 = 1.0 - (tz[:, m:] - ty)/m
    sx, sy = np.cov(v01), np.cov(v10)
    cov = sx/m + sy/n
    return aucs, np.atleast_2d(cov)

def delong_p(preds1, preds2, y):
    aucs, cov = fastDeLong([preds1, preds2], y)
    diff = aucs[0]-aucs[1]
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    if var <= 0: return aucs, diff, np.nan, (np.nan,np.nan)
    from scipy.stats import norm
    z = diff/np.sqrt(var)
    p = 2*(1-norm.cdf(abs(z)))
    ci = (diff-1.96*np.sqrt(var), diff+1.96*np.sqrt(var))
    return aucs, diff, p, ci

predsA_sub, yA_sub = loco_cv(op_rows, RATIO_FEATURES, "lr")
predsB, yB = loco_cv(op_rows, OP, "lr")
aucs, diff, p, ci = delong_p(predsA_sub, predsB, yA_sub)
print(f"Model A vs Model B (48-row subset): AUC_A={aucs[0]:.3f} AUC_B={aucs[1]:.3f} delta={diff:+.3f} "
      f"95%CI=({ci[0]:+.3f},{ci[1]:+.3f}) p={p:.4f}")

predsRF, yRF = loco_cv(op_rows, OP, "rf")
aucs2, diff2, p2, ci2 = delong_p(predsB, predsRF, yB)
print(f"LR vs RF (both w/ operational feats, 48-row subset): AUC_LR={aucs2[0]:.3f} AUC_RF={aucs2[1]:.3f} "
      f"delta={diff2:+.3f} 95%CI=({ci2[0]:+.3f},{ci2[1]:+.3f}) p={p2:.4f}")

predsA_full, yA_full = loco_cv(train, RATIO_FEATURES, "lr")
predsRF_full, yRF_full = loco_cv(train, RATIO_FEATURES, "rf")
aucs3, diff3, p3, ci3 = delong_p(predsA_full, predsRF_full, yA_full)
print(f"LR vs RF (full 87-row panel): AUC_LR={aucs3[0]:.3f} AUC_RF={aucs3[1]:.3f} "
      f"delta={diff3:+.3f} 95%CI=({ci3[0]:+.3f},{ci3[1]:+.3f}) p={p3:.4f}")
print("Done.")
