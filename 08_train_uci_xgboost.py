"""
08_train_uci_xgboost.py

Trains and compares two XGBoost classifiers on the UCI Taiwanese
Bankruptcy Prediction dataset (6,819 firm-years, 95 features, 220
bankrupt / 6,599 not — https://doi.org/10.24432/C5004D):

  Model A ("core financial ratios"): a standard set of ~20 liquidity,
    leverage, and profitability ratios — the type of static, single-period
    financial-ratio feature set most distress-prediction papers start from.

  Model B ("core ratios + growth/trend features"): Model A's features plus
    8 growth-rate columns already present in the UCI dataset (asset growth,
    profit growth, net-value growth, etc.).

Why this substitution, honestly stated: the original project design calls
for comparing "financials only" vs "financials + consumption-forecast risk
features" — but that comparison can only be run on ENEO itself, and ENEO
has only 2 firm-years with a full balance sheet (FY2021, FY2022), nowhere
near enough to train or evaluate a supervised classifier. The UCI dataset
is used here as the large-sample benchmark to validate the *methodology*
(does adding growth/trend information to a static ratio set improve
distress prediction, and is the improvement statistically significant via
DeLong's test) — using its own built-in growth-rate features as the
closest available analogue to "forecast-derived risk features". This is a
methodology validation exercise, not a prediction for ENEO specifically.

Input : uci_bankruptcy_clean.csv
Output: model_a_results.json, model_b_results.json, uci_test_predictions.csv
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

INPUT_CSV = "uci_bankruptcy_clean.csv"
TARGET = "Bankrupt"
SEED = 42

# Model A: core, static financial ratios (liquidity / leverage / profitability / cash flow)
CORE_RATIO_FEATURES = [
    "Current Ratio", "Quick Ratio", "Working Capital to Total Assets",
    "Current Liability to Assets", "Debt ratio %", "Total debt/Total net worth",
    "Net worth/Assets", "Borrowing dependency", "Long-term fund suitability ratio (A)",
    "ROA(A) before interest and % after tax", "ROA(B) before interest and depreciation after tax",
    "ROA(C) before interest and depreciation before interest",
    "Net Income to Total Assets", "Net Income to Stockholder's Equity",
    "Operating Profit Rate", "Gross Profit to Sales",
    "Cash Flow to Total Assets", "Cash Flow to Liability", "Cash flow rate",
    "Interest Coverage Ratio (Interest expense to EBIT)", "Retained Earnings to Total Assets",
]

# Growth/trend features already present in the UCI dataset — used as the
# available analogue to "consumption-forecast-derived risk features"
GROWTH_FEATURES = [
    "Realized Sales Gross Profit Growth Rate", "Operating Profit Growth Rate",
    "After-tax Net Profit Growth Rate", "Regular Net Profit Growth Rate",
    "Continuous Net Profit Growth Rate", "Total Asset Growth Rate",
    "Net Value Growth Rate", "Total Asset Return Growth Rate Ratio",
]


def train_eval(X_train, X_test, y_train, y_test, seed=SEED):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,  # class imbalance: ~3% positive
        eval_metric="auc", random_state=seed, n_jobs=4,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    importances = dict(zip(X_train.columns, model.feature_importances_.tolist()))
    return model, proba, auc, importances


def run(input_csv=INPUT_CSV):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    y = df[TARGET]

    X_A = df[CORE_RATIO_FEATURES]
    X_B = df[CORE_RATIO_FEATURES + GROWTH_FEATURES]

    # Same split (indices) for A and B so predictions are paired -> valid for DeLong test
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, random_state=SEED, stratify=y
    )

    XA_train, XA_test = X_A.loc[idx_train], X_A.loc[idx_test]
    XB_train, XB_test = X_B.loc[idx_train], X_B.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    model_a, proba_a, auc_a, imp_a = train_eval(XA_train, XA_test, y_train, y_test)
    model_b, proba_b, auc_b, imp_b = train_eval(XB_train, XB_test, y_train, y_test)

    print(f"Model A (core ratios only, {len(CORE_RATIO_FEATURES)} features): AUC = {auc_a:.4f}")
    print(f"Model B (core ratios + growth features, {len(CORE_RATIO_FEATURES) + len(GROWTH_FEATURES)} features): AUC = {auc_b:.4f}")
    print(f"Ablation: adding growth/trend features changed AUC by {auc_b - auc_a:+.4f}")

    top_a = sorted(imp_a.items(), key=lambda x: -x[1])[:5]
    top_b = sorted(imp_b.items(), key=lambda x: -x[1])[:5]
    print("\nModel A top-5 features:", top_a)
    print("Model B top-5 features:", top_b)

    with open("model_a_results.json", "w") as f:
        json.dump({"auc": auc_a, "features": CORE_RATIO_FEATURES,
                   "feature_importance": imp_a}, f, indent=2)
    with open("model_b_results.json", "w") as f:
        json.dump({"auc": auc_b, "features": CORE_RATIO_FEATURES + GROWTH_FEATURES,
                   "feature_importance": imp_b}, f, indent=2)

    pd.DataFrame({
        "y_true": y_test.values,
        "proba_model_a": proba_a,
        "proba_model_b": proba_b,
    }).to_csv("uci_test_predictions.csv", index=False)

    return auc_a, auc_b


if __name__ == "__main__":
    run()
