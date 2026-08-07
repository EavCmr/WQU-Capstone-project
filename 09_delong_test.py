"""
09_delong_test.py

Implements DeLong's test (DeLong, DeLong & Clarke-Pearson, 1988) for
comparing two correlated ROC AUCs — the standard approach when both models
are evaluated on the *same* test set (paired predictions), as is the case
here (Model A and Model B share the identical train/test split).

Reference implementation follows the fast O(n log n) method described in
Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for Comparing
the Areas Under Correlated Receiver Operating Characteristic Curves".

Input : uci_test_predictions.csv (y_true, proba_model_a, proba_model_b)
Output: prints AUCs, AUC difference, z-statistic, and two-sided p-value
"""

import numpy as np
import pandas as pd


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: shape (n_models, n_samples), positives first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, proba_a, proba_b):
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positives (1) first
    y_sorted = y_true[order]
    m = int(y_sorted.sum())  # number of positives

    preds = np.vstack([np.asarray(proba_a)[order], np.asarray(proba_b)[order]])
    aucs, cov = _fast_delong(preds, m)

    auc_a, auc_b = aucs[0], aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var_diff = max(var_diff, 1e-12)  # guard against tiny negative values from float error
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p_value


def run(csv_path="uci_test_predictions.csv"):
    df = pd.read_csv(csv_path)
    auc_a, auc_b, z, p = delong_roc_test(df["y_true"], df["proba_model_a"], df["proba_model_b"])

    print(f"Model A AUC (DeLong): {auc_a:.4f}")
    print(f"Model B AUC (DeLong): {auc_b:.4f}")
    print(f"AUC difference (B - A): {auc_b - auc_a:+.4f}")
    print(f"z-statistic: {z:.4f}")
    print(f"Two-sided p-value: {p:.4g}")
    if p < 0.05:
        print("-> Statistically significant difference at the 5% level: "
              "adding growth/trend features produces a genuine AUC improvement, "
              "not just noise.")
    else:
        print("-> NOT statistically significant at the 5% level: the AUC gain "
              "from adding growth/trend features could plausibly be due to chance.")

    return auc_a, auc_b, z, p


if __name__ == "__main__":
    run()
