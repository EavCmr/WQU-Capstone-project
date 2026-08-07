"""
07_financial_ratios_altman.py

Computes financial ratios and an Altman Z''-score (emerging-markets model
for private/non-manufacturing firms, no market-value-of-equity term) for
ENEO, and classifies each year as Distress / Grey Zone / Safe.

Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
  X1 = Working Capital / Total Assets
  X2 = Retained Earnings / Total Assets
  X3 = EBIT / Total Assets
  X4 = Book Value of Equity / Total Liabilities

Zones (standard, unadjusted for country risk):
  Z'' < 1.1        -> Distress zone
  1.1 <= Z'' < 2.6  -> Grey zone
  Z'' >= 2.6        -> Safe zone

IMPORTANT: this can only be computed for FY2021 and FY2022 — the only two
years with a full statutory balance sheet in the source reports (see
06_extract_eneo_financials.py for data provenance). For FY2019/2020/2023 we
report profitability-only ratios (no balance sheet available) and flag the
Altman score as not computable, rather than filling in an invented number.

Input : eneo_income_statement.csv, eneo_balance_sheet.csv
Output: eneo_ratios_distress.csv
"""

import numpy as np
import pandas as pd

INCOME_CSV = "eneo_income_statement.csv"
BALANCE_CSV = "eneo_balance_sheet.csv"
OUTPUT_CSV = "eneo_ratios_distress.csv"


def classify_zone(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "Not computable (no balance sheet)"
    if z < 1.1:
        return "Distress zone"
    if z < 2.6:
        return "Grey zone"
    return "Safe zone"


def run(income_csv=INCOME_CSV, balance_csv=BALANCE_CSV, output_csv=OUTPUT_CSV):
    inc = pd.read_csv(income_csv)
    bal = pd.read_csv(balance_csv)

    df = inc.merge(bal, on="year", how="left")

    # Profitability ratios (available for all years with income-statement data)
    df["net_margin_pct"] = 100 * df["net_income"] / df["revenue"]
    df["ebitda_margin_pct"] = 100 * df["ebitda"] / df["revenue"]

    # Balance-sheet-dependent ratios (only FY2021/FY2022)
    df["total_liabilities"] = df["total_assets"] - df["total_equity"]
    df["working_capital"] = df["total_current_assets"] - df["total_current_liabilities"]
    df["current_ratio"] = df["total_current_assets"] / df["total_current_liabilities"]
    df["net_worth_to_assets"] = df["total_equity"] / df["total_assets"]
    df["debt_ratio_pct"] = 100 * df["total_liabilities"] / df["total_assets"]

    df["X1_WC_TA"] = df["working_capital"] / df["total_assets"]
    df["X2_RE_TA"] = df["retained_earnings"] / df["total_assets"]
    df["X3_EBIT_TA"] = df["ebit"] / df["total_assets"]
    df["X4_BVE_TL"] = df["total_equity"] / df["total_liabilities"]

    df["altman_z2"] = (
        6.56 * df["X1_WC_TA"] + 3.26 * df["X2_RE_TA"]
        + 6.72 * df["X3_EBIT_TA"] + 1.05 * df["X4_BVE_TL"]
    )
    df["distress_zone"] = df["altman_z2"].apply(classify_zone)

    cols = ["year", "revenue", "net_income", "net_margin_pct", "ebitda_margin_pct",
            "current_ratio", "net_worth_to_assets", "debt_ratio_pct",
            "altman_z2", "distress_zone", "basis"]
    out = df[cols]
    out.to_csv(output_csv, index=False)

    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
