"""
06_extract_eneo_financials.py

Hand-transcribed financial figures from ENEO Cameroon's annual reports.
These annual reports are NOT text-extractable (glossy/vector-graphic PDFs,
no embedded text layer for most pages) except EneoAnnualReport2022_ENG.pdf,
so the underlying tables were read visually (page images) rather than
parsed programmatically. Every figure below is tagged with its source
document, page, and accounting basis so it can be independently checked
against the original PDFs.

Two distinct bases appear across the reports and are NOT interchangeable:
  - "statutory": audited financial statements (Deloitte), OHADA/SYSCOHADA
    format — found only in EneoAnnualReport2022_ENG.pdf (FY2021 & FY2022
    comparatives).
  - "management": internal budget-vs-actual "margin analysis" figures
    used in the glossy annual reports for 2019/2020 and 2022/2023 —
    these use a different revenue base (includes regulatory compensation
    add-backs) and are NOT directly comparable in level to the statutory
    turnover figures. The ~114bn FCFA gap between the statutory 2022
    turnover (356,346m) and the management-view 2022 "Income" figure
    (469,908m) shown in the 2023 report is exactly this basis difference
    — not a data error.

No balance sheet could be found in any report for FY2019, FY2020, FY2023,
FY2024 or FY2025 — only income-statement-level figures. Liquidity/leverage
ratios and the Altman Z''-score can therefore only be computed for FY2021
and FY2022, where a full statutory balance sheet exists.

Units: million FCFA unless noted. All figures divided down from the raw
FCFA amounts printed in the reports.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# INCOME STATEMENT (all years available, mixed basis — see notes)
# ---------------------------------------------------------------------------
income_statement = pd.DataFrame([
    # year, revenue, ebitda, ebit, net_income, interest_expense, da, income_tax, basis, source
    dict(year=2019, revenue=327080, ebitda=29681, ebit=None, net_income=-23895,
         interest_expense=3423, da=27530, income_tax=7224,
         basis="management", source="Eneo2020AnnualReport.pdf p.38,40 (2019 comparative column)"),
    dict(year=2020, revenue=341334, ebitda=29563, ebit=306, net_income=5577,
         interest_expense=7480, da=29257, income_tax=8644,
         basis="management", source="Eneo2020AnnualReport.pdf p.38,40"),
    dict(year=2021, revenue=342513, ebitda=49683, ebit=11443, net_income=-35521,
         interest_expense=None, da=44570, income_tax=9567,
         basis="statutory (Deloitte-audited)", source="EneoAnnualReport2022_ENG.pdf p.48 (2021 comparative column)"),
    dict(year=2022, revenue=356346, ebitda=93276, ebit=53686, net_income=10028,
         interest_expense=None, da=45983, income_tax=15491,
         basis="statutory (Deloitte-audited)", source="EneoAnnualReport2022_ENG.pdf p.48"),
    dict(year=2023, revenue=468658, ebitda=75994, ebit=None, net_income=8435,
         interest_expense=None, da=None, income_tax=None,
         basis="management (not comparable in level to statutory turnover)",
         source="Eneo 2023 Annual Report.pdf p.43 (margin analysis table)"),
])

# ---------------------------------------------------------------------------
# BALANCE SHEET (only FY2021 & FY2022 — full statutory statements found)
# ---------------------------------------------------------------------------
balance_sheet = pd.DataFrame([
    dict(year=2021, total_assets=888406, total_equity=117414,
         total_current_assets=379657, total_current_liabilities=425361,
         total_financial_debt=220762, total_cash_liabilities=124864,
         retained_earnings=34320, fixed_assets=477586,
         source="EneoAnnualReport2022_ENG.pdf p.46-47 (2021 comparative column)"),
    dict(year=2022, total_assets=940121, total_equity=127287,
         total_current_assets=399081, total_current_liabilities=467575,
         total_financial_debt=200289, total_cash_liabilities=144967,
         retained_earnings=-1201, fixed_assets=517894,
         source="EneoAnnualReport2022_ENG.pdf p.46-47"),
])

if __name__ == "__main__":
    income_statement.to_csv("eneo_income_statement.csv", index=False)
    balance_sheet.to_csv("eneo_balance_sheet.csv", index=False)
    print("Income statement:\n", income_statement[["year", "revenue", "ebitda", "net_income", "basis"]])
    print("\nBalance sheet:\n", balance_sheet[["year", "total_assets", "total_equity", "total_current_liabilities"]])
