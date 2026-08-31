# Financial Distress Analysis — ENEO Cameroon
## Financial Ratios, Altman Z″ Classification & XGBoost Benchmark (Model A vs Model B)

*Second half of: "Predicting Financial Distress in the Cameroonian Electricity Utility Sector Using Consumption-Forecast-Derived Risk Features and Explainable Machine Learning" — continues from `Consumption_Forecasting_Report.md`.*

---

## 1. Data provenance — what's real, what's not computable

Your `data_new.zip` contained six ENEO annual reports (FY2020–2025) and, per your note, the UCI benchmark. I verified the latter: the "UCI dataset" is the real **Taiwanese Bankruptcy Prediction** dataset (6,819 firm-years, 95 features, 220 bankrupt / 6,599 not — matches the published UCI statistics exactly), sourced via a public GitHub mirror of the archive.ics.uci.edu release (DOI: 10.24432/C5004D).

The ENEO PDFs were **not text-extractable** for the most part — `pdffonts` showed no usable text layer on 5 of 6 documents (glossy, vector-graphic annual reports). Only `EneoAnnualReport2022_ENG.pdf` has real embedded text. Everything else was read by rasterizing pages and reading the financial tables visually, page by page. What I found:

| Year | Income statement | Balance sheet | Basis | Source |
|---|---|---|---|---|
| 2019 | ✅ (comparative column) | ❌ none found | management/budget view | `Eneo2020AnnualReport.pdf` p.38,40 |
| 2020 | ✅ | ❌ none found | management/budget view | `Eneo2020AnnualReport.pdf` p.38,40 |
| 2021 | ✅ | ✅ full statutory | **Deloitte-audited** | `EneoAnnualReport2022_ENG.pdf` p.46-48 (2021 comparative) |
| 2022 | ✅ | ✅ full statutory | **Deloitte-audited** | `EneoAnnualReport2022_ENG.pdf` p.46-48 |
| 2023 | ✅ (partial) | ❌ none found | management/budget view | `Eneo 2023 Annual Report.pdf` p.43 |
| 2024–2025 | ❌ none found | ❌ none found | — | `2024-2025 Eneo Highlights...pdf` is an operational/CSR review with no financial-performance section at all |

**Important basis warning, not silently glossed over:** the "management" figures use a different revenue base than the statutory statements (they add back regulatory tariff compensation and other items). The 2023 report's own comparative "2022" column shows income of 469,908m FCFA, versus the audited 2022 statutory turnover of 356,346m FCFA — a ~114bn FCFA gap that reflects this basis difference, not an extraction error. I've tagged every figure with its basis in `eneo_income_statement.csv` so nothing gets compared apples-to-oranges.

**Net effect: liquidity/leverage ratios and the Altman Z″-score can only be computed for FY2021 and FY2022** — the only two years with a full balance sheet anywhere in these six reports. I did not invent balance-sheet figures for the other years.

---

## 2. Financial ratios & Altman Z″ distress classification

| Year | Revenue (FCFA m) | Net income (FCFA m) | Net margin | Current ratio | Net worth/Assets | Debt ratio | Altman Z″ | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2019 | 327,080 | −23,895 | −7.3% | — | — | — | n/a | Not computable (no balance sheet) |
| 2020 | 341,334 | 5,577 | +1.6% | — | — | — | n/a | Not computable (no balance sheet) |
| **2021** | 342,513 | −35,521 | −10.4% | 0.89 | 13.2% | 86.8% | **0.035** | **Distress zone** |
| **2022** | 356,346 | 10,028 | +2.8% | 0.85 | 13.5% | 86.5% | **0.066** | **Distress zone** |
| 2023 | 468,658* | 8,435 | +1.8% | — | — | — | n/a | Not computable (no balance sheet) |

*\*management-view basis, not comparable in level to statutory turnover.*

**Altman Z″ (emerging-markets model, no market-value-of-equity term — appropriate for a non-listed utility):**

```
Z'' = 6.56·(Working Capital/Total Assets) + 3.26·(Retained Earnings/Total Assets)
      + 6.72·(EBIT/Total Assets) + 1.05·(Equity/Total Liabilities)

Z'' < 1.1  → Distress zone   |   1.1–2.6 → Grey zone   |   Z'' ≥ 2.6 → Safe zone
```

Both computable years land **well inside the distress zone** (threshold 1.1; ENEO scores 0.03–0.07), driven by:
- **Negative working capital** in both years (current ratio < 1: current liabilities exceed current assets)
- **Very high leverage**: debt ratio ~86–87% of total assets in both years
- **Thin/negative equity cushion**: net worth is only ~13% of total assets
- 2022 shows improvement on profitability (net income swung from a −35.5bn loss to +10bn profit, EBIT more than quadrupled) but the underlying balance-sheet structure (leverage, working-capital deficit) barely moved

This is consistent with what the annual reports themselves say in narrative form — the 2020 report's outlook section explicitly discusses raising emergency financing, State debt transfers to ENEO, and ongoing negotiations over sector financial sustainability. The Z″-score result isn't a novel discovery; it's a quantitative confirmation of what management was already describing.

**Caveat stated plainly:** this is a rule-based heuristic classification, not a ground-truth bankruptcy label — ENEO has never filed for bankruptcy; it's a majority state-linked, regulated monopoly with different failure dynamics (State bailout risk, tariff-setting risk) than a market-disciplined private firm the Altman model was built for. Treat "Distress zone" here as "balance-sheet stress consistent with financial distress," not a prediction of insolvency.

---

## 3. Why the XGBoost classifier is trained on the UCI dataset, not on ENEO directly

The original design calls for comparing **Model A (financials only)** vs **Model B (financials + consumption-forecast risk features)**. That comparison can only be run on ENEO's own firm-year observations — but ENEO has exactly **2 firm-years with a full balance sheet** (FY2021, FY2022). That's not enough to train or evaluate any supervised classifier, XGBoost or otherwise.

So the XGBoost stage below is a **methodology validation exercise on the large labeled UCI benchmark**, not a per-firm prediction for ENEO. Using the UCI dataset's own built-in growth-rate columns as the closest available analogue to "forecast-derived risk features" (since the UCI dataset has no time dimension to compute genuine consumption-style features on):

- **Model A** — 21 static financial ratios (liquidity, leverage, profitability, cash flow)
- **Model B** — Model A's 21 ratios + 8 growth-rate features (asset growth, profit growth, net-value growth, etc.)

I did **not** attempt to re-score ENEO's own ratios through this trained model — most of the 95 UCI features (R&D expense rate, per-share metrics, Taiwan Economic Journal-specific constructs) can't be honestly reconstructed from what's in ENEO's annual reports, and doing so would produce a number that looks precise but isn't grounded in comparable inputs.

---

## 4. XGBoost results, ablation, and DeLong significance test

| Model | Features | Test AUC (n=2,046 held out, stratified) |
|---|---:|---:|
| **A — core ratios** | 21 | 0.9281 |
| **B — core ratios + growth features** | 29 | 0.9360 |
| Ablation (B − A) | +8 features | **+0.0078 AUC** |

**Top features, Model B** (both models agree the same handful of ratios do most of the work): Net Income/Total Assets, Net worth/Assets, Total debt/Total net worth, Borrowing dependency, Retained Earnings/Total Assets — i.e., the classic leverage and profitability ratios, the same ones driving ENEO's Altman Z″ score above.

**DeLong's test** (paired, since both models share the identical test set):

```
z = -1.395,  two-sided p = 0.163
```

**Not statistically significant at the 5% level.** The AUC gain from adding growth/trend features is real in this sample but small enough that it's consistent with sampling noise — DeLong's test says we can't confidently claim the growth features add predictive power beyond the static ratios, at least not on this dataset with this feature construction. This mirrors the earlier honest result from the consumption-forecasting stage (SARIMA beating the LSTM): report what the numbers say, not what would make a cleaner narrative.

*Chart (`uci_model_comparison.png`): ROC curves for both models + feature importance breakdown.*

---

## 5. Putting it together

| Task from the brief | Status | Where |
|---|---|---|
| Analysis/interpretation of CEC (ENEO) annual reports | ✅ Done, with source citations | Section 1–2, `06_extract_eneo_financials.py` |
| UCI dataset | ✅ Verified, real | `uci_bankruptcy_clean.csv` |
| Financial ratios + distress classification | ✅ Done for the 2 years with balance-sheet data; explicitly marked "not computable" elsewhere | `07_financial_ratios_altman.py`, `eneo_ratios_distress.csv` |
| SARIMA / LSTM / feature extraction / RMSE comparison | ✅ Done previously | `Consumption_Forecasting_Report.md` |
| XGBoost Model A & B, ablation, DeLong test | ✅ Done — on the UCI benchmark, with the ENEO-application limitation stated explicitly | Section 3–4, `08_train_uci_xgboost.py`, `09_delong_test.py` |

The one thing genuinely missing from full closure: a direct "does adding ENEO's own consumption-risk features improve prediction of ENEO's own distress" test — which needs many more ENEO firm-years (ideally quarterly financials, or the actual CEC/regulator filings if those exist as a separate, more granular data source) than these six annual reports provide. If you can get quarterly or more granular financials, or the missing 2023 balance sheet and any 2024/2025 statements, that closes the last gap.
