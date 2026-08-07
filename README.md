# ENEO Financial Distress Project — Python Code

Full pipeline behind `Consumption_Forecasting_Report.md` (Part 1: consumption
forecasting + risk features) and `Financial_Distress_Analysis_Report.md`
(Part 2: financial ratios, Altman Z″ classification, XGBoost benchmark).

## Setup

```bash
pip install -r requirements.txt
```

`torch` and `xgboost` are the heavy dependencies (CPU builds are fine — no
GPU needed for models this small).

## Files — Part 1: consumption forecasting

| Script | Purpose | Input | Output |
|---|---|---|---|
| `01_extract_consumption_series.py` | Extracts the national monthly billed-energy series from the ENEO workbook's `DATA` sheet | `Distribution_Efficiency_OCTOBRE_2025.xlsx` | `consumption_clean.csv` |
| `02_feature_extraction.py` | Builds consumption-forecast-derived risk features (growth rates, volatility, prorating gap) | `consumption_clean.csv` | `consumption_risk_features.csv` |
| `03_sarima_model.py` | Fits/compares 5 SARIMA specs, picks best by holdout RMSE | `consumption_clean.csv` | `sarima_results.json` |
| `04_lstm_model.py` | Trains a 1-layer LSTM (PyTorch) on the same train/test split | `consumption_clean.csv` | `lstm_results.json` |
| `05_plot_comparison.py` | Plots actual vs. SARIMA vs. LSTM over the holdout | `consumption_clean.csv`, `sarima_results.json`, `lstm_results.json` | `forecast_comparison.png` |

## Files — Part 2: financial distress

| Script | Purpose | Input | Output |
|---|---|---|---|
| `06_extract_eneo_financials.py` | Hand-transcribed income statement (2019-2023) & balance sheet (2021-2022) figures, with page-level source citations | *(hardcoded — see script docstring for provenance)* | `eneo_income_statement.csv`, `eneo_balance_sheet.csv` |
| `07_financial_ratios_altman.py` | Computes financial ratios + Altman Z″ score, classifies each year Distress/Grey/Safe zone | `eneo_income_statement.csv`, `eneo_balance_sheet.csv` | `eneo_ratios_distress.csv` |
| `08_train_uci_xgboost.py` | Trains XGBoost Model A (core ratios) vs Model B (core + growth features) on the UCI Taiwanese Bankruptcy dataset | `uci_bankruptcy_clean.csv` | `model_a_results.json`, `model_b_results.json`, `uci_test_predictions.csv` |
| `09_delong_test.py` | DeLong's test for statistical significance of the AUC difference between Model A and B | `uci_test_predictions.csv` | prints z-statistic, p-value |

## Run order

```bash
# Part 1
python 01_extract_consumption_series.py
python 02_feature_extraction.py
python 03_sarima_model.py
python 04_lstm_model.py
python 05_plot_comparison.py

# Part 2
python 06_extract_eneo_financials.py
python 07_financial_ratios_altman.py
python 08_train_uci_xgboost.py
python 09_delong_test.py
```

Place `Distribution_Efficiency_OCTOBRE_2025.xlsx` in the same folder before
running step 1, and `uci_bankruptcy_clean.csv` (the UCI Taiwanese Bankruptcy
dataset — 6,819 rows, 95 features) before running step 8. Everything
downstream only needs the CSVs/JSON each step produces.

## Key honest findings (not smoothed over)

- SARIMA beat the LSTM on the consumption series by ~4.8x RMSE — expected
  with only 62 training months, not a modeling bug.
- ENEO's annual reports only have a **full balance sheet for FY2021 and
  FY2022** — everywhere else, ratios/Altman Z″ are marked "not computable"
  rather than estimated.
- Both computable years (FY2021, FY2022) land in the Altman **distress
  zone** (Z″ ≈ 0.03–0.07 vs. a 1.1 threshold).
- ENEO alone has too few firm-years to train its own classifier, so
  Model A/B, the ablation, and the DeLong test are run on the UCI benchmark
  as a **methodology validation**, not a direct ENEO prediction.
- Adding growth/trend features improved AUC (0.928 → 0.936) but DeLong's
  test says this is **not statistically significant** (p = 0.163).

## Notes

- The holdout is the last 8 months of the 70-month series (Mar–Oct 2025).
- Both models are evaluated on the **same** split so their RMSE/MAE are
  directly comparable.
- SARIMA outperforming the LSTM (RMSE ≈2.28M vs ≈11.05M) is expected given
  only 62 training months — worth stating explicitly in the writeup rather
  than treated as a bug.
