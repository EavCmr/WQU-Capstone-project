import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats as scipy_stats

DATA_DIR = "../data"
cons = pd.read_csv(DATA_DIR + "/eneo_monthly_consumption.csv", parse_dates=["date"]).sort_values("date").set_index("date")
series = cons["Consumption_Total"].asfreq("MS")
TRAIN_END = "2022-12-01"
cons_train = series.loc[:TRAIN_END]
cons_test = series.loc["2023-01-01":]

sarima = SARIMAX(cons_train, order=(0,1,1), seasonal_order=(0,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc1 = sarima.get_forecast(steps=len(cons_test)).predicted_mean
fc1.index = cons_test.index

# FIXED naive seasonal baseline: forecast(t) = REAL actual value at t-12 months,
# drawn from the full real historical series (not restricted to the training window).
# This is valid/non-leaking because t-12 < t always for t in the test period, so the
# comparator at each step uses only information that would have been available a
# year earlier -- a legitimate walk-forward naive benchmark across the FULL 34-month
# test window, not just the first 12 months.
naive = pd.Series(index=cons_test.index, dtype=float)
for date in cons_test.index:
    lag_date = date - pd.DateOffset(years=1)
    if lag_date in series.index:
        naive.loc[date] = series.loc[lag_date]

sarima2 = SARIMAX(cons_train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc2 = sarima2.get_forecast(steps=len(cons_test)).predicted_mean
fc2.index = cons_test.index

def cfdr(actual, forecast, label):
    df = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    df["abs_pct_error"] = ((df["actual"] - df["forecast"]) / df["actual"]).abs()
    df["t"] = np.arange(1, len(df)+1)
    slope, intercept, r, p, se = scipy_stats.linregress(df["t"], df["abs_pct_error"])
    print(f"{label:45s} n={len(df):3d}  CFDR={slope*100:+.3f}pp/mo  r={r:.2f}  r2={r**2:.2f}  p={p:.4f}  "
          f"mean_err_early={df['abs_pct_error'].iloc[:3].mean()*100:.1f}%  mean_err_late={df['abs_pct_error'].iloc[-3:].mean()*100:.1f}%")
    return slope, p, df

print("SARIMA / FORECAST SPECIFICATION SENSITIVITY CHECK (v2: full-window naive baseline)")
cfdr(cons_test, fc1, "SARIMA(0,1,1)(0,1,1,12) [original spec]")
cfdr(cons_test, fc2, "SARIMA(1,1,1)(1,1,1,12) [alternative spec]")
slope_n, p_n, naive_df = cfdr(cons_test, naive, "Naive seasonal, FULL 34-month window")
print(f"\nNaive baseline coverage: {len(naive_df)} of {len(cons_test)} test months (should be 34 if fully extended)")
print("Done.")
