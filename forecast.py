"""
forecast.py
===========
Forecasts monthly divisional BIRTHS with five models and reports the real
out-of-sample R2 accuracy table (base-paper Table 2 analogue). Births are the
smooth, predictable backbone; MR demand is then derived from forecasted births
via the EPI schedule (MR1 at 9 months, MR2 at 15 months). Disruptions
(2024-2025 stockout, 2026 outbreak surge) are handled downstream in the stress
test, so they never contaminate forecaster accuracy.

Models: ARIMA, SARIMA, Linear Regression, Random Forest, XGBoost.
Split  : 80/20 -> months through 2023-12 = train, 2024 = test.

Outputs
-------
outputs/tables/table_forecast_r2.csv        : R2 per division x model
outputs/tables/best_model_per_division.csv
outputs/tables/forecast_demand_2026_2028.csv: baseline MR demand for planning
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import config as C

warnings.filterwarnings("ignore")

FEATURES = ["lag1", "lag2", "lag12", "month_sin", "month_cos"]


def load_births():
    """Monthly divisional births 2019-2027 (unique per division/month)."""
    df = pd.read_csv(C.PANEL_CSV)
    b = (df.groupby(["division", "year", "month"])["births"].first().reset_index()
         .rename(columns={"births": "y"}))
    return b.sort_values(["division", "year", "month"]).reset_index(drop=True)


def _lags(series):
    d = series.copy().reset_index(drop=True)
    d["lag1"] = d["y"].shift(1)
    d["lag2"] = d["y"].shift(2)
    d["lag12"] = d["y"].shift(12)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12.0)
    return d


def _split(series):
    return (series[series.year < C.TRAIN_TEST_SPLIT_YEAR],
            series[series.year == C.TRAIN_TEST_SPLIT_YEAR])


def r2_arima(series, seasonal=False):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train, test = _split(series)
    seas = (1, 1, 0, 12) if seasonal else (0, 0, 0, 0)
    try:
        m = SARIMAX(train["y"].astype(float).values, order=(1, 1, 1),
                    seasonal_order=seas, enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        return r2_score(test["y"].values, m.forecast(steps=len(test)))
    except Exception:
        return np.nan


def _ml_r2(series, est):
    d = _lags(series).dropna().reset_index(drop=True)
    tr, te = d[d.year < C.TRAIN_TEST_SPLIT_YEAR], d[d.year == C.TRAIN_TEST_SPLIT_YEAR]
    if len(te) == 0 or len(tr) < 12:
        return np.nan
    est.fit(tr[FEATURES], tr["y"])
    return r2_score(te["y"].values, est.predict(te[FEATURES]))


def _xgb():
    return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=C.SEED, verbosity=0)


def _rf():
    return RandomForestRegressor(n_estimators=300, random_state=C.SEED, n_jobs=-1)


MODELS = {
    "ARIMA": lambda s: r2_arima(s, False),
    "SARIMA": lambda s: r2_arima(s, True),
    "LinearRegression": lambda s: _ml_r2(s, LinearRegression()),
    "RandomForest": lambda s: _ml_r2(s, _rf()),
    "XGBoost": lambda s: _ml_r2(s, _xgb()),
}


def forecast_future_births(series, model_name):
    hist = series[series.year <= C.TRAIN_TEST_SPLIT_YEAR].copy()
    fmonths = [(y, m) for y in range(2025, C.FORECAST_END_YEAR + 1)
               for m in range(1, 13)]
    if model_name in ("ARIMA", "SARIMA"):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seas = (1, 1, 0, 12) if model_name == "SARIMA" else (0, 0, 0, 0)
        m = SARIMAX(hist["y"].astype(float).values, order=(1, 1, 1),
                    seasonal_order=seas, enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        preds = m.forecast(steps=len(fmonths))
        return pd.DataFrame([dict(year=y, month=mo, births=max(float(p), 0.0))
                             for (y, mo), p in zip(fmonths, preds)])
    est = {"LinearRegression": LinearRegression(), "RandomForest": _rf(),
           "XGBoost": _xgb()}[model_name]
    d = _lags(hist).dropna()
    est.fit(d[FEATURES], d["y"])
    buf = list(hist["y"].astype(float).values)
    out = []
    for (y, mo) in fmonths:
        feat = np.array([[buf[-1], buf[-2], buf[-12],
                          np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)]])
        yhat = max(float(est.predict(feat)[0]), 0.0)
        out.append(dict(year=y, month=mo, births=yhat))
        buf.append(yhat)
    return pd.DataFrame(out)


def births_to_baseline_demand(all_births):
    """Convert full monthly births (hist + forecast) to baseline MR demand for
    the 2026-2028 planning periods via EPI lags."""
    all_births = all_births.copy()
    all_births["ym"] = all_births["year"] * 12 + (all_births["month"] - 1)
    lut = {(r.division, r.ym): r.births for r in all_births.itertuples(index=False)}
    rows = []
    for y in C.PLAN_YEARS:
        for mo in range(1, 13):
            ym = y * 12 + (mo - 1)
            for div in C.DIVISIONS:
                for dose in C.MR_DOSES:
                    b = lut.get((div, ym - C.DOSE_LAG_MONTHS[dose]))
                    if b is None:
                        continue
                    cov = C.COVERAGE_2024[dose]  # end-of-decline baseline
                    rows.append(dict(year=y, month=mo, division=div, dose=dose,
                                     doses=b * cov * C.WASTAGE_FACTOR))
    return pd.DataFrame(rows)


def main():
    births = load_births()
    r2_rows = []
    for div in C.DIVISIONS:
        s = births[births.division == div].reset_index(drop=True)
        row = {"division": div}
        for name, fn in MODELS.items():
            row[name] = fn(s)
        r2_rows.append(row)
    r2 = pd.DataFrame(r2_rows).set_index("division")[list(MODELS.keys())]
    r2.round(4).to_csv(f"{C.TAB_DIR}/table_forecast_r2.csv")
    best = r2.idxmax(axis=1)
    best.to_frame("best_model").to_csv(f"{C.TAB_DIR}/best_model_per_division.csv")

    # Forecast future births per division (best model), then derive demand.
    fut = []
    for div in C.DIVISIONS:
        s = births[births.division == div].reset_index(drop=True)
        fdf = forecast_future_births(s, best[div])
        fdf["division"] = div
        fut.append(fdf)
    fut = pd.concat(fut, ignore_index=True)
    hist = births[births.year <= C.TRAIN_TEST_SPLIT_YEAR][
        ["year", "month", "division", "y"]].rename(columns={"y": "births"})
    full = pd.concat([hist, fut[["year", "month", "division", "births"]]],
                     ignore_index=True)
    demand = births_to_baseline_demand(full)
    demand.to_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv", index=False)

    print("=== Out-of-sample R2 on monthly births (test = 2024) ===")
    print((r2 * 100).round(1).to_string())
    print("\nBest model per division:")
    print(best.to_string())
    print("\nMean R2 by model (%):")
    print((r2.mean() * 100).round(1).to_string())
    print("\nWrote baseline planning demand rows:", len(demand))
    return r2


if __name__ == "__main__":
    main()
