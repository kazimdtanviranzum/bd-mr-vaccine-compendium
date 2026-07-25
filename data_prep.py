"""
data_prep.py
============
Builds the monthly divisional Measles-Rubella (MR) demand panel for the eight
administrative divisions of Bangladesh (Route B: DISCLOSED SYNTHETIC).

The panel is *synthetically generated* to match documented monthly divisional
birth trends reported for Bangladesh (Hossain et al. 2024; Shawon et al. 2026
DHIS2 precedent) and then converted to MR vaccine demand using the confirmed
EPI schedule (MR1 at 9 months, MR2 at 15 months).

Design (faithful to the base paper):
  * The smooth BIRTH backbone is the forecasting target (predictable).
  * Baseline MR demand = forecast-able births x declining coverage x wastage.
  * Two REAL disruptions are stored as SEPARATE overlay columns, applied only
    in the stress test, so they never contaminate forecaster accuracy:
        - 2024-2025 nationwide MR stockout (coverage floor), and
        - 2026 measles outbreak catch-up surge.

NOTHING here is real DHIS2 divisional data. The output CSV is explicitly named
`synthetic_mr_demand_panel.csv`. A real DHIS2 panel with the same schema can be
dropped in with no downstream code changes.

Output schema (long format):
    year, month, division, dose, births, coverage_baseline,
    doses_baseline, stockout_flag, surge_doses, doses_disrupted
"""

import numpy as np
import pandas as pd

import config as C


def _month_index(start_year, end_year):
    out = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            out.append((y, m))
    return out


def build_birth_panel(rng):
    """Synthetic monthly divisional live-birth panel, 2019-2027 (smooth)."""
    months = _month_index(C.HIST_START_YEAR - 1, C.FORECAST_END_YEAR - 1)
    seasonal = np.array(
        [0.98, 0.95, 0.93, 0.96, 1.00, 1.03, 1.06, 1.09, 1.08, 1.05, 1.02, 0.99]
    )
    seasonal = seasonal / seasonal.mean()

    rows = []
    for div in C.DIVISIONS:
        share = C.POP_SHARE[div]
        base_monthly = C.NATIONAL_ANNUAL_BIRTHS * share / 12.0
        trend = rng.uniform(-0.010, 0.005)
        for (y, m) in months:
            yr_offset = y - (C.HIST_START_YEAR - 1)
            level = base_monthly * (1.0 + trend) ** yr_offset
            births = level * seasonal[m - 1] * rng.normal(1.0, 0.03)
            rows.append((y, m, div, max(births, 0.0)))
    return pd.DataFrame(rows, columns=["year", "month", "division", "births"])


def _coverage_baseline(year, dose):
    """Smooth declining coverage (no stockout); the base-paper style backbone."""
    base, end = C.COVERAGE_BASE[dose], C.COVERAGE_2024[dose]
    if year <= 2019:
        return base
    if year >= 2024:
        return end
    frac = (year - 2019) / (2024 - 2019)
    return base + (end - base) * frac


def build_panel(rng):
    births = build_birth_panel(rng)
    births["ym"] = births["year"] * 12 + (births["month"] - 1)
    lookup = {(r.division, r.ym): r.births for r in births.itertuples(index=False)}

    records = []
    deferred = {d: 0.0 for d in C.DIVISIONS}
    for (y, m) in _month_index(C.HIST_START_YEAR, C.FORECAST_END_YEAR):
        ym = y * 12 + (m - 1)
        stockout = (y, m) in C.STOCKOUT_MONTHS
        for div in C.DIVISIONS:
            for dose in C.MR_DOSES:
                born_ym = ym - C.DOSE_LAG_MONTHS[dose]
                b = lookup.get((div, born_ym), np.nan)
                if np.isnan(b):
                    continue
                cov = _coverage_baseline(y, dose)
                baseline = b * cov * C.WASTAGE_FACTOR
                if stockout:
                    admin_cov = min(cov, C.STOCKOUT_COVERAGE_FLOOR)
                    deferred[div] += max((cov - admin_cov) * b * C.WASTAGE_FACTOR, 0.0)
                records.append(
                    dict(year=y, month=m, division=div, dose=dose, births=b,
                         coverage_baseline=cov, doses_baseline=baseline,
                         stockout_flag=int(stockout))
                )
    df = pd.DataFrame.from_records(records)

    # 2026 outbreak catch-up surge overlay (separate column).
    surge_set = set(C.OUTBREAK_SURGE_MONTHS)
    n_surge = len(C.OUTBREAK_SURGE_MONTHS)
    campaign_total = max(sum(deferred.values()), C.CAMPAIGN_TARGET_DOSES) * C.WASTAGE_FACTOR

    def _surge(row):
        if (row.year, row.month) in surge_set:
            share = C.OUTBREAK_BURDEN_SHARE[row.division]
            dose_split = 0.6 if row.dose == "MR1" else 0.4
            return campaign_total * share * dose_split / n_surge
        return 0.0

    df["surge_doses"] = df.apply(_surge, axis=1)
    # Disrupted demand = stockout-suppressed baseline + surge catch-up.
    df["doses_disrupted"] = np.where(
        df["stockout_flag"] == 1,
        df["doses_baseline"] * (C.STOCKOUT_COVERAGE_FLOOR /
                                df["coverage_baseline"].clip(lower=1e-9)).clip(upper=1.0),
        df["doses_baseline"],
    ) + df["surge_doses"]

    df = df.sort_values(["division", "dose", "year", "month"]).reset_index(drop=True)
    return df


def main():
    rng = np.random.default_rng(C.SEED)
    panel = build_panel(rng)
    panel.to_csv(C.PANEL_CSV, index=False)
    summary = (panel.groupby(["division", "dose"])["doses_baseline"].sum()
               .round(0).reset_index()
               .pivot(index="division", columns="dose", values="doses_baseline"))
    summary.to_csv(f"{C.TAB_DIR}/panel_summary.csv")
    print("Wrote synthetic panel:", C.PANEL_CSV, "| rows:", len(panel))
    print("\nBaseline MR demand by division/dose (2020-2028, doses):")
    print(summary.to_string())
    print("\nDisruption overlays present:",
          "stockout months =", panel["stockout_flag"].sum(),
          "| total surge doses =", round(panel["surge_doses"].sum()))
    return panel


if __name__ == "__main__":
    main()
