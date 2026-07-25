"""
seasonality.py
==============
Demand season segmentation (base-paper Section 5.2) plus the NEW ensemble-based
segmentation upgrade (Section 4.8 / 5.2).

Classical step (replicates the MINITAB procedure with scipy):
    For each division x dose, run sequential two-sample t-tests between
    consecutive months, using the forecasted planning-year values (2026-2028)
    as the sample. Consecutive months with p-value > 0.80 are merged into the
    same season.

Ensemble upgrade:
    Repeat the sequential tests using the GAN scenario ensemble (N_SCENARIOS
    draws per month) as the sample. The larger, uncertainty-aware sample yields
    more powerful and more stable season boundaries.

Outputs
-------
outputs/tables/table_pvalues_showcase.csv : Dhaka MR1 consecutive-month p-values
outputs/tables/table_season_assignment_showcase.csv
outputs/tables/table_season_counts.csv    : seasons per division x dose (both methods)
outputs/tables/season_map.csv             : month->season for every division x dose
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as C


def classical_samples():
    """Monthly forecasted demand per division x dose across planning years."""
    d = pd.read_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv")
    return d


def ensemble_samples():
    """GAN scenario ensemble per division x month (dose-agnostic shares scaled)."""
    return pd.read_csv(f"{C.TAB_DIR}/gan_scenarios.csv")


def segment_from_month_samples(month_to_values, threshold=C.SEASON_PVALUE_THRESHOLD):
    """
    Given {month: array_of_values}, run sequential two-sample t-tests between
    consecutive months. Merge into the current season while p > threshold;
    otherwise open a new season. Returns (season_of_month dict, pvalues list).
    """
    seasons = {1: 1}
    pvals = []
    cur = 1
    for m in range(2, 13):
        a = np.asarray(month_to_values[m - 1], dtype=float)
        b = np.asarray(month_to_values[m], dtype=float)
        if len(a) < 2 or len(b) < 2 or (a.std() == 0 and b.std() == 0):
            p = 1.0
        else:
            p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            if np.isnan(p):
                p = 1.0
        pvals.append(p)
        if p <= threshold:      # significantly different -> new season
            cur += 1
        seasons[m] = cur
    return seasons, pvals


def run_classical(demand):
    counts = []
    season_map = []
    showcase_p = None
    showcase_assign = None
    for div in C.DIVISIONS:
        for dose in C.MR_DOSES:
            sub = demand[(demand.division == div) & (demand.dose == dose)]
            m2v = {}
            for m in range(1, 13):
                vals = sub[sub.month == m]["doses"].values
                m2v[m] = vals
            seasons, pvals = segment_from_month_samples(m2v)
            n_seasons = max(seasons.values())
            counts.append(dict(division=div, dose=dose, seasons_classical=n_seasons))
            for m in range(1, 13):
                season_map.append(dict(division=div, dose=dose, month=m,
                                       season=seasons[m]))
            if div == "Dhaka" and dose == "MR1":
                labels = ["Jan-Feb", "Feb-Mar", "Mar-Apr", "Apr-May", "May-Jun",
                          "Jun-Jul", "Jul-Aug", "Aug-Sep", "Sep-Oct", "Oct-Nov",
                          "Nov-Dec"]
                showcase_p = pd.DataFrame({"pair": labels,
                                           "p_value": np.round(pvals, 3)})
                showcase_assign = pd.DataFrame({
                    "month": list(range(1, 13)),
                    "season": [seasons[m] for m in range(1, 13)]})
    return (pd.DataFrame(counts), pd.DataFrame(season_map),
            showcase_p, showcase_assign)


def run_ensemble(scen):
    counts = []
    for div in C.DIVISIONS:
        sub = scen[scen.division == div]
        m2v = {m: sub[sub.month == m]["demand"].values for m in range(1, 13)}
        seasons, _ = segment_from_month_samples(m2v)
        counts.append(dict(division=div, seasons_ensemble=max(seasons.values())))
    return pd.DataFrame(counts)


def main():
    demand = classical_samples()
    scen = ensemble_samples()

    counts_c, season_map, show_p, show_a = run_classical(demand)
    counts_e = run_ensemble(scen)

    show_p.to_csv(f"{C.TAB_DIR}/table_pvalues_showcase.csv", index=False)
    show_a.to_csv(f"{C.TAB_DIR}/table_season_assignment_showcase.csv", index=False)
    season_map.to_csv(f"{C.TAB_DIR}/season_map.csv", index=False)

    # Season counts table: classical (per dose) pivoted + ensemble (per division)
    pivot = counts_c.pivot(index="division", columns="dose",
                           values="seasons_classical")
    pivot = pivot.merge(counts_e.set_index("division"), left_index=True,
                        right_index=True)
    pivot = pivot.reindex(C.DIVISIONS)
    pivot.to_csv(f"{C.TAB_DIR}/table_season_counts.csv")

    print("=== Season counts per division ===")
    print(pivot.to_string())
    print("\nShowcase (Dhaka MR1) consecutive-month p-values:")
    print(show_p.to_string(index=False))
    print("\nDhaka MR1 season assignment:")
    print(show_a.to_string(index=False))
    print("\nMean seasons (classical MR1): %.2f | ensemble: %.2f"
          % (pivot["MR1"].mean(), pivot["seasons_ensemble"].mean()))
    return pivot


if __name__ == "__main__":
    main()
