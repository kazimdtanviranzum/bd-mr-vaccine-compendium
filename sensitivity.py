"""
sensitivity.py
==============
Explainable global sensitivity analysis (Section 6.2), combining the base
paper's MLR metamodel with the NEW SHAP-based GSA upgrade (Section 4.10).

Design
------
Five input parameters are perturbed simultaneously:
    de     seasonal demand
    p      seasonal vaccine price (purchasing)
    pm     seasonal production cost
    tcc    seasonal transportation cost
    hm_hcc seasonal holding cost (factory + region combined)

Two designs are run:
  * a classical Taguchi L-25 (five levels: -40,-20,0,+20,+40 %), reproducing the
    base-paper Table 9 procedure, and
  * a Latin Hypercube design (LHS_SAMPLES points) for a richer metamodel and
    for SHAP attribution.

For every design point the deterministic MILP is solved and the five economic
outputs (Z, PRC, PUC, TC, HC) are recorded. We then:
  1. fit MLR metamodels (base-paper Eqs 23-27) and report adjusted R2, and
  2. train gradient-boosted metamodels and compute SHAP values, giving the
     model-agnostic share of each input in the variability of each output.

Outputs
-------
outputs/tables/table_taguchi_l25.csv
outputs/tables/table_mlr_coefficients.csv
outputs/tables/table_mlr_adjusted_r2.csv
outputs/tables/table_shap_shares.csv     (Table 10 analogue)
outputs/tables/lhs_design_results.csv
"""

import itertools

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import config as C
import model as M

OUTPUTS = ["Z", "PRC", "PUC", "TC", "HC"]
PARAMS = C.SENS_PARAMS  # de, p, pm, tcc, hm_hcc
RNG = np.random.default_rng(C.SEED)


def _solve_point(base_demand, de_mult, cost_mult):
    """Scale demand and costs, solve, return (Z,PRC,PUC,TC,HC)."""
    demand = {k: v * de_mult for k, v in base_demand.items()}
    m = M.build_det(demand, mult=cost_mult)
    M.solve(m)
    PRC, PUC, TC, HC, _ = m._costs()
    prc, puc, tc, hc = (pyo.value(PRC), pyo.value(PUC), pyo.value(TC),
                        pyo.value(HC))
    return dict(Z=prc + puc + tc + hc, PRC=prc, PUC=puc, TC=tc, HC=hc)


def _mult_from_pct(pcts):
    """pcts: dict param->percent change. Return (de_mult, cost_mult dict)."""
    de_mult = 1.0 + pcts["de"] / 100.0
    cost_mult = {p: 1.0 + pcts[p] / 100.0 for p in ["p", "pm", "tcc", "hm_hcc"]}
    return de_mult, cost_mult


# --------------------------------------------------------------------------- #
# Taguchi L-25 design (5 factors, 5 levels)
# --------------------------------------------------------------------------- #
def taguchi_l25():
    """Standard L25(5^6) orthogonal array columns (use first 5 columns)."""
    # L25 array (levels 1..5). Rows are runs.
    L25 = []
    for i in range(25):
        a = i // 5           # factor 1
        b = i % 5            # factor 2
        c = (i // 5 + i) % 5
        d = (2 * (i // 5) + i) % 5
        e = (3 * (i // 5) + i) % 5
        L25.append([a, b, c, d, e])
    return np.array(L25)


def run_taguchi(base_demand):
    arr = taguchi_l25()
    levels = C.PARAM_LEVELS_PCT
    rows = []
    for run, combo in enumerate(arr, 1):
        pcts = {PARAMS[j]: levels[combo[j]] for j in range(5)}
        de_mult, cost_mult = _mult_from_pct(pcts)
        out = _solve_point(base_demand, de_mult, cost_mult)
        row = dict(run=run)
        row.update({p: levels[combo[j]] for j, p in enumerate(PARAMS)})
        row.update({k: round(v) for k, v in out.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# LHS design
# --------------------------------------------------------------------------- #
def lhs_design(n, k):
    """Latin Hypercube in [0,1]^k, mapped to [-0.4, 0.4] multipliers."""
    cut = np.linspace(0, 1, n + 1)
    u = RNG.uniform(size=(n, k))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(k):
        RNG.shuffle(pts[:, j])
    return -0.4 + pts * 0.8  # in [-0.4, 0.4]


def run_lhs(base_demand, n=C.LHS_SAMPLES):
    X = lhs_design(n, len(PARAMS))  # fractional changes
    rows = []
    for i in range(n):
        pcts = {PARAMS[j]: X[i, j] * 100.0 for j in range(len(PARAMS))}
        de_mult, cost_mult = _mult_from_pct(pcts)
        out = _solve_point(base_demand, de_mult, cost_mult)
        row = {PARAMS[j]: X[i, j] for j in range(len(PARAMS))}
        row.update(out)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MLR metamodels + SHAP GSA
# --------------------------------------------------------------------------- #
def fit_mlr(df):
    coeffs, adjr2 = {}, {}
    Xstd = df[PARAMS].values  # already centred multipliers
    n, k = Xstd.shape
    for out in OUTPUTS:
        y = df[out].values
        lr = LinearRegression().fit(Xstd, y)
        yhat = lr.predict(Xstd)
        r2 = r2_score(y, yhat)
        adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        coeffs[out] = dict(zip(PARAMS, lr.coef_), intercept=lr.intercept_)
        adjr2[out] = 100 * adj
    return pd.DataFrame(coeffs).T, pd.Series(adjr2, name="adj_R2_pct")


def shap_shares(df):
    shares = {}
    for out in OUTPUTS:
        X = df[PARAMS].values
        y = df[out].values
        est = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           random_state=C.SEED, verbosity=0).fit(X, y)
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(X)
        imp = np.abs(sv).mean(axis=0)
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        shares[out] = dict(zip(PARAMS, imp))
    return pd.DataFrame(shares).T[PARAMS]


def main():
    base_demand = M.seasonal_demand()

    print("Running Taguchi L-25 ...")
    tag = run_taguchi(base_demand)
    tag.to_csv(f"{C.TAB_DIR}/table_taguchi_l25.csv", index=False)

    print("Running LHS design (%d points) ..." % C.LHS_SAMPLES)
    lhs = run_lhs(base_demand)
    lhs.to_csv(f"{C.TAB_DIR}/lhs_design_results.csv", index=False)

    coeff, adj = fit_mlr(lhs)
    coeff.to_csv(f"{C.TAB_DIR}/table_mlr_coefficients.csv")
    adj.to_frame().to_csv(f"{C.TAB_DIR}/table_mlr_adjusted_r2.csv")

    shp = shap_shares(lhs)
    shp.round(3).to_csv(f"{C.TAB_DIR}/table_shap_shares.csv")

    print("\n=== MLR adjusted R2 (%) ===")
    print(adj.round(2).to_string())
    print("\n=== SHAP-based share of each input in output variability ===")
    print(shp.round(3).to_string())
    print("\nSHAP ranking for Z:",
          ", ".join(shp.loc["Z"].sort_values(ascending=False).index))
    return shp


if __name__ == "__main__":
    main()
