"""
model.py
========
Resilient seasonal MR supply-chain optimisation.

* Deterministic multi-period, multi-season MILP (Section 4.5-4.7), solved with
  the free HiGHS solver via Pyomo (no CPLEX licence needed).
* Two-stage STOCHASTIC reformulation (NEW contribution 4.9): first-stage
  production and node-opening decisions are here-and-now; shipment, inventory
  and any shortfall are scenario recourse over the GAN demand ensemble. The
  Value of the Stochastic Solution (VSS) is reported.

Network-flow formulation
------------------------
Production slots PT = (period t, production season n); demand slots
DT = (period t, demand season s). A shipment variable ship[v,f,pt,d,dt] moves
doses produced at node f in slot pt to division d in demand slot dt, allowed
only when dt is not earlier than pt and the elapsed time is within the vaccine
shelf life (MR = 12 months). Holding cost accrues with the number of seasons a
dose is carried. The resilience policy requires at least RESILIENCE_ALPHA of
each demand slot to be served from strictly earlier production (carried
inventory), mirroring the base-paper resilience constraints (18)-(21).

Outputs
-------
outputs/tables/opt_cost_breakdown.csv    : PRC/PUC/TC/HC shares (deterministic)
outputs/tables/opt_production_by_season.csv
outputs/tables/opt_shipment_showcase.csv : MR1 shipment plan, first period
outputs/tables/opt_stochastic_summary.csv: deterministic vs stochastic, VSS
"""

import itertools

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs

import config as C

N_DEMAND_SEASONS = 4       # quarters per period (tractable seasonal structure)
N_PROD_SEASONS = C.PRODUCTION_SEASONS_PER_PERIOD  # 4
SHELF_LIFE_SEASONS = 4     # MR shelf life 12 months = 4 quarters
SHORTFALL_PENALTY = 50.0   # USD/dose recourse penalty (stochastic only)


# --------------------------------------------------------------------------- #
# Demand assembly: aggregate monthly baseline demand to quarter-seasons
# --------------------------------------------------------------------------- #
def seasonal_demand():
    """Return dict demand[(v,d,t,s)] = doses (deterministic baseline)."""
    df = pd.read_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv")
    df["season"] = ((df["month"] - 1) // 3) + 1  # 1..4 quarters
    g = (df.groupby(["dose", "division", "year", "season"])["doses"].sum()
         .reset_index())
    t_of = {y: i + 1 for i, y in enumerate(C.PLAN_YEARS)}
    demand = {}
    for r in g.itertuples(index=False):
        demand[(r.dose, r.division, t_of[r.year], int(r.season))] = float(r.doses)
    return demand


def scenario_demand():
    """Return demand[s][(v,d,t,q)] using GAN scenario seasonal shares scaled to
    each planning year's baseline dose level."""
    scen = pd.read_csv(f"{C.TAB_DIR}/gan_scenarios.csv")
    base = pd.read_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv")
    base["q"] = ((base["month"] - 1) // 3) + 1
    # baseline annual level per division/dose/year
    lvl = base.groupby(["dose", "division", "year"])["doses"].sum()
    t_of = {y: i + 1 for i, y in enumerate(C.PLAN_YEARS)}
    # scenario monthly shares -> quarter shares per division
    scen["q"] = ((scen["month"] - 1) // 3) + 1
    qshare = scen.groupby(["scenario", "division", "q"])["demand"].sum()
    qtot = scen.groupby(["scenario", "division"])["demand"].sum()
    out = {}
    for s in sorted(scen["scenario"].unique()):
        dd = {}
        for div in C.DIVISIONS:
            denom = qtot.loc[(s, div)]
            for q in range(1, 5):
                frac = qshare.loc[(s, div, q)] / denom if denom > 0 else 0.25
                for dose in C.MR_DOSES:
                    for y in C.PLAN_YEARS:
                        level = float(lvl.loc[(dose, div, y)])
                        dd[(dose, div, t_of[y], q)] = frac * level
        out[int(s)] = dd
    return out


# --------------------------------------------------------------------------- #
# Cost / structure helpers
# --------------------------------------------------------------------------- #
def _slot_time(t, s):
    """Absolute season index (1-based) across the horizon."""
    return (t - 1) * N_DEMAND_SEASONS + s


def _infl(base, slot_time):
    return base * (1 + C.SEASONAL_INFLATION) ** (slot_time - 1)


def _prod_cost(v, slot_time):
    # production (manufacture/handling) cost per dose; MR2 handling slightly
    # higher. Seasonally inflated.
    b = 0.35 if v == "MR1" else 0.40
    return _infl(b, slot_time)


def _price(slot_time):
    return _infl(C.MR_PRICE_USD, slot_time)


def _holding(slot_time):
    return _infl(C.MR_PRICE_USD * C.HOLDING_FRAC, slot_time)


def _capacity(v, f, t, n):
    """Seasonal production capacity per node/dose/slot (doses)."""
    # EPI_Central carries the bulk; Regional_Store is a surge/secondary tier.
    base_nat = 1_200_000 if f == "EPI_Central" else 700_000
    return base_nat


def _allowed_flows():
    """Enumerate (pt, dt) pairs that respect time order and shelf life."""
    PT = [(t, n) for t in range(1, C.N_PERIODS + 1)
          for n in range(1, N_PROD_SEASONS + 1)]
    DT = [(t, s) for t in range(1, C.N_PERIODS + 1)
          for s in range(1, N_DEMAND_SEASONS + 1)]
    pairs = []
    for (pt_t, pt_n) in PT:
        for (dt_t, dt_s) in DT:
            span = _slot_time(dt_t, dt_s) - _slot_time(pt_t, pt_n)
            if 0 <= span <= SHELF_LIFE_SEASONS:
                pairs.append(((pt_t, pt_n), (dt_t, dt_s), span))
    return PT, DT, pairs


# --------------------------------------------------------------------------- #
# Deterministic MILP
# --------------------------------------------------------------------------- #
def build_det(demand, allow_shortfall=False, cap_override=None, mult=None):
    """mult: optional dict with keys p, pm, tcc, hm_hcc scaling the respective
    unit costs (demand scaling is applied by the caller to `demand`)."""
    mult = mult or {}
    mp = mult.get("p", 1.0)
    mpm = mult.get("pm", 1.0)
    mtcc = mult.get("tcc", 1.0)
    mhold = mult.get("hm_hcc", 1.0)
    m = pyo.ConcreteModel()
    V, F, D = C.MR_DOSES, C.SUPPLY_NODES, C.DIVISIONS
    PT, DT, pairs = _allowed_flows()
    span_of = {(pt, dt): sp for (pt, dt, sp) in pairs}

    ship_keys = [(v, f, pt, d, dt) for v in V for f in F
                 for (pt, dt, _) in pairs for d in D]
    prod_keys = [(v, f, pt) for v in V for f in F for pt in PT]

    m.ship = pyo.Var(ship_keys, domain=pyo.NonNegativeReals)
    m.prod = pyo.Var(prod_keys, domain=pyo.NonNegativeReals)
    m.open = pyo.Var(F, domain=pyo.Binary)
    m.openslot = pyo.Var(prod_keys, domain=pyo.Binary)
    if allow_shortfall:
        short_keys = [(v, d, dt) for v in V for d in D for dt in DT]
        m.short = pyo.Var(short_keys, domain=pyo.NonNegativeReals)

    # Production definition
    m.c_prod = pyo.ConstraintList()
    for v in V:
        for f in F:
            for pt in PT:
                m.c_prod.add(
                    m.prod[v, f, pt] == sum(
                        m.ship[v, f, pt, d, dt] for d in D
                        for (p2, dt, _) in pairs if p2 == pt))

    # Demand satisfaction
    m.c_dem = pyo.ConstraintList()
    for v in V:
        for d in D:
            for dt in DT:
                req = demand.get((v, d, dt[0], dt[1]), 0.0)
                served = sum(m.ship[v, f, pt, d, dt2]
                             for f in F for (pt, dt2, _) in pairs if dt2 == dt)
                if allow_shortfall:
                    m.c_dem.add(served + m.short[v, d, dt] == req)
                else:
                    m.c_dem.add(served == req)

    # Capacity + open linking
    m.c_cap = pyo.ConstraintList()
    for v in V:
        for f in F:
            for pt in PT:
                cap = _capacity(v, f, pt[0], pt[1])
                if cap_override is not None and (v, f, pt) in cap_override:
                    cap = cap_override[(v, f, pt)]
                m.c_cap.add(m.prod[v, f, pt] <= cap * m.openslot[v, f, pt])
                m.c_cap.add(m.openslot[v, f, pt] <= m.open[f])

    # Resilience: >= alpha of each demand slot from strictly earlier production
    m.c_res = pyo.ConstraintList()
    for v in V:
        for d in D:
            for dt in DT:
                req = demand.get((v, d, dt[0], dt[1]), 0.0)
                if req <= 0:
                    continue
                early = [m.ship[v, f, pt, d, dt2]
                         for f in F for (pt, dt2, sp) in pairs
                         if dt2 == dt and sp >= 1]
                if early:
                    m.c_res.add(sum(early) >= C.RESILIENCE_ALPHA * req)

    # Objective
    def cost_expr():
        PRC = sum(m.prod[v, f, pt] * _prod_cost(v, _slot_time(*pt)) * mpm
                  for v in V for f in F for pt in PT)
        PUC = sum(m.ship[v, f, pt, d, dt] * _price(_slot_time(*dt)) * mp
                  for v in V for f in F for (pt, dt, _) in pairs for d in D)
        TC = sum(m.ship[v, f, pt, d, dt] * C.TRANSPORT_COST_PER_KM * mtcc
                 * C.DISTANCE_KM[(f, d)]
                 for v in V for f in F for (pt, dt, _) in pairs for d in D)
        HC = sum(m.ship[v, f, pt, d, dt] * _holding(_slot_time(*dt)) * mhold
                 * span_of[(pt, dt)]
                 for v in V for f in F for (pt, dt, _) in pairs for d in D)
        expr = PRC + PUC + TC + HC
        if allow_shortfall:
            expr = expr + sum(m.short[v, d, dt] * SHORTFALL_PENALTY
                              for v in V for d in D for dt in DT)
        return PRC, PUC, TC, HC, expr

    m._costs = cost_expr
    _, _, _, _, total = cost_expr()
    m.obj = pyo.Objective(expr=total, sense=pyo.minimize)
    m._pairs = pairs
    return m


def solve(m):
    res = Highs().solve(m)
    return res


def report_deterministic(m):
    PRC, PUC, TC, HC, _ = m._costs()
    prc, puc, tc, hc = (pyo.value(PRC), pyo.value(PUC), pyo.value(TC), pyo.value(HC))
    Z = prc + puc + tc + hc
    breakdown = pd.DataFrame({
        "component": ["PRC", "PUC", "TC", "HC", "Z"],
        "value_usd": [prc, puc, tc, hc, Z],
        "share_pct": [100 * prc / Z, 100 * puc / Z, 100 * tc / Z, 100 * hc / Z, 100.0],
    })
    breakdown.to_csv(f"{C.TAB_DIR}/opt_cost_breakdown.csv", index=False)

    # Production by (period, production season) summed over dose & node
    rows = []
    for v in C.MR_DOSES:
        for f in C.SUPPLY_NODES:
            for t in range(1, C.N_PERIODS + 1):
                for n in range(1, N_PROD_SEASONS + 1):
                    val = pyo.value(m.prod[v, f, (t, n)])
                    rows.append(dict(dose=v, node=f, period=t, prod_season=n,
                                     doses=val))
    prod_df = pd.DataFrame(rows)
    (prod_df.groupby(["period", "prod_season"])["doses"].sum().reset_index()
     .to_csv(f"{C.TAB_DIR}/opt_production_by_season.csv", index=False))

    # Shipment showcase: MR1, first period
    ship_rows = []
    PT, DT, pairs = _allowed_flows()
    for f in C.SUPPLY_NODES:
        for (pt, dt, _) in pairs:
            if dt[0] != 1:
                continue
            for d in C.DIVISIONS:
                val = pyo.value(m.ship["MR1", f, pt, d, dt])
                if val > 1e-6:
                    ship_rows.append(dict(node=f, prod_period=pt[0],
                                          prod_season=pt[1], demand_period=dt[0],
                                          demand_season=dt[1], division=d,
                                          doses=round(val)))
    pd.DataFrame(ship_rows).sort_values(["division", "prod_season"]).to_csv(
        f"{C.TAB_DIR}/opt_shipment_showcase.csv", index=False)
    return breakdown, Z


# --------------------------------------------------------------------------- #
# Two-stage stochastic reformulation
# --------------------------------------------------------------------------- #
def build_stochastic(scen_dem, first_stage_fixed=None):
    """
    First stage: prod, open, openslot (shared across scenarios).
    Second stage: ship_s, short_s per scenario. Objective = first-stage
    production/open cost + expected second-stage (purchasing+transport+holding
    +shortfall penalty).
    If first_stage_fixed is provided (a dict of prod values), production is
    fixed to evaluate a given here-and-now plan under all scenarios.
    """
    m = pyo.ConcreteModel()
    V, F, D = C.MR_DOSES, C.SUPPLY_NODES, C.DIVISIONS
    PT, DT, pairs = _allowed_flows()
    span_of = {(pt, dt): sp for (pt, dt, sp) in pairs}
    S = sorted(scen_dem.keys())
    prob = 1.0 / len(S)

    prod_keys = [(v, f, pt) for v in V for f in F for pt in PT]
    m.prod = pyo.Var(prod_keys, domain=pyo.NonNegativeReals)
    m.open = pyo.Var(F, domain=pyo.Binary)
    m.openslot = pyo.Var(prod_keys, domain=pyo.Binary)

    ship_keys = [(s, v, f, pt, d, dt) for s in S for v in V for f in F
                 for (pt, dt, _) in pairs for d in D]
    short_keys = [(s, v, d, dt) for s in S for v in V for d in D for dt in DT]
    m.ship = pyo.Var(ship_keys, domain=pyo.NonNegativeReals)
    m.short = pyo.Var(short_keys, domain=pyo.NonNegativeReals)

    m.c = pyo.ConstraintList()
    # Capacity / open
    for v in V:
        for f in F:
            for pt in PT:
                cap = _capacity(v, f, pt[0], pt[1])
                m.c.add(m.prod[v, f, pt] <= cap * m.openslot[v, f, pt])
                m.c.add(m.openslot[v, f, pt] <= m.open[f])
    if first_stage_fixed is not None:
        for k, val in first_stage_fixed.items():
            m.prod[k].fix(val)

    # Per scenario: production balance, demand, resilience
    for s in S:
        dem = scen_dem[s]
        for v in V:
            for f in F:
                for pt in PT:
                    m.c.add(m.prod[v, f, pt] >= sum(
                        m.ship[s, v, f, pt, d, dt] for d in D
                        for (p2, dt, _) in pairs if p2 == pt))
        for v in V:
            for d in D:
                for dt in DT:
                    req = dem.get((v, d, dt[0], dt[1]), 0.0)
                    served = sum(m.ship[s, v, f, pt, d, dt2]
                                 for f in F for (pt, dt2, _) in pairs if dt2 == dt)
                    m.c.add(served + m.short[s, v, d, dt] == req)
                    if req > 0:
                        early = [m.ship[s, v, f, pt, d, dt2]
                                 for f in F for (pt, dt2, sp) in pairs
                                 if dt2 == dt and sp >= 1]
                        if early:
                            m.c.add(sum(early) >= C.RESILIENCE_ALPHA
                                    * (req - m.short[s, v, d, dt]))

    first = sum(m.prod[v, f, pt] * _prod_cost(v, _slot_time(*pt))
                for v in V for f in F for pt in PT)
    second = 0
    for s in S:
        puc = sum(m.ship[s, v, f, pt, d, dt] * _price(_slot_time(*dt))
                  for v in V for f in F for (pt, dt, _) in pairs for d in D)
        tc = sum(m.ship[s, v, f, pt, d, dt] * C.TRANSPORT_COST_PER_KM
                 * C.DISTANCE_KM[(f, d)]
                 for v in V for f in F for (pt, dt, _) in pairs for d in D)
        hc = sum(m.ship[s, v, f, pt, d, dt] * _holding(_slot_time(*dt))
                 * span_of[(pt, dt)]
                 for v in V for f in F for (pt, dt, _) in pairs for d in D)
        pen = sum(m.short[s, v, d, dt] * SHORTFALL_PENALTY
                  for v in V for d in D for dt in DT)
        second = second + prob * (puc + tc + hc + pen)
    m.obj = pyo.Objective(expr=first + second, sense=pyo.minimize)
    m._prod_keys = prod_keys
    return m


def run_stochastic():
    scen = scenario_demand()
    # (1) Stochastic solution: optimise here-and-now over all scenarios.
    m_sp = build_stochastic(scen)
    solve(m_sp)
    z_sp = pyo.value(m_sp.obj)
    prod_sp = {k: pyo.value(m_sp.prod[k]) for k in m_sp._prod_keys}

    # (2) Expected value (mean-scenario) solution: solve deterministic on the
    # mean demand, then evaluate that fixed production under all scenarios.
    mean_dem = {}
    keys = set().union(*[set(d.keys()) for d in scen.values()])
    for k in keys:
        mean_dem[(k[0], k[1], k[2], k[3])] = np.mean([scen[s].get(k, 0.0)
                                                      for s in scen])
    m_ev = build_det(mean_dem, allow_shortfall=True)
    solve(m_ev)
    PT, _, _ = _allowed_flows()
    prod_ev = {(v, f, pt): pyo.value(m_ev.prod[v, f, pt])
               for v in C.MR_DOSES for f in C.SUPPLY_NODES for pt in PT}
    m_eev = build_stochastic(scen, first_stage_fixed=prod_ev)
    solve(m_eev)
    z_eev = pyo.value(m_eev.obj)

    vss = z_eev - z_sp  # Value of the Stochastic Solution (>= 0)
    summary = pd.DataFrame({
        "quantity": ["Z_stochastic (RP)", "Z_expected_value_solution (EEV)",
                     "VSS = EEV - RP", "VSS_pct_of_RP"],
        "value_usd": [z_sp, z_eev, vss, 100 * vss / z_sp if z_sp else 0.0],
    })
    summary.to_csv(f"{C.TAB_DIR}/opt_stochastic_summary.csv", index=False)
    return summary


def main():
    demand = seasonal_demand()
    m = build_det(demand)
    res = solve(m)
    breakdown, Z = report_deterministic(m)
    print("=== Deterministic MILP solved (HiGHS) ===")
    print("Termination:", res.termination_condition)
    print(breakdown.round(2).to_string(index=False))
    print("\nTotal cost Z = %.0f USD" % Z)

    print("\n=== Two-stage stochastic reformulation ===")
    summary = run_stochastic()
    print(summary.round(2).to_string(index=False))
    return breakdown, summary


if __name__ == "__main__":
    main()
