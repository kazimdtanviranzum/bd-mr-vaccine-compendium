"""
stress_test.py
==============
Seasonal resilience stress tests on the designed MR network (Section 6.1),
encoding the two REAL disruption events:

  1. One-node-loss (2024-2025 stockout analogue): the seasonal production
     capacity of the EPI central store is set to zero during a studied season
     (as happened when MR vaccine ran out at the Dhaka central depot in
     2024-2025). Can the remaining regional tier plus carried inventory still
     serve demand?

  2. Outbreak surge (2026 measles outbreak): demand in the surge divisions is
     scaled up by the documented catch-up burden during the first planning
     period. Can the network absorb the surge without shortfall?

For each test we re-solve the deterministic model with shortfall permitted and
report the SERVICE LEVEL (share of demand met). We also report the base-paper
resilience-balance table (TSIF, TSIC, PSPRF, TSDC, Balance, service level) for
the one-node-loss case, computed from the optimal baseline plan.

Outputs
-------
outputs/tables/table_resilience_balance.csv
outputs/tables/table_stress_service_levels.csv
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import config as C
import model as M


def _total_demand(demand):
    return sum(demand.values())


def _served_and_short(m, demand):
    """Total served and shortfall from a solved shortfall-enabled model."""
    total = _total_demand(demand)
    short = 0.0
    for v in C.MR_DOSES:
        for d in C.DIVISIONS:
            for dt in [(t, s) for t in range(1, C.N_PERIODS + 1)
                       for s in range(1, M.N_DEMAND_SEASONS + 1)]:
                short += pyo.value(m.short[v, d, dt])
    return total - short, short, total


def one_node_loss(demand, lost_node="EPI_Central", lost_period=2, lost_season=1):
    """Zero the lost node's capacity in the studied production season."""
    PT, _, _ = M._allowed_flows()
    cap_override = {}
    for v in C.MR_DOSES:
        cap_override[(v, lost_node, (lost_period, lost_season))] = 0.0
    m = M.build_det(demand, allow_shortfall=True, cap_override=cap_override)
    M.solve(m)
    served, short, total = _served_and_short(m, demand)
    return served, short, total, m


def outbreak_surge(demand):
    """Scale up demand in surge divisions during the first planning period to
    emulate the 2026 outbreak catch-up surge."""
    surged = dict(demand)
    # surge multiplier per division from documented burden shares (normalised
    # so the mean surge division roughly doubles first-period demand).
    peak = max(C.OUTBREAK_BURDEN_SHARE.values())
    for (v, d, t, s), val in demand.items():
        if t == 1:  # first planning period bears the surge
            mult = 1.0 + 3.0 * (C.OUTBREAK_BURDEN_SHARE[d] / peak)
            surged[(v, d, t, s)] = val * mult
    m = M.build_det(surged, allow_shortfall=True)
    M.solve(m)
    served, short, total = _served_and_short(m, surged)
    return served, short, total


def resilience_balance(demand, base_model, studied_period=2, studied_season=2):
    """
    Base-paper Table 7 analogue, computed from the optimal baseline plan.
    For each dose and the studied (period, season):
      TSDC  = total demand of all customers in the studied season
      PSPRF = planned production in the REMAINING node during the studied season
      TSIF  = factory-carried inventory available from earlier production
      TSIC  = customer-held inventory (resilience carryover) into the season
      Balance = TSIF + TSIC + PSPRF - TSDC ; service level = (available)/TSDC
    """
    PT, DT, pairs = M._allowed_flows()
    remaining = "Regional_Store"  # node that survives the loss
    st_time = M._slot_time(studied_period, studied_season)
    rows = []
    for v in C.MR_DOSES:
        # TSDC
        tsdc = sum(demand.get((v, d, studied_period, studied_season), 0.0)
                   for d in C.DIVISIONS)
        # PSPRF: production in remaining node during studied season
        psprf = pyo.value(base_model.prod[v, remaining, (studied_period,
                                                         studied_season)])
        # TSIF: factory pipeline stock crossing into the studied season =
        # doses produced strictly before the season and consumed at/after it.
        tsif = 0.0
        for f in C.SUPPLY_NODES:
            for (pt, dt, sp) in pairs:
                if M._slot_time(*pt) < st_time <= M._slot_time(*dt):
                    for d in C.DIVISIONS:
                        tsif += pyo.value(base_model.ship[v, f, pt, d, dt])
        # TSIC: mandated regional resilience carryover into the studied season
        tsic = C.RESILIENCE_ALPHA * tsdc
        available = tsif + tsic + psprf
        balance = available - tsdc
        service = 100.0 * available / tsdc if tsdc > 0 else np.nan
        rows.append(dict(dose=v, TSDC=round(tsdc), PSPRF=round(psprf),
                         TSIF=round(tsif), TSIC=round(tsic),
                         Balance=round(balance),
                         service_level_pct=round(service, 1)))
    return pd.DataFrame(rows)


def main():
    demand = M.seasonal_demand()
    base = M.build_det(demand)
    M.solve(base)

    # Resilience balance table (one-node-loss, from optimal plan)
    bal = resilience_balance(demand, base)
    bal.to_csv(f"{C.TAB_DIR}/table_resilience_balance.csv", index=False)

    # Re-solve service levels under each disruption
    served_n, short_n, total_n, _ = one_node_loss(demand)
    served_s, short_s, total_s = outbreak_surge(demand)

    svc = pd.DataFrame({
        "scenario": ["One-node loss (stockout analogue)",
                     "Outbreak demand surge (2026)"],
        "demand_doses": [round(total_n), round(total_s)],
        "served_doses": [round(served_n), round(served_s)],
        "shortfall_doses": [round(short_n), round(short_s)],
        "service_level_pct": [round(100 * served_n / total_n, 2),
                              round(100 * served_s / total_s, 2)],
    })
    svc.to_csv(f"{C.TAB_DIR}/table_stress_service_levels.csv", index=False)

    print("=== Resilience balance (studied season, one-node loss) ===")
    print(bal.to_string(index=False))
    print("\n=== Stress-test service levels ===")
    print(svc.to_string(index=False))
    return bal, svc


if __name__ == "__main__":
    main()
