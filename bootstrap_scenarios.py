"""
bootstrap_scenarios.py
======================
Uncertainty-aware seasonal demand scenario generation by MOVING-BLOCK BOOTSTRAP.

Replaces the conditional GAN. Rationale: the training set contains only ~72
twelve-month share profiles (8 divisions x 2 doses x 5 years), which is far too
few to identify a conditional GAN over 8 divisions; the adversarial game never
leaves its degenerate equilibrium (d_loss pinned at 2*ln2).

Generator, per division d:
  1. Draw a base profile uniformly from that division's historical share
     profiles (preserves the smooth month-to-month structure exactly).
  2. Draw a residual field by CIRCULAR MOVING-BLOCK BOOTSTRAP (block length B)
     from the pooled residual set (profile - division mean shape). Blocks
     preserve local serial dependence that an i.i.d. noise bootstrap destroys.
  3. scenario = base + residual, clipped positive and renormalised to sum 1.

Outputs mirror the GAN module exactly so downstream code is unchanged.
"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/tmp/comp/bd-mr-vaccine-compendium/code")
import config as C

BLOCK_LEN = 3          # months per bootstrap block
RNG = np.random.default_rng(C.SEED)


def historical_profiles(year_max=C.HIST_END_YEAR, year_min=C.HIST_START_YEAR):
    """{division: array (n,12)} of normalised monthly demand share profiles."""
    df = pd.read_csv(C.PANEL_CSV)
    df = df[df.year.between(year_min, year_max)]
    out = {}
    for (div, dose, yr), g in df.groupby(["division", "dose", "year"]):
        g = g.sort_values("month")
        v = g["doses_baseline"].values.astype(float)
        if len(v) != 12 or v.sum() <= 0:
            continue
        out.setdefault(div, []).append(v / v.sum())
    return {k: np.array(v) for k, v in out.items()}


def circular_block_residual(residuals, rng, block=BLOCK_LEN):
    """Assemble one 12-vector residual from circular moving blocks."""
    n, m = residuals.shape
    out = np.zeros(m)
    pos = 0
    while pos < m:
        r = residuals[rng.integers(0, n)]
        start = rng.integers(0, m)
        take = min(block, m - pos)
        idx = [(start + k) % m for k in range(take)]
        out[pos:pos + take] = r[idx]
        pos += take
    return out


def generate(profiles, n_scen, rng):
    """{division: array (n_scen,12)} of generated share profiles."""
    gen = {}
    for div, H in profiles.items():
        mean_shape = H.mean(0)
        resid = H - mean_shape
        S = []
        for _ in range(n_scen):
            base = H[rng.integers(0, len(H))].copy()
            e = circular_block_residual(resid, rng)
            v = np.clip(base + e, 1e-6, None)
            S.append(v / v.sum())
        gen[div] = np.array(S)
    return gen


# ------------------------------------------------------------------ metrics --
def amplitude(V):
    return float(np.mean(V.max(1) - V.min(1)))


def lag1(V):
    a = []
    for v in V:
        vm = v - v.mean()
        d = (vm ** 2).sum()
        if d > 0:
            a.append((vm[1:] * vm[:-1]).sum() / d)
    return float(np.mean(a))


def energy_distance(A, B):
    def pd_(X, Y):
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * pd_(A, B) - pd_(A, A) - pd_(B, B))


def coverage(gen_pool, hist_pool):
    p10 = np.percentile(gen_pool, 10, axis=0)
    p90 = np.percentile(gen_pool, 90, axis=0)
    return float(np.mean((hist_pool >= p10) & (hist_pool <= p90)) * 100)


def per_division_coverage(gen, hist):
    """Honest coverage: band built per division, tested on that division."""
    covs = []
    for div in gen:
        p10 = np.percentile(gen[div], 10, axis=0)
        p90 = np.percentile(gen[div], 90, axis=0)
        H = hist[div]
        covs.append(np.mean((H >= p10) & (H <= p90)) * 100)
    return float(np.mean(covs)), covs


def main():
    # ---------------- in-sample validation (all history 2020-2024) ----------
    prof = historical_profiles()
    hist_pool = np.vstack([prof[d] for d in sorted(prof)])
    gen = generate(prof, 200, np.random.default_rng(C.SEED))
    gen_pool = np.vstack([gen[d] for d in sorted(gen)])

    # i.i.d. noise bootstrap baseline (the one in gan_validation.py)
    rng = np.random.default_rng(0)
    iid = []
    for _ in range(len(gen_pool)):
        b = hist_pool[rng.integers(0, len(hist_pool))].copy()
        b = np.clip(b * (1 + 0.05 * rng.standard_normal(12)), 1e-6, None)
        iid.append(b / b.sum())
    iid = np.array(iid)

    rows = []
    for name, V in [("Historical", hist_pool),
                    ("Block bootstrap", gen_pool),
                    ("i.i.d. noise bootstrap", iid)]:
        rows.append(dict(source=name,
                         amplitude=round(amplitude(V), 4),
                         lag1_autocorr=round(lag1(V), 3),
                         monthshare_mad_vs_hist=round(
                             float(np.mean(np.abs(V.mean(0) - hist_pool.mean(0)))), 5),
                         energy_dist_vs_hist=round(energy_distance(V, hist_pool), 5)))
    tab = pd.DataFrame(rows)
    print("=== In-sample shape fidelity ===")
    print(tab.to_string(index=False))
    tab.to_csv("/tmp/work/bootstrap_validation.csv", index=False)

    cov_pool = coverage(gen_pool, hist_pool)
    cov_div, _ = per_division_coverage(gen, prof)
    print("\nPooled p10-p90 coverage of historical shares : %.1f%%" % cov_pool)
    print("Per-division p10-p90 coverage (mean)         : %.1f%%" % cov_div)

    # ---------------- held-out validation: fit 2020-2023, test 2024 ---------
    prof_tr = historical_profiles(year_max=2023)
    prof_te = historical_profiles(year_min=2024, year_max=2024)
    gen_tr = generate(prof_tr, 200, np.random.default_rng(C.SEED + 1))
    cov_ho, per = per_division_coverage(gen_tr, prof_te)
    print("\n=== Held-out validation (generator fit 2020-2023, tested on 2024) ===")
    print("Mean per-division p10-p90 coverage of 2024 shapes: %.1f%%" % cov_ho)
    ho = pd.DataFrame({"division": sorted(prof_te), "coverage_pct_2024": [round(c, 1) for c in per]})
    print(ho.to_string(index=False))
    ho.to_csv("/tmp/work/bootstrap_heldout_coverage.csv", index=False)

    with open("/tmp/work/bootstrap_coverage.txt", "w") as f:
        f.write("pooled=%.1f\nper_division=%.1f\nheldout_2024=%.1f\n" % (cov_pool, cov_div, cov_ho))

    # ---------------- write scenarios in the GAN module's exact format ------
    gen_scen = generate(prof, C.N_SCENARIOS, np.random.default_rng(C.SEED))
    demand = pd.read_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv")
    annual = demand[demand.year == C.PLAN_YEARS[0]].groupby("division")["doses"].sum()

    rows, bands = [], []
    for div in C.DIVISIONS:
        S = gen_scen[div]
        level = float(annual.get(div, 0.0))
        p10, p50, p90 = np.percentile(S, [10, 50, 90], axis=0)
        for mo in range(12):
            bands.append(dict(division=div, month=mo + 1, p10=p10[mo], p50=p50[mo], p90=p90[mo]))
        for s in range(C.N_SCENARIOS):
            for mo in range(12):
                rows.append(dict(scenario=s, division=div, month=mo + 1,
                                 demand=S[s, mo] * level))
    pd.DataFrame(rows).to_csv("/tmp/work/bootstrap_scenarios.csv", index=False)
    pd.DataFrame(bands).to_csv("/tmp/work/bootstrap_ensemble_bands.csv", index=False)
    print("\nScenarios written: %d rows" % len(rows))


if __name__ == "__main__":
    main()
