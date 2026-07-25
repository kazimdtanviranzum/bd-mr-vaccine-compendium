"""
config.py
=========
Central configuration for the Bangladesh Measles-Rubella (MR) resilient seasonal
supply chain compendium.

Every structural constant used across the pipeline lives here so that a single
edit propagates to data preparation, forecasting, optimization, stress testing
and sensitivity analysis. Nothing in this file is an empirical *result*; these
are modelling choices, published calibration weights and schedule facts. See
data/DATA_SOURCES.md for the provenance of each real-world constant.

All monetary values are in USD. All demand values are in vaccine doses.
"""

import os

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 20260706  # global seed; every stochastic component derives from this

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PANEL_CSV = os.path.join(DATA_DIR, "synthetic_mr_demand_panel.csv")

for _d in (DATA_DIR, OUT_DIR, TAB_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Case-study geography: the eight administrative divisions of Bangladesh
# used as demand regions (Hossain et al. 2024; Shawon et al. 2026 DHIS2
# precedent). Population weights are approximate divisional shares derived from
# the 2022 Bangladesh Population and Housing Census and are used ONLY to shape
# the synthetic demand backbone (Route B). They are calibration constants, not
# results. See DATA_SOURCES.md.
# --------------------------------------------------------------------------- #
DIVISIONS = [
    "Dhaka",
    "Chattogram",
    "Khulna",
    "Rajshahi",
    "Rangpur",
    "Mymensingh",
    "Sylhet",
    "Barishal",
]

# Approximate divisional population shares (fractions sum to 1.0), 2022 census.
POP_SHARE = {
    "Dhaka": 0.276,
    "Chattogram": 0.207,
    "Rajshahi": 0.128,
    "Khulna": 0.108,
    "Rangpur": 0.115,
    "Mymensingh": 0.076,
    "Sylhet": 0.070,
    "Barishal": 0.056,
}

# National annual live births used as the backbone scale for the birth panel.
# Order-of-magnitude figure consistent with UN/SVRS reporting (~3.0 million
# live births per year). Calibration constant, not a result.
NATIONAL_ANNUAL_BIRTHS = 3_000_000

# --------------------------------------------------------------------------- #
# Two supply nodes (source structure). We model the national EPI central store
# and a regional store tier as the two production/supply nodes, mirroring the
# base paper's two-producer structure while reflecting Bangladesh's EPI
# central-depot and divisional-store logistics.
# --------------------------------------------------------------------------- #
SUPPLY_NODES = ["EPI_Central", "Regional_Store"]

# --------------------------------------------------------------------------- #
# EPI schedule facts (real). Bangladesh administers MR1 at 9 months and MR2 at
# 15 months of age. A birth cohort therefore generates MR1 demand 9 months
# later and MR2 demand 15 months later. See DATA_SOURCES.md.
# --------------------------------------------------------------------------- #
MR_DOSES = ["MR1", "MR2"]
DOSE_LAG_MONTHS = {"MR1": 9, "MR2": 15}          # age at administration
# Programmatic coverage applied to the eligible cohort when converting births
# to administered doses. Calibrated to the documented decline (MR1 88.6%->86%,
# MR2 89%->80.7%) rather than assumed. Baseline pre-decline values:
COVERAGE_BASE = {"MR1": 0.886, "MR2": 0.89}
COVERAGE_2024 = {"MR1": 0.86, "MR2": 0.807}
# Wastage multiplier (doses procured per dose administered). 10-dose MR vials
# carry a conventional planning wastage factor ~1.15.
WASTAGE_FACTOR = 1.15

# --------------------------------------------------------------------------- #
# Time horizon
# --------------------------------------------------------------------------- #
HIST_START_YEAR = 2020   # first year of synthetic history
HIST_END_YEAR = 2024     # last fully-historical year (training window basis)
FORECAST_END_YEAR = 2028 # last forecast year
PLAN_YEARS = [2026, 2027, 2028]  # three planning periods (t = 1, 2, 3)
N_PERIODS = len(PLAN_YEARS)

# 80/20 split: months up to and including 2023-12 are train, 2024 is test.
TRAIN_TEST_SPLIT_YEAR = 2024

# --------------------------------------------------------------------------- #
# Seasonality analytics
# --------------------------------------------------------------------------- #
# Sequential two-sample test p-value threshold above which consecutive months
# are grouped into the same season (base paper uses 0.8).
SEASON_PVALUE_THRESHOLD = 0.80
PRODUCTION_SEASONS_PER_PERIOD = 4  # fixed, mirroring the base paper

# --------------------------------------------------------------------------- #
# Real disruption events encoded in the synthetic backbone and stress tests.
# --------------------------------------------------------------------------- #
# 1) 2024-2025 nationwide MR vaccine stockout (WHO DON598; Science 2026).
STOCKOUT_MONTHS = [(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 13)]
STOCKOUT_COVERAGE_FLOOR = 0.59  # only ~59% of eligible children reached in 2025

# 2) 2026 measles outbreak demand surge (WHO DON598; arXiv 2604.25951).
OUTBREAK_YEAR = 2026
OUTBREAK_SURGE_MONTHS = [(2026, m) for m in range(3, 7)]  # Mar-Jun 2026
# Division-level relative burden of the outbreak (suspected-case shares used to
# shape the catch-up surge), derived from WHO DON598 / IJID cumulative counts.
OUTBREAK_BURDEN_SHARE = {
    "Dhaka": 0.38,
    "Rajshahi": 0.16,
    "Chattogram": 0.11,
    "Khulna": 0.065,
    "Rangpur": 0.075,
    "Mymensingh": 0.085,
    "Sylhet": 0.06,
    "Barishal": 0.06,
}
# Emergency campaign catch-up doses (children reached, converted to doses).
# Anchored to the UNICEF appeal (11.9 million MR doses) and the 18 million
# children ultimately reached; distributed across surge months and divisions.
CAMPAIGN_TARGET_DOSES = 18_000_000

# --------------------------------------------------------------------------- #
# Economic parameters (per dose, USD). MR presentation is a combined vaccine;
# a UNICEF/Gavi-consistent planning price is used. See DATA_SOURCES.md.
# --------------------------------------------------------------------------- #
MR_PRICE_USD = 0.60          # per dose planning price (UNICEF/Gavi range)
HOLDING_FRAC = 0.08          # holding cost ~8% of price per season (base paper)
TRANSPORT_COST_PER_KM = 0.001 * MR_PRICE_USD  # per dose per km (base paper)
# Seasonal inflation applied season-by-season to price/holding/transport.
SEASONAL_INFLATION = 0.02

# Resilience policy: minimum share of a season's demand that must be coverable
# from previous-season inventory (base paper uses 0.20).
RESILIENCE_ALPHA = 0.20

# Approximate road distances (km) between supply nodes and divisional capitals.
# EPI_Central is Dhaka-based; Regional_Store is modelled as a central-north hub.
# Distances are illustrative planning constants (see DATA_SOURCES.md).
DISTANCE_KM = {
    ("EPI_Central", "Dhaka"): 15,
    ("EPI_Central", "Chattogram"): 264,
    ("EPI_Central", "Khulna"): 273,
    ("EPI_Central", "Rajshahi"): 256,
    ("EPI_Central", "Rangpur"): 304,
    ("EPI_Central", "Mymensingh"): 120,
    ("EPI_Central", "Sylhet"): 236,
    ("EPI_Central", "Barishal"): 277,
    ("Regional_Store", "Dhaka"): 120,
    ("Regional_Store", "Chattogram"): 350,
    ("Regional_Store", "Khulna"): 330,
    ("Regional_Store", "Rajshahi"): 180,
    ("Regional_Store", "Rangpur"): 190,
    ("Regional_Store", "Mymensingh"): 60,
    ("Regional_Store", "Sylhet"): 300,
    ("Regional_Store", "Barishal"): 360,
}

# --------------------------------------------------------------------------- #
# GAN scenario generation
# --------------------------------------------------------------------------- #
GAN_EPOCHS = 400
GAN_BATCH = 32
GAN_LATENT = 16
GAN_HIDDEN = 64
GAN_LR = 2e-3
N_SCENARIOS = 20   # demand ensembles for the two-stage stochastic model

# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
# Five parameter levels (base paper Table 8): -40, -20, 0, +20, +40 percent.
PARAM_LEVELS_PCT = [-40, -20, 0, 20, 40]
SENS_PARAMS = ["de", "p", "pm", "tcc", "hm_hcc"]  # demand, price, prod, transp, holding
LHS_SAMPLES = 200  # Latin Hypercube samples for the metamodel design
