# Bangladesh MR Vaccine — Resilient Seasonal Supply Chain Compendium

Replication code for a resilient, seasonal supply-chain study of the **Measles–Rubella (MR)** vaccine across the eight administrative divisions of Bangladesh. The compendium runs an end-to-end pipeline — from demand construction through forecasting, uncertainty-aware scenario generation, seasonal optimisation, resilience stress testing, and explainable global sensitivity analysis — and regenerates every table and figure from data with no manual editing.

The optimisation is built entirely on **free solvers** (Pyomo + HiGHS): no CPLEX licence is required.

> **Data disclosure.** The demand panel shipped here (`synthetic_mr_demand_panel.csv`) is **disclosed synthetic** — generated to match documented monthly divisional birth trends and the confirmed EPI schedule. It is *not* real DHIS2 data. A real panel with the same schema can be dropped in with no downstream code changes. Structural constants (population shares, schedule facts, disruption events, prices) come from published sources; see the `DATA_SOURCES.md` provenance note referenced in `config.py`.

---

## What the study models

- **Geography:** the eight divisions of Bangladesh (Dhaka, Chattogram, Khulna, Rajshahi, Rangpur, Mymensingh, Sylhet, Barishal), weighted by 2022-census divisional population shares.
- **Schedule:** MR1 administered at 9 months and MR2 at 15 months of age, so each birth cohort generates dose demand 9 and 15 months later.
- **Supply structure:** two nodes — a national EPI central store and a regional store tier.
- **Real disruption events** encoded in the backbone and stress tests:
  - the **2024–2025 nationwide MR stockout** (coverage floor ~59%), and
  - the **2026 measles outbreak** catch-up surge (anchored to the UNICEF appeal and reached-children figures).

## Pipeline

The stages run in dependency order via `run_all.py`:

| # | Module | What it does |
|---|--------|--------------|
| 1 | `data_prep.py` | Builds the monthly divisional MR demand panel. Predictable birth backbone → baseline demand (coverage × wastage); the two real disruptions are stored as **separate overlay columns** so they never contaminate forecaster accuracy. |
| 2 | `forecast.py` | Forecasts monthly divisional **births** with five models (ARIMA, SARIMA, Linear Regression, Random Forest, XGBoost), reports out-of-sample R², and derives baseline MR demand for the planning years. 80/20 split (≤2023-12 train, 2024 test). |
| 3 | `gan_scenarios.py` / `bootstrap_scenarios.py` | Uncertainty-aware demand ensembles. A compact conditional GAN (pure NumPy) learns seasonal demand shapes; **`bootstrap_scenarios.py` is the recommended replacement** — a circular moving-block bootstrap that is far more stable given the small (~72-profile) training set. Outputs are interchangeable. |
| 4 | `seasonality.py` | Segments demand into production seasons via sequential two-sample tests between consecutive months (p > 0.80 → same season), plus an ensemble-based upgrade that uses the scenario draws for more stable boundaries. |
| 5 | `model.py` | The core optimisation: a deterministic multi-period, multi-season **MILP** (HiGHS via Pyomo) and a **two-stage stochastic** reformulation over the demand ensemble. Reports the Value of the Stochastic Solution (VSS) and enforces a resilience policy (each demand slot must be partly served from carried inventory). |
| 6 | `stress_test.py` | Seasonal resilience tests: one-node-loss (stockout analogue) and outbreak surge. Re-solves with shortfall permitted and reports service levels plus the resilience-balance table. |
| 7 | `sensitivity.py` | Explainable global sensitivity analysis: a Taguchi L-25 design and a Latin Hypercube design, MLR metamodels (adjusted R²), and **SHAP** attribution of each input's share in output variability. |
| 8 | `make_figures.py` | Regenerates all twelve manuscript figures from the real pipeline outputs — no hand-typed numbers. |

## Repository contents

```
config.py                        Central configuration: geography, schedule,
                                 disruptions, economics, solver/GAN/sensitivity params
data_prep.py                     Stage 1  — build the demand panel
forecast.py                      Stage 2  — five-model birth forecasting
gan_scenarios.py                 Stage 3  — conditional GAN scenario ensembles
bootstrap_scenarios.py           Stage 3' — moving-block bootstrap (GAN replacement)
seasonality.py                   Stage 4  — season segmentation
model.py                         Stage 5  — deterministic + stochastic MILP
stress_test.py                   Stage 6  — resilience stress tests
sensitivity.py                   Stage 7  — Taguchi/LHS + MLR + SHAP GSA
make_figures.py                  Stage 8  — regenerate all figures
run_all.py                       Runs the full pipeline in order

synthetic_mr_demand_panel.csv    Disclosed-synthetic demand panel (shipped copy)
bootstrap_validation.csv         Bootstrap vs historical fidelity diagnostics
bootstrap_heldout_coverage.csv   Held-out 2024 coverage by division
requirements.txt                 Python dependencies
LICENSE                          MIT
```

## Requirements

- Python 3.12 (tested)
- Dependencies in `requirements.txt`: numpy, pandas, scipy, scikit-learn, statsmodels, xgboost, pyomo, highspy, shap, matplotlib

The conditional GAN is pure NumPy, so **no deep-learning framework is needed**. To swap in a PyTorch implementation, install `torch>=2.2` and replace `ConditionalGAN`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the whole pipeline:

```bash
python run_all.py
```

Or run any stage on its own (each module exposes a `main()` and respects the dependency order — earlier stages must have produced their outputs first):

```bash
python forecast.py
python model.py
```

Tables are written to `outputs/tables/` and figures to `outputs/figures/`, created automatically. `config.py` resolves paths relative to the script location and expects a `code/` + `data/` + `outputs/` layout; if you run from a flat checkout, adjust `ROOT`/`DATA_DIR` in `config.py` (or place the modules under a `code/` folder) so the panel and outputs land where you expect.

## Reproducibility

Every stochastic component derives from a single global seed (`SEED` in `config.py`), and all structural constants live in `config.py` so a single edit propagates across the whole pipeline.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
