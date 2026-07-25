"""
run_all.py
==========
Runs the full compendium pipeline end-to-end, in dependency order, from a clean
checkout. Every table in outputs/tables and every figure in outputs/figures is
regenerated from the (disclosed-synthetic) data with no manual editing.

Usage:
    cd code && python run_all.py
"""

import importlib
import time

STEPS = [
    ("data_prep", "Build synthetic divisional MR demand panel (Route B)"),
    ("forecast", "Forecast births (5 models) and derive baseline demand"),
    ("gan_scenarios", "Train conditional GAN and generate demand ensembles"),
    ("seasonality", "Segment demand seasons (classical + ensemble upgrade)"),
    ("model", "Solve deterministic + two-stage stochastic MILP"),
    ("stress_test", "Run seasonal resilience stress tests"),
    ("sensitivity", "Taguchi/LHS design, MLR metamodels, SHAP GSA"),
    ("make_figures", "Regenerate all figures from real outputs"),
]


def main():
    t0 = time.time()
    for mod_name, desc in STEPS:
        print("\n" + "=" * 72)
        print(f">>> {mod_name}: {desc}")
        print("=" * 72)
        t = time.time()
        mod = importlib.import_module(mod_name)
        mod.main()
        print(f"[{mod_name} done in {time.time() - t:.1f}s]")
    print("\nPipeline complete in %.1fs." % (time.time() - t0))


if __name__ == "__main__":
    main()
