"""
make_figures.py
===============
Regenerates every manuscript figure from the REAL pipeline outputs in
outputs/tables/ and data/. No figure contains a hand-typed number.

Figures (outputs/figures/*.png)
  fig01_disruption_timeline   national MR demand with 2024-25 stockout & 2026 surge
  fig02_forecast_r2           R2 heatmap, division x model
  fig03_forecast_series       Dhaka births: history vs forecast
  fig04_demand_dhaka          Dhaka forecasted MR1/MR2 demand
  fig05_gan_ensemble          GAN demand ensemble fan chart (Dhaka)
  fig06_season_counts         detected seasons per division (classical vs ensemble)
  fig07_cost_pie              optimal cost-component shares (Z)
  fig08_production_by_season  seasonal production across periods
  fig09_vss                   stochastic RP vs EEV and VSS
  fig10_service_levels        stress-test service levels
  fig11_shap_z                SHAP share of inputs in Z variability
  fig12_shap_heatmap          SHAP shares, input x output
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
T = C.TAB_DIR
F = C.FIG_DIR


def fig01():
    df = pd.read_csv(C.PANEL_CSV)
    nat = (df.groupby(["year", "month"])[["doses_baseline", "doses_disrupted"]]
           .sum().reset_index())
    nat["date"] = pd.to_datetime(dict(year=nat.year, month=nat.month, day=1))
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.plot(nat.date, nat.doses_baseline / 1e6, label="Baseline demand", lw=1.6)
    ax.plot(nat.date, nat.doses_disrupted / 1e6, label="With disruptions",
            lw=1.6, color="crimson")
    ax.axvspan(pd.Timestamp("2024-07-01"), pd.Timestamp("2025-12-31"),
               color="grey", alpha=0.15, label="2024-25 MR stockout")
    ax.axvspan(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-30"),
               color="orange", alpha=0.20, label="2026 outbreak surge")
    ax.set_ylabel("National MR demand (million doses)")
    ax.set_title("Synthetic national MR demand with real disruption overlays")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{F}/fig01_disruption_timeline.png"); plt.close(fig)


def fig02():
    r2 = pd.read_csv(f"{T}/table_forecast_r2.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow((r2.values * 100).clip(-50, 100), cmap="RdYlGn",
                   aspect="auto", vmin=-50, vmax=100)
    ax.set_xticks(range(len(r2.columns))); ax.set_xticklabels(r2.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(r2.index))); ax.set_yticklabels(r2.index)
    for i in range(len(r2.index)):
        for j in range(len(r2.columns)):
            ax.text(j, i, f"{r2.values[i, j]*100:.0f}", ha="center", va="center",
                    fontsize=7)
    ax.set_title("Out-of-sample $R^2$ (%) on monthly births, test = 2024")
    fig.colorbar(im, ax=ax, shrink=0.8, label="$R^2$ (%)")
    fig.tight_layout(); fig.savefig(f"{F}/fig02_forecast_r2.png"); plt.close(fig)


def fig03():
    df = pd.read_csv(C.PANEL_CSV)
    b = (df[df.division == "Dhaka"].groupby(["year", "month"])["births"].first()
         .reset_index())
    b["date"] = pd.to_datetime(dict(year=b.year, month=b.month, day=1))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hist = b[b.year <= 2024]; fut = b[b.year >= 2025]
    ax.plot(hist.date, hist.births / 1e3, label="History (2020-2024)", lw=1.6)
    ax.plot(fut.date, fut.births / 1e3, label="Forecast basis (2025-2027)",
            lw=1.6, ls="--", color="darkorange")
    ax.set_ylabel("Monthly births (thousands)")
    ax.set_title("Dhaka division: synthetic birth backbone")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig03_forecast_series.png"); plt.close(fig)


def fig04():
    d = pd.read_csv(f"{T}/forecast_demand_2026_2028.csv")
    d = d[d.division == "Dhaka"]
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for dose in C.MR_DOSES:
        s = d[d.dose == dose].sort_values("date")
        ax.plot(s.date, s.doses / 1e3, marker="o", ms=3, label=dose)
    ax.set_ylabel("Monthly demand (thousand doses)")
    ax.set_title("Dhaka division: forecasted MR demand, 2026-2028")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{F}/fig04_demand_dhaka.png"); plt.close(fig)


def fig05():
    scen = pd.read_csv(f"{T}/gan_scenarios.csv")
    s = scen[scen.division == "Dhaka"]
    piv = s.pivot_table(index="scenario", columns="month", values="demand")
    p10, p50, p90 = (piv.quantile(q) for q in (0.1, 0.5, 0.9))
    months = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.fill_between(months, p10 / 1e3, p90 / 1e3, alpha=0.25, color="steelblue",
                    label="p10-p90 ensemble")
    ax.plot(months, p50 / 1e3, color="navy", lw=1.8, label="median")
    for sc in piv.index[:8]:
        ax.plot(months, piv.loc[sc] / 1e3, color="grey", alpha=0.35, lw=0.7)
    ax.set_xlabel("Month"); ax.set_ylabel("Demand (thousand doses)")
    ax.set_title("GAN demand ensemble, Dhaka (uncertainty-aware scenarios)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{F}/fig05_gan_ensemble.png"); plt.close(fig)


def fig06():
    sc = pd.read_csv(f"{T}/table_season_counts.csv", index_col=0)
    x = np.arange(len(sc.index)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, sc["MR1"], w, label="Classical (MR1)")
    ax.bar(x + w / 2, sc["seasons_ensemble"], w, label="Ensemble upgrade")
    ax.set_xticks(x); ax.set_xticklabels(sc.index, rotation=30, ha="right")
    ax.set_ylabel("Detected demand seasons")
    ax.set_title("Season segmentation per division")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{F}/fig06_season_counts.png"); plt.close(fig)


def fig07():
    cb = pd.read_csv(f"{T}/opt_cost_breakdown.csv")
    cb = cb[cb.component != "Z"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.pie(cb.share_pct, labels=cb.component, autopct="%1.1f%%",
           colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], startangle=90)
    ax.set_title("Optimal cost-component shares of $Z$")
    fig.tight_layout(); fig.savefig(f"{F}/fig07_cost_pie.png"); plt.close(fig)


def fig08():
    p = pd.read_csv(f"{T}/opt_production_by_season.csv")
    piv = p.pivot(index="period", columns="prod_season", values="doses").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(piv))
    for col in piv.columns:
        ax.barh(piv.index.astype(str), piv[col] / 1e6, left=bottom,
                label=f"Season {col}")
        bottom += (piv[col] / 1e6).values
    ax.set_xlabel("Production (million doses)"); ax.set_ylabel("Planning period")
    ax.set_title("Seasonal production across periods")
    ax.legend(fontsize=8, ncol=4); fig.tight_layout()
    fig.savefig(f"{F}/fig08_production_by_season.png"); plt.close(fig)


def fig09():
    s = pd.read_csv(f"{T}/opt_stochastic_summary.csv")
    rp = s.loc[s.quantity.str.startswith("Z_stochastic"), "value_usd"].iloc[0]
    eev = s.loc[s.quantity.str.startswith("Z_expected"), "value_usd"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(["Stochastic (RP)", "Mean-value (EEV)"], [rp / 1e6, eev / 1e6],
                  color=["#55A868", "#C44E52"])
    ax.set_ylabel("Expected total cost (million USD)")
    ax.set_title("Two-stage stochastic vs mean-value solution")
    vss = eev - rp
    ax.text(0.5, max(rp, eev) / 1e6 * 0.5, f"VSS = {vss/1e6:.2f}M\n({100*vss/rp:.1f}%)",
            ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{F}/fig09_vss.png"); plt.close(fig)


def fig10():
    s = pd.read_csv(f"{T}/table_stress_service_levels.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(s.scenario, s.service_level_pct, color=["#4C72B0", "#DD8452"])
    ax.axhline(100, ls="--", color="grey", lw=1)
    ax.set_ylim(90, 102); ax.set_ylabel("Service level (%)")
    ax.set_title("Resilience: service level under disruption")
    for b, v in zip(bars, s.service_level_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center",
                fontsize=9)
    plt.xticks(rotation=8, fontsize=8); fig.tight_layout()
    fig.savefig(f"{F}/fig10_service_levels.png"); plt.close(fig)


def fig11():
    sh = pd.read_csv(f"{T}/table_shap_shares.csv", index_col=0)
    z = sh.loc["Z"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.pie(z.values, labels=z.index, autopct="%1.1f%%", startangle=90,
           colors=["#4C72B0", "#DD8452", "#8C8C8C", "#E1B000", "#55A868"])
    ax.set_title("SHAP share of inputs in total-cost ($Z$) variability")
    fig.tight_layout(); fig.savefig(f"{F}/fig11_shap_z.png"); plt.close(fig)


def fig12():
    sh = pd.read_csv(f"{T}/table_shap_shares.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    im = ax.imshow(sh.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(sh.columns))); ax.set_xticklabels(sh.columns)
    ax.set_yticks(range(len(sh.index))); ax.set_yticklabels(sh.index)
    for i in range(len(sh.index)):
        for j in range(len(sh.columns)):
            ax.text(j, i, f"{sh.values[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_title("SHAP-based input shares in output variability")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share")
    fig.tight_layout(); fig.savefig(f"{F}/fig12_shap_heatmap.png"); plt.close(fig)


def fig13():
    df = pd.read_csv(C.PANEL_CSV)
    g = (df.groupby("division")["doses_baseline"].sum()
         .reindex(C.DIVISIONS) / 1e6)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(g.index, g.values, color="#4C72B0")
    ax.set_ylabel("Total baseline demand (million doses)")
    ax.set_title("Total baseline MR demand by division (2020 to 2028)")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(f"{F}/fig13_division_demand.png"); plt.close(fig)


def fig14():
    s = pd.read_csv(f"{T}/opt_shipment_showcase.csv")
    piv = (s.groupby(["division", "node"])["doses"].sum().unstack(fill_value=0)
           .reindex(C.DIVISIONS))
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    bottom = np.zeros(len(piv))
    colors = {"EPI_Central": "#DD8452", "Regional_Store": "#55A868"}
    for node in piv.columns:
        ax.bar(piv.index, piv[node] / 1e3, bottom=bottom,
               label=node.replace("_", " "), color=colors.get(node))
        bottom += (piv[node] / 1e3).values
    ax.set_ylabel("MR1 doses shipped, period 1 (thousands)")
    ax.set_title("Optimal node to division allocation, MR1 first period")
    ax.legend(); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(f"{F}/fig14_allocation.png"); plt.close(fig)


def fig15():
    tag = pd.read_csv(f"{T}/table_taguchi_l25.csv")
    params = ["de", "p", "pm", "tcc", "hm_hcc"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for pnm in params:
        me = tag.groupby(pnm)["Z"].mean() / 1e6
        ax.plot(me.index, me.values, marker="o", ms=4, label=pnm)
    ax.set_xlabel("Parameter level (percent change)")
    ax.set_ylabel("Mean total cost Z (million USD)")
    ax.set_title("Taguchi main effects on total cost")
    ax.legend(title="Parameter", fontsize=8); fig.tight_layout()
    fig.savefig(f"{F}/fig15_taguchi_main_effects.png"); plt.close(fig)


def fig16():
    r2 = pd.read_csv(f"{T}/table_forecast_r2.csv", index_col=0)
    m = (r2.mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in m.values]
    ax.barh(m.index, m.values, color=colors)
    ax.set_xlabel("Mean out of sample R squared (percent)")
    ax.set_title("Mean forecasting accuracy by model")
    for i, v in enumerate(m.values):
        ax.text(v, i, f" {v:.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig16_mean_accuracy.png"); plt.close(fig)


def fig17():
    scen = pd.read_csv(f"{T}/gan_scenarios.csv")
    nat = scen.groupby(["scenario", "month"])["demand"].sum().reset_index()
    piv = nat.pivot(index="scenario", columns="month", values="demand")
    months = piv.columns.values
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for sc in piv.index:
        ax.plot(months, piv.loc[sc] / 1e3, color="steelblue", alpha=0.25, lw=0.7)
    ax.plot(months, piv.median() / 1e3, color="navy", lw=2, label="median")
    ax.set_xlabel("Month"); ax.set_ylabel("National demand (thousand doses)")
    ax.set_title("GAN demand scenarios aggregated to national level")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{F}/fig17_national_scenarios.png"); plt.close(fig)


def main():
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
               fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17]:
        fn()
        print("wrote", fn.__name__)
    print("All figures written to", F)


if __name__ == "__main__":
    main()
