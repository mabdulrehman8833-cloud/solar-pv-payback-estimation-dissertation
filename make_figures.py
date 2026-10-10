"""Build every main-text and appendix figure for the dissertation.

Print-first: 300 dpi PNG, sized to the Word text column (6.3 in), with a
validated categorical palette plus marker/linestyle as secondary encoding so
the figures survive greyscale printing and colour-vision deficiency.

Run:  python make_figures.py
Out:  figures/fig1_stl_crawley.png ... figures/figA*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- house style
W_FULL, DPI = 6.3, 300
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

# Validated categorical slots (light mode). Colour follows the ENTITY, never rank.
C = {
    "RF (weather-aware)":      ("#2a78d6", "o", "-"),
    "XGB (weather-aware)":     ("#eb6834", "s", "-"),
    "SARIMA/Fourier":          ("#1baf7a", "^", "--"),
    "RF (weather-blind)":      ("#eda100", "D", ":"),
    "XGB (weather-blind)":     ("#e87ba4", "v", ":"),
    "PVGIS (vendor analogue)": ("#008300", "P", "-."),
}
SERIES = "#2a78d6"

REGIONS = {
    19: "Crawley, West Sussex", 10: "Bury St Edmunds, Suffolk",
    11: "Castle Donington, Derbyshire", 23: "Castleford, West Yorkshire",
    15: "Cramlington, Northumberland", 17: "Perth, Scotland",
}


def daily_panel():
    """Daily capacity factor and mean irradiance per region, gaps interpolated."""
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    out = {}
    for pid, g in df.groupby("pes_id"):
        g = g.sort_values("timestamp").set_index("timestamp")
        d = pd.DataFrame({
            "gen": g.generation_mw.resample("D").sum(),
            "cap": g.installed_mwp.resample("D").mean(),
            "swr": g.shortwave_radiation.resample("D").mean(),
        })
        d["cf"] = (d.gen / (d.cap * 24)).clip(0, 1)
        d["swr"] = d.swr.interpolate(limit_direction="both")
        d = d[(d.cap > 0)].dropna(subset=["cf", "swr"])
        d.index = d.index.tz_localize(None)
        out[pid] = d
    return out


PANEL = daily_panel()


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p)


# ============================================================ Figure 1 — STL
def fig1():
    d = PANEL[19]
    s = d.cf.asfreq("D").interpolate()
    res = STL(s, period=365, robust=False).fit()

    # All four panels share the observed series' vertical scale. Auto-scaling
    # the trend panel to its own tiny range would draw a near-zero trend as a
    # dramatic rising line -- the single most common way this figure misleads.
    span = float(s.max() - s.min())
    fig, ax = plt.subplots(4, 1, figsize=(W_FULL, 6.4), sharex=True)
    for a, (series, label) in zip(ax, [
            (s, "Observed"), (res.trend, "Trend"),
            (res.seasonal, "Seasonal (annual)"), (res.resid, "Remainder")]):
        a.plot(series.index, series.values, color=SERIES, lw=0.8)
        a.set_ylabel(label, color=INK)
        a.tick_params(length=0)
        mid = float(np.mean(series))
        a.set_ylim(mid - span * 0.60, mid + span * 0.60)
    ax[3].axhline(0, color=INK2, lw=0.7, ls="--")
    ax[0].set_title("Daily capacity factor, Crawley (West Sussex), 2023–2025",
                    color=INK, loc="left", pad=8)
    fig.align_ylabels(ax)
    save(fig, "fig1_stl_crawley.png")


# ================================== Figure 2 — model comparison (two panels)
def fig2():
    mc = pd.read_csv("model_comparison.csv")
    mc["key"] = mc.model + " (" + mc.info_set + ")"
    mc["key"] = mc.key.str.replace("SARIMA/Fourier (weather-blind)",
                                   "SARIMA/Fourier", regex=False)
    ladder = pd.read_csv("method_ladder_fullyear_summary.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_FULL, 3.1),
                                 gridspec_kw={"wspace": 0.34})

    # (a) R2 by horizon — lines, one per method
    order = ["RF (weather-aware)", "XGB (weather-aware)", "SARIMA/Fourier",
             "RF (weather-blind)", "XGB (weather-blind)"]
    for k in order:
        g = mc[mc.key == k].sort_values("h")
        col, mk, ls = C[k]
        a1.plot(g.h, g.R2, color=col, marker=mk, ls=ls, lw=1.8,
                ms=5, mec="white", mew=0.7, label=k)
    a1.set_xticks([1, 7, 30])
    a1.set_xlabel("Forecast horizon (days ahead)")
    a1.set_ylabel("Coefficient of determination (R²)")
    a1.set_title("(a) Skill by horizon", color=INK, loc="left")
    a1.set_ylim(0.4, 1.0)
    a1.set_xlim(-1, 33)
    # relief for the low-contrast slots: band labels rather than per-point
    a1.annotate("weather-aware", (17, 0.958), ha="center", va="center",
                color=INK2, fontsize=7.5)
    a1.annotate("weather-blind", (17, 0.448), ha="center", va="center",
                color=INK2, fontsize=7.5)

    # (b) mean absolute annual error by method — bars
    lab = {"RF (weather-aware)": "RF\nwx", "XGB (weather-aware)": "XGB\nwx",
           "SARIMA/Fourier": "SARIMA\nFourier", "RF (weather-blind)": "RF\nblind",
           "XGB (weather-blind)": "XGB\nblind",
           "PVGIS (vendor analogue)": "PVGIS"}
    ladder = ladder.sort_values("MAE")
    cols = [C[m][0] for m in ladder.method]
    bars = a2.bar(range(len(ladder)), ladder.MAE, color=cols,
                  width=0.68, edgecolor="white", linewidth=1.2)
    a2.set_xticks(range(len(ladder)))
    a2.set_xticklabels([lab[m] for m in ladder.method], fontsize=7.2)
    a2.set_ylabel("Mean absolute annual error (%)")
    a2.set_title("(b) Annual generation error", color=INK, loc="left")
    a2.set_ylim(0, ladder.MAE.max() * 1.22)
    for b, v in zip(bars, ladder.MAE):
        a2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", color=INK, fontsize=7.5)
    a2.grid(axis="x", visible=False)

    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13),
               handlelength=2.2, columnspacing=1.6, fontsize=7.6)
    save(fig, "fig2_model_comparison.png")


# ================================= Figure 3 — Crawley h=30 forecast vs actual
def fig3():
    p = pd.read_csv("backtest_preds_pes19.csv", parse_dates=["target"])
    p = p[p.h == 30].sort_values("target")
    fig, ax = plt.subplots(figsize=(W_FULL, 2.9))
    ax.plot(p.target, p.y_true, color=INK2, lw=0.9, label="Observed")
    ax.plot(p.target, p.y_pred, color=SERIES, lw=2.0, label="Forecast (h = 30)")
    ax.set_ylabel("Daily capacity factor")
    ax.set_title("Crawley: thirty-day-ahead forecast against observed outturn",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(bottom=0)
    save(fig, "fig3_crawley_h30.png")


# ================================== Figure 4 — PVGIS vs PV_Live by latitude
def fig4():
    pv = pd.read_csv("pvgis_estimates.csv")
    act = pd.DataFrame([{"pes_id": p, "actual": d.cf.mean() * 8760}
                        for p, d in PANEL.items()])
    m = pv.merge(act, on="pes_id").sort_values("lat")

    fig, ax = plt.subplots(figsize=(W_FULL, 3.2))
    ax.plot(m.lat, m.pvgis_kwh_per_kwp, color=C["PVGIS (vendor analogue)"][0],
            marker="P", ls="-.", lw=1.8, ms=7, mec="white", mew=0.8,
            label="PVGIS estimate")
    ax.plot(m.lat, m.actual, color=SERIES, marker="o", ls="-", lw=1.8,
            ms=7, mec="white", mew=0.8, label="Measured (PV_Live)")
    for i, (_, r) in enumerate(m.iterrows()):
        short = r.region.split(",")[0]
        ax.annotate(short, (r.lat, r.actual), textcoords="offset points",
                    xytext=(0, -20 if i % 2 else -13), ha="center",
                    color=INK2, fontsize=6.8)
    ax.set_xlim(m.lat.min() - 0.45, m.lat.max() + 0.45)
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Annual yield (kWh per kWp)")
    ax.set_title("Vendor estimate against measured outturn, by latitude",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="upper right")
    ax.set_ylim(760, 1100)
    save(fig, "fig4_pvgis_latitude.png")


# ============================ Figure 5 — efficiency drift vs residual error
def fig5():
    ml = pd.read_csv("ml_backtest_summary.csv")
    ml = ml[(ml.info_set == "weather-aware") & (ml.h == 1)]
    rows = []
    for pid, d in PANEL.items():
        y = d.copy()
        y["year"] = y.index.year
        e = y.groupby("year").apply(lambda g: g.cf.sum() / g.swr.sum(),
                                    include_groups=False)
        yrs = sorted(e.index)
        rows.append({"pes_id": pid, "region": REGIONS[pid],
                     "drift": 100 * (e[yrs[-1]] / e[yrs[0]] - 1)})
    drift = pd.DataFrame(rows)
    m = ml.merge(drift, on="pes_id")

    fig, ax = plt.subplots(figsize=(W_FULL, 3.2))
    labels = []
    for mdl, key in [("RF", "RF (weather-aware)"), ("XGB", "XGB (weather-aware)")]:
        g = m[m.model == mdl]
        col, mk, _ = C[key]
        # Spearman, not Pearson: on six points the rank coefficient is stable
        # under the run-to-run variation in the fitted errors, and Pearson is
        # not (it moves from -0.776 to -0.808, across p = 0.05). See
        # drift_correlation.py for the demonstration.
        rho, p = spearmanr(g.drift, g.annual_error_pct)
        r = np.corrcoef(g.drift, g.annual_error_pct)[0, 1]
        print(f"     {key}: rho = {rho:.3f} (p = {p:.3f}), Pearson r = {r:.3f}")
        labels.append(f"{mdl}  ρ = {rho:.2f}, p = {p:.3f}")
        ax.scatter(g.drift, g.annual_error_pct, color=col, marker=mk, s=55,
                   edgecolor="white", linewidth=0.9, zorder=3,
                   label=labels[-1])
        b, a = np.polyfit(g.drift, g.annual_error_pct, 1)
        xs = np.linspace(g.drift.min(), g.drift.max(), 20)
        ax.plot(xs, b * xs + a, color=col, lw=1.2, ls="--", alpha=0.75, zorder=2)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("Change in capacity factor per unit irradiance, 2023 to 2025 (%)")
    ax.set_ylabel("Residual annual error (%)")
    ax.set_title("Efficiency drift against weather-aware model error",
                 color=INK, loc="left", pad=8)
    # n is small enough that stating it on the figure is the honest thing to do
    ax.text(0.015, 0.05, "n = 6 regions", transform=ax.transAxes,
            color=INK2, fontsize=7.5)
    ax.legend(loc="upper right")
    save(fig, "fig5_drift_vs_error.png")


# ======================================================= Appendix figures
def figA_stl_all():
    fig, axes = plt.subplots(3, 2, figsize=(W_FULL, 6.2), sharex=True)
    for a, (pid, name) in zip(axes.ravel(), REGIONS.items()):
        s = PANEL[pid].cf.asfreq("D").interpolate()
        res = STL(s, period=365, robust=True).fit()
        a.plot(s.index, res.trend, color=SERIES, lw=1.4)
        a.set_title(name.split(",")[0], color=INK, loc="left", fontsize=8)
        a.tick_params(labelsize=7, length=0)
    fig.suptitle("STL trend component, all six regions", color=INK,
                 x=0.06, ha="left", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "figA1_stl_all_regions.png")


def figA_acf():
    fig, axes = plt.subplots(6, 2, figsize=(W_FULL, 9.4))
    for i, (pid, name) in enumerate(REGIONS.items()):
        s = PANEL[pid].cf.asfreq("D").interpolate()
        plot_acf(s, lags=40, ax=axes[i, 0], color=SERIES, vlines_kwargs={"colors": SERIES})
        plot_pacf(s, lags=40, ax=axes[i, 1], color=SERIES, method="ywm",
                  vlines_kwargs={"colors": SERIES})
        axes[i, 0].set_title(f"{name.split(',')[0]} — ACF", fontsize=8, loc="left", color=INK)
        axes[i, 1].set_title(f"{name.split(',')[0]} — PACF", fontsize=8, loc="left", color=INK)
        for a in axes[i]:
            a.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "figA2_acf_pacf.png")


def figA_horizon():
    ml = pd.read_csv("ml_backtest_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 4.2), sharey=True, sharex=True)
    for a, (pid, name) in zip(axes.ravel(), REGIONS.items()):
        g = ml[ml.pes_id == pid].copy()
        g["key"] = g.model + " (" + g.info_set + ")"
        for k, sub in g.groupby("key"):
            if k not in C:
                continue
            col, mk, ls = C[k]
            sub = sub.sort_values("h")
            a.plot(sub.h, sub.R2, color=col, marker=mk, ls=ls, lw=1.4, ms=4,
                   mec="white", mew=0.6, label=k)
        a.set_title(name.split(",")[0], fontsize=8, loc="left", color=INK)
        a.set_xticks([1, 7, 30])
    axes[0, 0].set_ylabel("R²")
    axes[1, 0].set_ylabel("R²")
    for a in axes[1]:
        a.set_xlabel("Horizon (days)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "figA3_horizon_curves.png")



# ================== Figure 6 — irradiance trend and window sensitivity
def fig6():
    """Is the fitted brightening trend real, and what does it cost the estimate?

    Panel (a) shows why a ten-year record cannot answer the first question;
    panel (b) shows that the answer matters anyway.
    """
    W = pd.read_csv("longrun_weather.csv", parse_dates=["timestamp"])
    W["year"] = W.timestamp.dt.year
    a = W.groupby("year").shortwave_radiation.mean()
    recent = pd.read_csv("solar_weather_dataset.csv",
                         usecols=["shortwave_radiation"]).mean().iloc[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_FULL, 3.1),
                                 gridspec_kw={"wspace": 0.30})

    a1.plot(a.index, a.values, color=SERIES, marker="o", ls="none",
            ms=6, mec="white", mew=0.8, zorder=3, label="Annual mean")
    xs = np.array(a.index, dtype=float)
    for sub, col, ls, lab in [
            (a, SERIES, "-", "Fit, all years  (+7.1%/decade, p = 0.06)"),
            (a.drop(2022), "#eb6834", "--",
             "Fit, excluding 2022  (+3.8%/decade, p = 0.28)")]:
        b, c = np.polyfit(np.array(sub.index, dtype=float), sub.values, 1)
        a1.plot(xs, b * xs + c, color=col, ls=ls, lw=1.8, label=lab)
    a1.axhline(recent, color=INK2, lw=1.0, ls=":")
    a1.annotate("2023–25 mean", (2013.2, recent), textcoords="offset points",
                xytext=(0, 4), color=INK2, fontsize=7.2)
    a1.set_xlabel("Year")
    a1.set_ylabel("Mean shortwave irradiance (W m$^{-2}$)")
    a1.set_title("(a) Is there a trend?", color=INK, loc="left")
    a1.set_xticks(range(2013, 2023, 2))

    # (b) what the window choice costs the annual estimate
    win = [("2 yr\n2023–24", -5.28), ("10 yr\n2013–22", -5.77),
           ("13 yr\n2010–22", -6.56), ("28 yr\n1995–22", -7.09)]
    cols = [C["RF (weather-aware)"][0] if lab.startswith("10") else "#b9c9dd"
            for lab, _ in win]
    bars = a2.bar(range(4), [v for _, v in win], color=cols, width=0.66,
                  edgecolor="white", linewidth=1.2)
    a2.set_xticks(range(4))
    a2.set_xticklabels([lab for lab, _ in win], fontsize=7.2)
    a2.set_ylabel("Annual estimate error (%)")
    a2.set_title("(b) What the window choice costs", color=INK, loc="left")
    a2.set_ylim(-8.4, 0)
    for b, (_, v) in zip(bars, win):
        a2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, -11),
                    ha="center", color=INK, fontsize=7.4)
    a2.grid(axis="x", visible=False)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.20),
               handlelength=2.2, columnspacing=1.6, fontsize=7.4)
    save(fig, "fig6_irradiance_trend.png")


if __name__ == "__main__":
    print("building figures...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    figA_stl_all(); figA_acf(); figA_horizon()
    print("done.")
