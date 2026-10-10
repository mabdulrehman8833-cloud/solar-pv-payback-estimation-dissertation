import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE   = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8880"
SERIES = {  # fixed slot order, validated (light mode)
    "RF · weather-aware":     "#2a78d6",
    "XGB · weather-aware":    "#eb6834",
    "SARIMA/Fourier":         "#1baf7a",
    "RF · weather-blind":     "#eda100",
    "XGB · weather-blind":    "#e87ba4",
}

LABEL_DY = {"RF · weather-aware": 7, "XGB · weather-aware": -7,
            "SARIMA/Fourier": 0,
            "RF · weather-blind": 7, "XGB · weather-blind": -7}

c = pd.read_csv("model_comparison.csv")
c["key"] = np.where(c["model"].str.contains("SARIMA"), "SARIMA/Fourier",
                    c["model"] + " · " + c["info_set"])

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)

# ---- Panel A: explained variance vs horizon --------------------------------
ax = axes[0]
ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for label, colour in SERIES.items():
    g = c[c["key"] == label].sort_values("h")
    ax.plot(g["h"], g["R2"], color=colour, lw=2, marker="o", ms=8,
            mec=SURFACE, mew=2, zorder=3, label=label)
    dy = LABEL_DY[label]
    ax.annotate(f"{g['R2'].iloc[-1]:.2f}", (g["h"].iloc[-1], g["R2"].iloc[-1]),
                xytext=(9, dy), textcoords="offset points", va="center",
                fontsize=9, color=INK, fontweight="medium")
ax.set_xticks([1, 7, 30]); ax.set_xlim(-1, 37); ax.set_ylim(0.4, 1.0)
ax.set_xlabel("Forecast horizon (days)", color=INK2, fontsize=10)
ax.set_ylabel("R²  (out-of-sample)", color=INK2, fontsize=10)
ax.set_title("Explained variance by horizon", color=INK, fontsize=11.5,
             fontweight="semibold", loc="left", pad=10)

# ---- Panel B: annual generation error at h=30 ------------------------------
ax = axes[1]
ax.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0)
ax.set_axisbelow(True)
g30 = (c[c["h"] == 30].set_index("key").reindex(SERIES.keys()))
ypos = np.arange(len(g30))[::-1]
ax.barh(ypos, g30["annual_error_pct"], height=0.6,
        color=[SERIES[k] for k in g30.index], zorder=3)
ax.axvline(0, color=MUTED, lw=1)
for yv, val, name in zip(ypos, g30["annual_error_pct"], g30.index):
    off = 0.35 if val >= 0 else -0.35
    ax.text(val + off, yv, f"{val:+.1f}%", va="center",
            ha="left" if val >= 0 else "right", fontsize=9.5,
            color=INK, fontweight="medium")
ax.set_yticks(ypos); ax.set_yticklabels(g30.index, fontsize=9.5, color=INK)
ax.set_xlim(-15, 4)
ax.set_xlabel("Error in total annual generation (%)", color=INK2, fontsize=10)
ax.set_title("Annual generation error — the input to payback", color=INK,
             fontsize=11.5, fontweight="semibold", loc="left", pad=10)

axes[0].legend(frameon=False, fontsize=9, loc="upper right",
               bbox_to_anchor=(0.99, 0.82), labelcolor=INK2,
               handlelength=1.6, borderpad=0)
fig.suptitle("Weather information, not model family, drives accuracy",
             fontsize=13.5, fontweight="semibold", color=INK, x=0.008, ha="left", y=0.99)
fig.text(0.008, 0.925,
         "Mean across 6 GB regions · identical walk-forward origins · "
         "weather-aware = recorded weather on the target date (upper bound)",
         fontsize=9, color=INK2, ha="left")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("model_comparison_figure.png", dpi=160, facecolor=SURFACE)
print("saved model_comparison_figure.png")
