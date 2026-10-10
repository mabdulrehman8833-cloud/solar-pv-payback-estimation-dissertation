import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8880"
C_PVGIS="#2a78d6"; C_ACT="#eb6834"

pv=pd.read_csv("pvgis_estimates.csv"); est=pd.read_csv("annual_generation_estimates.csv")
act=est[est.method=="Actual (PV_Live)"].set_index("pes_id")["specific_yield_kwh_per_kwp"]
pv=pv.set_index("pes_id"); pv["actual"]=act
pv=pv.sort_values("lat")
short=[r.split(",")[0] for r in pv["region"]]

fig,ax=plt.subplots(figsize=(9.5,5.2),facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(axis="y",color="#e6e5e0",lw=0.8,zorder=0); ax.set_axisbelow(True)
ax.plot(pv["lat"],pv["pvgis_kwh_per_kwp"],color=C_PVGIS,lw=2,marker="o",ms=9,
        mec=SURFACE,mew=2,zorder=3,label="PVGIS  (4 kWp, 35°, due south)")
ax.plot(pv["lat"],pv["actual"],color=C_ACT,lw=2,marker="o",ms=9,
        mec=SURFACE,mew=2,zorder=3,label="PV_Live actual  (regional fleet)")
for x,yp,ya,nm in zip(pv["lat"],pv["pvgis_kwh_per_kwp"],pv["actual"],short):
    ax.annotate(nm,(x,min(yp,ya)),xytext=(0,-20),textcoords="offset points",
                ha="center",fontsize=8,color=INK2,rotation=0)
ax.invert_xaxis()
ax.set_xlabel("Latitude (°N)  —  south to the right",color=INK2,fontsize=10)
ax.set_ylabel("Specific yield (kWh/kWp/yr)",color=INK2,fontsize=10)
ax.set_ylim(830,1080)
ax.spines[["top","right"]].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
ax.tick_params(colors=INK2,labelsize=9)
ax.legend(frameon=False,fontsize=9.5,loc="upper left",labelcolor=INK2)
fig.suptitle("PVGIS shows a strong north–south gradient; the PV_Live fleet barely does",
             color=INK,fontsize=12.5,fontweight="semibold",x=0.012,ha="left",y=0.985)
fig.text(0.012,0.925,"16.1% range across regions vs 6.0% — and Perth out-yields the Midlands, "
         "which is not physically plausible",fontsize=9,color=INK2)
plt.tight_layout(rect=[0,0,1,0.895])
plt.savefig("pvgis_gradient.png",dpi=160,facecolor=SURFACE)
print("saved")
