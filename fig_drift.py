import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8880"; C="#2a78d6"

df=pd.read_csv("solar_weather_dataset.csv",parse_dates=["timestamp"]); df["year"]=df.timestamp.dt.year
d=df[df.is_daylight==1]
g=d.groupby(["region_name","year"]).agg(cf=("capacity_factor","mean"),sw=("shortwave_radiation","mean"))
g["eff"]=g.cf/g.sw; p=g["eff"].unstack(); drift=100*(p[2025]/p[2023]-1)

lad=pd.read_csv("method_ladder.csv")
err=lad[lad.method=="RF (weather-aware)"].set_index("region")["error_pct"]
j=pd.DataFrame({"drift":drift,"err":err}).dropna()
r=j.drift.corr(j.err)

fig,ax=plt.subplots(figsize=(8.6,5.4),facecolor=SURFACE); ax.set_facecolor(SURFACE)
ax.grid(color="#e6e5e0",lw=0.8,zorder=0); ax.set_axisbelow(True)
ax.axhline(0,color=MUTED,lw=1,zorder=1); ax.axvline(0,color=MUTED,lw=1,zorder=1)

# fitted line
b,a=np.polyfit(j.drift,j.err,1); xs=np.linspace(j.drift.min()-0.8,j.drift.max()+0.8,50)
ax.plot(xs,a+b*xs,color=MUTED,lw=1.6,ls="--",zorder=2)
ax.scatter(j.drift,j.err,s=110,color=C,edgecolor=SURFACE,linewidth=2,zorder=4)
OFF={"Castleford, West Yorkshire":(-16,-6),"Crawley, West Sussex":(14,4),
     "Cramlington, Northumberland":(0,-20)}
for reg,row in j.iterrows():
    dx,dy=OFF.get(reg,(0,13))
    ha="right" if dx<0 else ("left" if dx>0 else "center")
    ax.annotate(reg.split(",")[0],(row.drift,row.err),xytext=(dx,dy),
                textcoords="offset points",ha=ha,fontsize=9,color=INK2)

ax.set_xlabel("Drift in capacity factor per unit irradiance, 2023 → 2025 (%)",color=INK2,fontsize=10)
ax.set_ylabel("RF weather-aware annual error (%)",color=INK2,fontsize=10)
ax.set_ylim(-6.5,5.8); ax.set_xlim(-5.6,3.4)
ax.spines[["top","right"]].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
ax.tick_params(colors=INK2,labelsize=9)
ax.text(0.985,0.045,f"r = {r:+.2f}",transform=ax.transAxes,ha="right",
        fontsize=11,color=INK,fontweight="semibold")

fig.suptitle("The model's remaining error tracks drift in the data, not model weakness",
             color=INK,fontsize=12.5,fontweight="semibold",x=0.012,ha="left",y=0.985)
fig.text(0.012,0.925,"Where the fleet's apparent efficiency fell between training and test years, "
         "the model over-predicts — and vice versa",fontsize=9,color=INK2)
plt.tight_layout(rect=[0,0,1,0.895])
plt.savefig("efficiency_drift.png",dpi=160,facecolor=SURFACE); print("saved")
