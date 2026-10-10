"""
Fetch PVGIS annual/monthly generation estimates for each PES region centroid.

RUN THIS IN COLAB — the JRC API is not reachable from every environment.
PVGIS v5.2 PVcalc is the closest public analogue to a vendor quote: it uses a
typical-meteorological-year irradiance database rather than actual weather,
which is exactly the assumption this dissertation is testing.

NOTE: this has not been executed against the live API from here, so it prints
the raw JSON keys if the expected fields are missing — check the first response
and adjust the field paths if PVGIS has changed its schema.
"""
import json, time, urllib.request
import pandas as pd

REGIONS = {
    17: ("Perth, Scotland",              56.42, -3.46),
    15: ("Cramlington, Northumberland",  55.09, -1.59),
    23: ("Castleford, West Yorkshire",   53.72, -1.37),
    11: ("Castle Donington, Derbyshire", 52.83, -1.31),
    10: ("Bury St Edmunds, Suffolk",     52.25,  0.73),
    19: ("Crawley, West Sussex",         51.12, -0.19),
}

PEAK_KWP   = 4.0     # typical UK domestic system
SYSTEM_LOSS = 14.0   # PVGIS default system loss (%)
TILT, AZIMUTH = 35, 0   # 35 deg pitch, due south

rows = []
for pes_id, (name, lat, lon) in REGIONS.items():
    url = ("https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
           f"?lat={lat}&lon={lon}&peakpower={PEAK_KWP}&loss={SYSTEM_LOSS}"
           f"&angle={TILT}&aspect={AZIMUTH}&outputformat=json")
    raw = json.loads(urllib.request.urlopen(url, timeout=60).read().decode())

    try:
        totals = raw["outputs"]["totals"]["fixed"]
        e_y = totals["E_y"]                       # kWh/year
        monthly = {m["month"]: m["E_m"] for m in raw["outputs"]["monthly"]["fixed"]}
    except (KeyError, TypeError):
        print("Unexpected schema — inspect these keys and adjust:")
        print(json.dumps(raw, indent=2)[:1500])
        raise

    rows.append({"pes_id": pes_id, "region": name, "lat": lat, "lon": lon,
                 "pvgis_kwh_year": e_y,
                 "pvgis_kwh_per_kwp": e_y / PEAK_KWP,
                 **{f"m{k:02d}": v for k, v in monthly.items()}})
    print(f"{name:32s} {e_y:8.1f} kWh/yr  ({e_y/PEAK_KWP:6.1f} kWh/kWp)")
    time.sleep(1)

pd.DataFrame(rows).to_csv("pvgis_estimates.csv", index=False)
print("\nSaved pvgis_estimates.csv")
