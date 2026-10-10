"""
================================================================================
THE HEADLINE FINANCIAL RESULT

Every estimation method produces an annual generation figure. Push each one
through the same payback calculator and the spread in PAYBACK YEARS is what a
homeowner actually experiences when a vendor's estimate is wrong.

This is the step most of the solar-forecasting literature omits: papers stop at
RMSE. Translating generation error into a financial decision error is the
distinctive contribution.

--------------------------------------------------------------------------------
WHAT IT REPORTS
--------------------------------------------------------------------------------
For each method, per region:
    predicted annual kWh  ->  predicted payback (what the customer is told)
    actual annual kWh     ->  actual payback    (what they get)
    payback error in YEARS = the consequence of the estimation error

--------------------------------------------------------------------------------
EVERY PRICE BELOW IS A PLACEHOLDER
--------------------------------------------------------------------------------
Replace with current cited figures before reporting anything. ~GBP 1,565/kWp for
England was a defensible anchor at the time of writing; source your own and put
the full assumption set in a table in the method chapter. Payback is far more
sensitive to these than to which model you used -- which is itself worth saying.
================================================================================
"""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

# ---- assumptions live in economics.py so both scripts stay in sync ----------
from economics import (SYSTEM_KWP, IMPORT_PRICE, SEG_RATE, SELF_CONSUMPTION,
                       DEGRADATION, INVERTER_REPLACE, DISCOUNT_RATE,
                       HORIZON_YEARS, capital_cost, assumptions_table)

SOURCE = "method_ladder_fullyear.csv"   # or climatology_longrun.csv


def payback_years(annual_kwh_y1, capital):
    """Simple payback, linearly interpolated within the crossing year."""
    cum, prev = -capital, -capital
    for t in range(1, HORIZON_YEARS + 1):
        gen    = annual_kwh_y1 * (1 - DEGRADATION) ** (t - 1)
        saving = (gen * SELF_CONSUMPTION * IMPORT_PRICE
                  + gen * (1 - SELF_CONSUMPTION) * SEG_RATE)
        cost   = INVERTER_REPLACE[1] if t == INVERTER_REPLACE[0] else 0.0
        prev, cum = cum, cum + saving - cost
        if cum >= 0:
            return t - 1 + (0 - prev) / (cum - prev)
    return np.nan          # not reached within the horizon


if __name__ == "__main__":
    L = pd.read_csv(SOURCE)
    yield_col = "yield" if "yield" in L.columns else "specific_yield_kwh_per_kwp"

    capital = capital_cost(SYSTEM_KWP)
    actual = (L[L["method"] == "Actual (PV_Live)"]
                .set_index("region")[yield_col])

    rows = []
    for _, r in L.iterrows():
        if r["method"] == "Actual (PV_Live)":
            continue
        pred_kwh = r[yield_col] * SYSTEM_KWP
        act_kwh  = actual[r["region"]] * SYSTEM_KWP
        pb_pred  = payback_years(pred_kwh, capital)
        pb_act   = payback_years(act_kwh,  capital)
        rows.append({
            "region": r["region"], "method": r["method"],
            "predicted_kwh": pred_kwh, "actual_kwh": act_kwh,
            "gen_error_pct": 100 * (pred_kwh - act_kwh) / act_kwh,
            "payback_predicted": pb_pred, "payback_actual": pb_act,
            "payback_error_years": pb_pred - pb_act,
        })

    P = pd.DataFrame(rows)
    P.to_csv("payback_by_method.csv", index=False)

    print("=" * 100)
    print(f"PAYBACK BY ESTIMATION METHOD — {SYSTEM_KWP:.0f} kWp, "
          f"GBP {capital:,.0f} capital, {IMPORT_PRICE:.2f}/kWh import, "
          f"{SELF_CONSUMPTION:.0%} self-consumption")
    print("=" * 100)
    print(assumptions_table().to_string(index=False))
    print()

    summ = (P.groupby("method")
              .agg(gen_error_pct=("gen_error_pct", "mean"),
                   payback_predicted=("payback_predicted", "mean"),
                   payback_actual=("payback_actual", "mean"),
                   payback_error_years=("payback_error_years", "mean"),
                   worst_region_error=("payback_error_years",
                                       lambda s: s.abs().max()))
              .sort_values("payback_error_years", key=abs))
    print(summ.to_string(float_format=lambda v: f"{v:+.2f}"))

    print("\n" + "=" * 100)
    print("PER-REGION PAYBACK ERROR (years)")
    print("=" * 100)
    piv = P.pivot_table(index="method", columns="region",
                        values="payback_error_years")
    print(piv.to_string(float_format=lambda v: f"{v:+.2f}"))

    print("""
HOW TO READ THIS
  A POSITIVE payback error means the method predicts a LONGER payback than the
  customer actually gets -- conservative, and harmless.
  A NEGATIVE error means it predicts a SHORTER payback than reality -- the
  customer is told the system pays for itself sooner than it does. That is the
  consumer-detriment case your dissertation is about.

SENSITIVITY
  Payback is more sensitive to electricity price and self-consumption than to
  which estimation method is used. Report the grid from postcode_predict.py
  alongside this table so the reader can see both effects, and say which
  dominates -- that comparison is itself a finding.
""")
