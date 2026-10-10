"""
================================================================================
FINANCIAL ASSUMPTIONS — single source of truth
================================================================================

Both payback_comparison.py and postcode_predict.py import from here, so the two
cannot drift apart. Every payback figure in the dissertation depends on these
values, so this file is effectively a results table: reproduce it verbatim in the
method chapter, with the SOURCE column filled in.

FILL IN EVERY 'source' STRING. A number without a citation cannot go in a
dissertation. The placeholders below are anchors to replace, not values to quote.
================================================================================
"""

# ---- provenance -------------------------------------------------------------
# A dissertation needs FIXED, DATED assumptions, not live ones: if prices updated
# themselves, your reported payback would change between writing and marking, and
# an examiner re-running the code would get different numbers from your text.
# So these values are frozen and stamped. AS_OF records when they were valid;
# check_freshness() warns if they have gone stale.
#
# (A deployed consumer tool would refresh them; a dissertation must not.)
AS_OF = "2026-09-15"          # date these values were checked
STALE_AFTER_DAYS = 90         # Ofgem's cap changes quarterly


def check_freshness(verbose=True):
    """Warn if the assumption set is older than one price-cap period."""
    from datetime import date
    y, m, d = (int(x) for x in AS_OF.split("-"))
    age = (date.today() - date(y, m, d)).days
    stale = age > STALE_AFTER_DAYS
    if verbose:
        msg = (f"assumptions checked {AS_OF} ({age} days ago)"
               + ("  -- STALE: re-verify the Ofgem cap and SEG rates"
                  if stale else "  -- current"))
        print(msg)
    return not stale


# ---- climatology window -----------------------------------------------------
# The "typical year" is built from these dates. THIS IS A MODELLING CHOICE and it
# moves the annual estimate by up to 1.8 percentage points, so it must be stated
# in the method chapter and used consistently everywhere.
#
# Measured accuracy vs the 3-year mean actual, by window:
#     2023-2024 (2 yr)    -5.28%   (overlaps training -- not usable)
#     2013-2022 (10 yr)   -5.77%   <-- CHOSEN
#     2010-2022 (13 yr)   -6.56%
#     1995-2022 (28 yr)   -7.09%
# Longer windows read lower, consistent with an upward tendency in irradiance.
# 2013-2022 is preferred: recent enough to reflect current conditions, long
# enough to average out interannual variability, and it ends before the PV_Live
# period so the evaluation is not circular.
#
# IMPORTANT: end the window BEFORE 2023 so the evaluation against the 2023-25
# PV_Live actuals is not circular. 2013-2022 gives a 10-year window that is both
# recent and clean.
CLIM_START = "2013-01-01"
CLIM_END   = "2022-12-31"

# Accuracy of the deployed tool on the window above, mean absolute annual error.
# RE-DERIVE THIS with extend_climatology.py whenever you change the window --
# do not carry a figure over from a different one.
DEPLOYED_ACCURACY_PCT = 5.77      # 2013-2022 window, mean absolute annual error

# ---- system -----------------------------------------------------------------
SYSTEM_KWP = 4.0
# source: the modal domestic system size, constrained by the 3.68 kW (16 A)
#         single-phase limit for connection under Engineering Recommendation
#         G98 without prior DNO approval. Corroborated by DESNZ solar PV cost
#         data (release 28 May 2026), in which the 0-4 kW band accounted for
#         107,137 of 176,894 domestic installations in 2025/26 -- roughly 61%,
#         the largest single band.
#         Cite the G98 limit to the ENA Engineering Recommendation and the
#         installation counts to the DESNZ release.

# ---- capital ----------------------------------------------------------------
CAPEX_PER_KWP = 2095.51
# source: DESNZ (2026) "Average costs of small-scale solar PV - Declared Net
#         Capacity (DNC)", Solar PV cost data, GOV.UK official statistics,
#         released 28 May 2026, covering April 2025 - March 2026.
#         https://www.gov.uk/government/statistics/solar-pv-cost-data
#         Table: "Domestic costs (DNC)", financial year 2025/26, Installation
#         Type = Domestic, 0-4 kW band, MEAN cost = GBP 2,095.51 per kW,
#         computed over 107,137 domestic installations.
#         Underlying data are sourced from the MCS database and the cost
#         includes generation equipment, installation, connection to the
#         electricity supply, and VAT where applicable.
#
# BASIS -- state this in the method chapter. These statistics are published on a
# DECLARED NET CAPACITY basis, i.e. per kW of maximum deliverable output, which
# for a domestic system is the inverter/connection limit. The modelled array is
# 4 kWp of PANEL capacity (Total Installed Capacity basis), and a 4 kWp array is
# typically inverter-limited to 3.68 kW under the G98 single-phase connection
# limit. Applying a DNC unit cost to a kWp figure therefore overstates capital
# slightly, which is the CONSERVATIVE direction: it lengthens estimated payback
# and so works against, not for, this study's central claim that vendor
# estimates are optimistic. Declare it as a stated limitation.
#
# A TIC-basis version of the same publication exists at the URL above and would
# remove the mismatch; switching to it is the cleanest future refinement.
#
# Previous value was GBP 1,565/kWp. That was the INFLATION-ADJUSTED mean for
# 2024/25 from the summary table -- a real-terms index figure, not the cash
# price a household pays. Do not reintroduce it.
#
# MUST be a UK INSTALLED cost: panels, inverter, mounting, labour, scaffolding,
# certification, installer margin.
#
# DO NOT use Alibaba / Made-in-China / wholesale module prices. Those are
# ex-factory panel costs only -- the same error as the 20p/W figure -- and omit
# every UK-side cost, which is most of the total.

BATTERY_COST = 0.0
# source: not applicable -- no battery is modelled. State this as a scope
#         decision in the method chapter: the study evaluates the accuracy of
#         GENERATION estimates, and storage changes the self-consumption
#         fraction rather than generation, so including it would add a second
#         source of variation without testing anything about the estimate.
# Set > 0 to model storage. Raises SELF_CONSUMPTION; see below.

# ---- tariffs ----------------------------------------------------------------
IMPORT_PRICE = 0.2632
# source: Ofgem energy price cap, electricity unit rate, direct debit,
#         1 Oct - 31 Dec 2026 = 26.32 p/kWh. The cap changes quarterly, so
#         this value is frozen at AS_OF rather than refreshed; see the note
#         at the top of this file.
#         https://www.ofgem.gov.uk/information-consumers/energy-advice-households/
#         energy-price-cap-unit-rates-and-standing-charges

SEG_RATE = 0.05
# DECISION: 5p/kWh is used deliberately as a CONSERVATIVE MARKET FLOOR, not as
# a market median. Justify it in exactly those terms. SEG rates are supplier-set
# and uncapped, ranging from roughly 1p to over 15p, and the highest advertised
# rates are typically conditional on also buying the import tariff from the same
# supplier. A household that does not meet those conditions, or does not shop
# around, receives something close to the floor.
#
# A higher rate shortens payback for EVERY method equally, so it does not change
# the study's central comparison -- but it does change the absolute payback
# figures substantially, so the sensitivity must be reported. Do NOT quote 15p
# anywhere in the write-up unless this value is changed and everything re-run.
#
# source: Ofgem publishes an annual Smart Export Guarantee report listing every
#         licensed tariff -- the primary source. For a market snapshot, Which?
#         and Sunsave maintain comparison tables:
#         https://www.which.co.uk/news/article/smart-export-guarantee-rates-the-best-and-worst-seg-tariffs-for-solar-panel-owners-azICP0i78MD8
#         https://www.sunsave.energy/solar-panels-advice/exporting-to-the-grid/best-seg-rates
# Smart Export Guarantee. Supplier-set and NOT capped, so rates vary widely --
# from about 1p to over 15p/kWh, with the highest usually tied to buying your
# import tariff from the same supplier.
#
# Ofgem publishes annual SEG reports listing every licensed tariff -- that is
# the citable primary source. Which? and Sunsave maintain comparison tables.
# State whether you used a market median, a specific named tariff, or a
# conservative floor, and justify it. 5p is a defensible conservative middle.

PRICE_ESCALATION = 0.00
# source: modelling assumption, stated not cited. Electricity prices are held
#         flat in REAL terms over the appraisal period. This is the neutral and
#         most defensible choice: a positive real escalation would shorten
#         payback for every method equally and so would flatter the study's
#         own models as much as the vendor formula, while requiring a forecast
#         of real energy prices over 25 years that no source can supply
#         credibly. State it explicitly as an assumption.
# Real annual increase in electricity price. 0.00 = flat in real terms, which is
# the conservative and easiest-to-defend assumption. If you use a non-zero value,
# justify it and note that payback is highly sensitive to it.

# ---- behaviour --------------------------------------------------------------
SELF_CONSUMPTION = 0.35
# source: adapted downward from McKenna, E., Pless, J. and Darby, S.J. (2018)
#         'Solar photovoltaic self-consumption in the UK residential sector:
#         New estimates from a smart grid demonstration project', Energy Policy,
#         118, pp. 482-491. doi:10.1016/j.enpol.2018.04.006
#         That study monitored 302 UK households at high resolution and found a
#         mean self-consumption of approximately 45% of generation, on an
#         average system size of 2.9 kWp.
#
# WHY 35% AND NOT 45% -- justify it in these terms:
#   (a) self-consumed FRACTION falls as system size rises, and the modelled
#       array is 4 kWp against their 2.9 kWp mean;
#   (b) participants in a smart-grid demonstration project are plausibly more
#       energy-engaged than households generally;
#   (c) a lower fraction lengthens payback, so it is the conservative direction.
# Report 45% in the sensitivity grid so the effect of the choice is visible.
#
# Fraction of generation used on site rather than exported. Typically quoted
# around 30-50% without storage, higher with a battery. This is one of the two
# assumptions payback is most sensitive to -- report the sensitivity grid.

# ---- technical --------------------------------------------------------------
DEGRADATION = 0.005
# source: 0.5% per year, consistent with the linear performance warranties
#         published by Tier 1 module manufacturers (typically 0.4-0.55%/yr).
#         The value used is the warranted floor, which is what a household
#         is actually guaranteed, and it is described as such wherever it
#         is reported. A measured field rate would be the stronger basis;
#         Jordan and Kurtz, "Photovoltaic Degradation Rates: An Analytical
#         Review", Progress in Photovoltaics, is the standard meta-analysis
#         if that route is preferred, but its bibliographic details must be
#         confirmed on the publisher page before it is cited.

INVERTER_REPLACE = (13, 900.0)   # (year, cost GBP)
# source: modelling assumption. String inverters for domestic systems are
#         generally warranted for 5-12 years and commonly replaced once within
#         a 25-year panel life; year 13 places the replacement just beyond a
#         typical extended warranty. The cost is a modelling assumption, not
#         a quoted price, and is reported as such. Payback sensitivity to it
#         is included in the sensitivity analysis.
# NOTE: this differs from the GBP 800 at year 12 quoted in an earlier draft of
#       the assumptions table. The code value is authoritative -- regenerate the
#       table with assumptions_table() rather than retyping it.
# Inverters typically outlast neither the panels nor the payback period.

# ---- appraisal --------------------------------------------------------------
DISCOUNT_RATE = 0.05
# source: PRIVATE HOUSEHOLD opportunity cost of capital, NOT a social discount
#         rate. The appraisal in this study is taken from the perspective of an
#         individual household choosing between installing solar and leaving the
#         capital in an alternative use, so the relevant rate is the return that
#         household forgoes. 5% is defensible as a long-run nominal return on
#         household savings or as an approximate cost of secured borrowing.
#         SUBSTANTIATE IT: cite either a Bank of England household deposit or
#         mortgage rate series for the study period, or a named product rate,
#         and state the date.
#
# STATE EXPLICITLY IN THE METHOD CHAPTER that this is NOT the HM Treasury Green
# Book Social Time Preference Rate (historically 3.5% for the first 30 years).
# Citing the Green Book while using 5% would be an error. The two frameworks
# answer different questions and this study uses the private one.
#   https://www.gov.uk/government/publications/green-book-supplementary-guidance-discounting
#
# This choice materially affects NPV and whether discounted payback is reached
# at all within the horizon.

HORIZON_YEARS = 25
# source: standard module performance-warranty period. Tier 1 manufacturers
#         typically warrant linear performance for 25 years, and the same
#         horizon is used in the comparable UK techno-economic literature --
#         e.g. Cui, Y., Tian, S., Zhu, J., Zoras, S. and Shao, Y. (2023)
#         'Energy and stochastic economic assessments of photovoltaic systems
#         in the East Midlands', Energies, 16(18), 6723. doi:10.3390/en16186723
#         Using the same horizon makes the payback figures directly comparable
#         to that study, which is worth saying in the discussion.


def capital_cost(system_kwp=SYSTEM_KWP):
    return CAPEX_PER_KWP * system_kwp + BATTERY_COST


def assumptions_table():
    """The assumption set, with sources, for the method chapter.

    GENERATE THE DISSERTATION TABLE FROM THIS FUNCTION. Do not retype the
    values into Word by hand -- that is exactly how the table and the code
    drifted apart before, and a table that disagrees with the code that
    produced the results is a reproducibility failure an examiner can catch
    in one minute.

        python economics.py --csv > assumptions_table.csv

    then insert that CSV into the document.
    """
    import pandas as pd
    rows = [
        ("System size", f"{SYSTEM_KWP:.1f} kWp",
         "Modal domestic size; 3.68 kW G98 single-phase connection limit. "
         "DESNZ (2026) solar PV cost data: 0-4 kW band = 61% of domestic "
         "installations, 2025/26."),
        ("Installed cost", f"£{CAPEX_PER_KWP:,.2f} /kW",
         "DESNZ (2026) 'Average costs of small-scale solar PV (DNC)', released "
         "28 May 2026. Domestic costs table, 2025/26, 0–4 kW mean, n = 107,137. "
         "Declared Net Capacity basis — see method chapter."),
        ("Battery", f"£{BATTERY_COST:,.0f}",
         "No storage modelled; scope decision (storage alters self-consumption, "
         "not generation)."),
        ("Total capital", f"£{capital_cost():,.0f}",
         "Calculated: installed cost × system size."),
        ("Import price", f"{IMPORT_PRICE*100:.2f} p/kWh",
         "Ofgem energy price cap, electricity unit rate, direct debit, "
         "1 Oct – 31 Dec 2026."),
        ("Export (SEG) rate", f"{SEG_RATE*100:.2f} p/kWh",
         "Smart Export Guarantee. Conservative market floor, not a median; "
         "supplier-set and uncapped. Sensitivity reported."),
        ("Real price escalation", f"{PRICE_ESCALATION:.1%} /yr",
         "Stated assumption: prices held flat in real terms."),
        ("Self-consumption", f"{SELF_CONSUMPTION:.0%}",
         "Adapted downward from McKenna, Pless and Darby (2018), Energy Policy "
         "118, 482-491, who measured ~45% on 2.9 kWp systems across 302 UK "
         "households. Lower fraction reflects the larger array; conservative."),
        ("Degradation", f"{DEGRADATION:.2%} /yr",
         "Tier 1 module linear performance warranty, typically 0.4 to "
         "0.55% per year. This is a warranted floor rather than a "
         "measured field degradation rate."),
        ("Inverter replacement",
         f"year {INVERTER_REPLACE[0]}, £{INVERTER_REPLACE[1]:,.0f}",
         "Modelling assumption: a single mid-life replacement. String "
         "inverters are typically warranted for 5 to 12 years, so year "
         "13 places the replacement just beyond a typical extended "
         "warranty. Payback sensitivity to this assumption is reported."),
        ("Discount rate", f"{DISCOUNT_RATE:.1%}",
         "Private household opportunity cost of capital, not the HM "
         "Treasury Green Book social time preference rate."),
        ("Appraisal horizon", f"{HORIZON_YEARS} years",
         "Standard module performance-warranty period; matches Cui et al. "
         "(2023), Energies 16(18), 6723, for comparability."),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value",
                                       "Justification / Source"])


if __name__ == "__main__":
    import sys
    t = assumptions_table()
    if "--csv" in sys.argv:
        print(t.to_csv(index=False))
    else:
        check_freshness()
        print()
        print(t.to_string(index=False))
        print(f"\nTotal capital: GBP {capital_cost():,.0f}")
        print("\nRegenerate the dissertation table from this function "
              "(python economics.py --csv). Never retype it.")
