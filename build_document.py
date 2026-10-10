"""Corrections, figures and tables for AL4IIP_draft_corrected_v3.docx.

Sections:
  1  equation objects -> plain text
  2  factual / arithmetic corrections verified against the CSVs
  3  British English + typography
  4  table renumbering (ten tables, document order) and cross-references
  5  new tables built from tables/*.csv
  6  figures 1-5 inserted as images, captions beneath
  7  Chapter 6 split out of Chapter 5
"""
import copy
import os
import shutil
import sys

import pandas as pd
from lxml import etree
from PIL import Image

SCRIPTS = os.environ.get("SOLAR_SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

UNPACKED = "unpacked"
DOC = f"{UNPACKED}/word/document.xml"
RELS = f"{UNPACKED}/word/_rels/document.xml.rels"
CT = f"{UNPACKED}/[Content_Types].xml"
SRC = SCRIPTS

TEXT_W_DXA = 9020          # A4 minus 1 inch margins each side
TEXT_W_IN = 6.2

tree = etree.parse(DOC)
body = tree.getroot().find(W + "body")


# ------------------------------------------------------------------ helpers
def ptext(el):
    return "".join(el.itertext()).strip()


def find(pred, start=0):
    for i in range(start, len(body)):
        if etree.QName(body[i]).localname == "p" and pred(ptext(body[i])):
            return i
    raise LookupError(pred)


def set_para(el, text):
    ts = list(el.iter(W + "t"))
    if not ts:
        r = el.find(W + "r")
        if r is None:
            return
        ts = [etree.SubElement(r, W + "t")]
    ts[0].text = text
    ts[0].set(XML_SPACE, "preserve")
    for t in ts[1:]:
        t.text = ""


def para_template(idx):
    """Nearest preceding body paragraph with real text, to clone for style.

    Starts one BEFORE the insertion point. Starting at it would clone the
    element being pushed down -- which, when inserting a caption above a
    section heading, produced bold captions.
    """
    for j in range(idx, -1, -1):
        p = body[j]
        if (etree.QName(p).localname == "p" and p.findall(W + "r")
                and ptext(p) and p.find(W + "pPr/" + W + "outlineLvl") is None):
            return p
    raise LookupError("no template paragraph")


def new_para(at_idx, text, italic=False):
    p = copy.deepcopy(para_template(at_idx - 1))
    for om in list(p.iter(M + "oMath")):
        om.getparent().remove(om)
    for d in list(p.iter(W + "drawing")):
        d.getparent().getparent().remove(d.getparent())
    set_para(p, text)
    if italic:
        for rpr in p.iter(W + "rPr"):
            if rpr.find(W + "i") is None:
                rpr.insert(0, etree.Element(W + "i"))
    body.insert(at_idx, p)
    return p


def tbl_after(idx, span=6):
    for j in range(idx, min(idx + span, len(body))):
        if etree.QName(body[j]).localname == "tbl":
            return j
    raise LookupError(f"no table after {idx}")


def rebuild_table(tbl, rows, total_dxa=TEXT_W_DXA, col_ratio=None):
    ncols = max(len(r) for r in rows)
    template_row = tbl.findall(W + "tr")[-1]
    while len(tbl.findall(W + "tr")) > len(rows):
        tbl.remove(tbl.findall(W + "tr")[-1])
    while len(tbl.findall(W + "tr")) < len(rows):
        tbl.append(copy.deepcopy(template_row))

    if col_ratio and len(col_ratio) == ncols:
        tot = sum(col_ratio)
        widths = [total_dxa * c // tot for c in col_ratio]
    else:
        widths = [total_dxa // ncols] * ncols

    for tr in tbl.findall(W + "tr"):
        tcs = tr.findall(W + "tc")
        while len(tr.findall(W + "tc")) > ncols:
            tr.remove(tr.findall(W + "tc")[-1])
        while len(tr.findall(W + "tc")) < ncols:
            tr.append(copy.deepcopy(tcs[-1]))
        for ci, tc in enumerate(tr.findall(W + "tc")):
            tcpr = tc.find(W + "tcPr")
            if tcpr is None:
                tcpr = etree.Element(W + "tcPr")
                tc.insert(0, tcpr)
            for old in tcpr.findall(W + "tcW"):
                tcpr.remove(old)
            tcw = etree.Element(W + "tcW")
            tcw.set(W + "w", str(widths[ci]))
            tcw.set(W + "type", "dxa")
            tcpr.insert(0, tcw)

    # The source tables are tblW=auto, so Word re-derives column widths from
    # content and ignores tcW. Pin the table and switch to fixed layout.
    tblpr = tbl.find(W + "tblPr")
    if tblpr is not None:
        for tw in tblpr.findall(W + "tblW"):
            tw.set(W + "w", str(total_dxa))
            tw.set(W + "type", "dxa")
        if tblpr.find(W + "tblLayout") is None:
            layout = etree.Element(W + "tblLayout")
            layout.set(W + "type", "fixed")
            anchor = tblpr.find(W + "tblCellMar")
            if anchor is not None:
                anchor.addprevious(layout)
            else:
                tblpr.append(layout)

    grid = tbl.find(W + "tblGrid")
    if grid is not None:
        for gc in grid.findall(W + "gridCol"):
            grid.remove(gc)
        for wdt in widths:
            etree.SubElement(grid, W + "gridCol").set(W + "w", str(wdt))

    for tr, vals in zip(tbl.findall(W + "tr"), rows):
        for tc, val in zip(tr.findall(W + "tc"), list(vals) + [""] * ncols):
            ps = tc.findall(W + "p")
            for p in ps[1:]:
                tc.remove(p)
            set_para(ps[0], str(val))
    return tbl


def insert_table(at_idx, df, template_tbl, col_ratio=None):
    tbl = copy.deepcopy(template_tbl)
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    rebuild_table(tbl, rows, col_ratio=col_ratio)
    body.insert(at_idx, tbl)
    return tbl


# ------------------------------------------------------------ image support
_rels = etree.parse(RELS)
_rels_root = _rels.getroot()
_next_rid = [max(int(r.get("Id")[3:]) for r in _rels_root) + 1]
os.makedirs(f"{UNPACKED}/word/media", exist_ok=True)
_pic_id = [1000]

PIC_XML = """<w:p xmlns:w="{w}" xmlns:r="{r}" xmlns:wp="{wp}" xmlns:a="{a}" xmlns:pic="{pic}">
<w:pPr><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>
<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{pid}" name="Picture {pid}" descr="{desc}"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="{picuri}"><pic:pic>
<pic:nvPicPr><pic:cNvPr id="{pid}" name="{fname}" descr="{desc}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"""

NS = dict(
    w="http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    r="http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    pic="http://schemas.openxmlformats.org/drawingml/2006/picture",
    picuri="http://schemas.openxmlformats.org/drawingml/2006/picture",
)


def insert_image(at_idx, png, desc, width_in=TEXT_W_IN):
    fname = os.path.basename(png)
    shutil.copy(png, f"{UNPACKED}/word/media/{fname}")
    rid = f"rId{_next_rid[0]}"
    _next_rid[0] += 1
    rel = etree.SubElement(
        _rels_root,
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    rel.set("Id", rid)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/"
                    "2006/relationships/image")
    rel.set("Target", f"media/{fname}")

    w_px, h_px = Image.open(png).size
    cx = int(width_in * 914400)
    cy = int(cx * h_px / w_px)
    _pic_id[0] += 1
    xml = PIC_XML.format(cx=cx, cy=cy, rid=rid, fname=fname, desc=desc,
                         pid=_pic_id[0], **NS)
    body.insert(at_idx, etree.fromstring(xml))


# =========================================================== 1  equations
KEEP_EQ = "Capacity"
EQ_SPECIAL = {
    "Capacity\xa0Factor=Generation\xa0(MWh)Installed\xa0Capacity\xa0(MWp)×24":
        "Capacity Factor = Generation (MWh) ÷ "
        "[Installed Capacity (MWp) × 24]",
}
EQ_TIDY = [("R2", "R²"), ("−", "-"), ("∘", "°"),
           ("\xa0", " ")]


def convert_equations():
    converted = kept = 0
    for om in list(tree.getroot().iter(M + "oMath")):
        raw = "".join(t.text or "" for t in om.iter(M + "t")).strip()
        if raw.startswith(KEEP_EQ):
            kept += 1
            continue
        text = EQ_SPECIAL.get(raw, raw)
        if text is raw or EQ_SPECIAL.get(raw) is None:
            for a, b in EQ_TIDY:
                text = text.replace(a, b)
        # Prefer a sibling ordinary run's rPr: the body runs carry an explicit
        # Arial rFonts, so merely stripping the math font would drop the text
        # to the theme default and render it in a visibly different face.
        host = om.getparent()
        if etree.QName(host).localname == "oMathPara":
            host = host.getparent()
        # Nearest sibling run, searching backwards then forwards -- not simply
        # the first, which in a bulleted paragraph is the bold lead-in and
        # would render the converted value in bold.
        src_rpr = None
        kids = list(host)
        pos = kids.index(om if om in kids else om.getparent())
        for step in (range(pos - 1, -1, -1), range(pos + 1, len(kids))):
            for j in step:
                el = kids[j]
                if el.tag == W + "r" and el.find(W + "rPr") is not None:
                    src_rpr = el.find(W + "rPr")
                    break
            if src_rpr is not None:
                break
        if src_rpr is None:
            for mr in om.iter(M + "r"):
                if mr.find(W + "rPr") is not None:
                    src_rpr = mr.find(W + "rPr")
        run = etree.Element(W + "r")
        if src_rpr is not None:
            new_rpr = copy.deepcopy(src_rpr)
            if src_rpr.getparent() is not None and \
                    etree.QName(src_rpr.getparent()).localname == "r" and \
                    src_rpr.getparent().tag == M + "r":
                for fonts in new_rpr.findall(W + "rFonts"):
                    new_rpr.remove(fonts)
            run.append(new_rpr)
        t = etree.SubElement(run, W + "t")
        t.text = text
        t.set(XML_SPACE, "preserve")
        parent = om.getparent()
        target = om
        if etree.QName(parent).localname == "oMathPara":
            target, parent = parent, parent.getparent()
        idx = list(parent).index(target)
        parent.remove(target)
        parent.insert(idx, run)
        converted += 1
    print(f"  equations: {converted} converted, {kept} kept as formulas")


# ================================================ 2 & 3  text substitutions
SUBS = [
    # --- §4.1 diagnostics: ranges were computed on three regions, not six,
    #     and on a robust STL fit that collapses the residual into 2024.
    ("yielding trend strengths between 0.007 and 0.016, alongside seasonal "
     "strengths of 0.805 to 0.817",
     "yielding trend strengths between 0.057 and 0.105, alongside seasonal "
     "strengths of 0.870 to 0.896", 1),

    # --- §4.5 and §5.2: measured latitude gradient, recomputed on all six
    #     regions (span 6.44%, r = -0.777).
    ("that the measured fleet does not exhibit (6.0% range, r=-0.477)",
     "that the measured fleet exhibits far more weakly (6.4% range, "
     "r = -0.777)", 1),
    ("However, the actual measured PV_Live fleet data exhibited a much "
     "narrower span of only 6.0% (r=-0.477).",
     "However, the actual measured PV_Live fleet data exhibited a much "
     "narrower span of only 6.4% (r = -0.777).", 1),

    # --- §5.2: Perth does NOT out-yield the Midlands (827.8 against 837.0
    #     kWh/kWp over the three years). It does out-yield the two regions
    #     immediately south of it, which is the defensible version.
    ("Most notably, the northernmost region, Perth (56.4°N), consistently "
     "out-yielded the Midlands over the three-year study period. Perth "
     "converted incident irradiance into capacity factor approximately 10% "
     "more efficiently than every other region.",
     "Most notably, the northernmost region, Perth (56.4°N), out-yielded "
     "both West Yorkshire and Northumberland over the three-year study "
     "period, despite lying between one and three degrees further north. "
     "Perth converted incident irradiance into capacity factor around 11% "
     "more efficiently than the mean of the other five regions, and between "
     "8% and 14% more efficiently than each of them individually.", 1),

    # --- §5.3: the window sensitivity spans 1.81 percentage points across 26
    #     years of window extension, which is of the order of one per cent per
    #     decade, not five. Claim restated as what the numbers support.
    ("This pattern implies an underlying atmospheric trend equating to "
     "roughly a 5% increase in regional solar yield per decade.",
     "Extending the window backwards by twenty-six years lowers the estimate "
     "by 1.81 percentage points, which is consistent with an underlying "
     "brightening trend of the order of one to one and a half per cent per "
     "decade — the same order of magnitude as the dimming and brightening "
     "phases documented in the radiation literature.", 1),

    # --- §4.7: sensitivity range restated to match the grid in Table 10.
    ("payback ranges from 11.2 years to beyond the 25-year appraisal horizon",
     "payback ranges from 15.0 years to beyond the 25-year appraisal horizon",
     1),

    # --- §5.1: anchor the illustrative figures to a real region.
    ("it dictates whether a homeowner allocates thousands of pounds of "
     "capital under the assumption of an 18-year or a 22-year return on "
     "investment.",
     "in Bury St Edmunds it dictates whether a homeowner allocates thousands "
     "of pounds of capital under the assumption of the 18.7-year return "
     "PVGIS implies or the 22.0-year return the measured data supports.", 1),

    # --- cross-references, after renumbering.
    ("the complete set of assumptions is reproduced in Table 3.",
     "the complete set of assumptions is reproduced in Table 4.", 1),
    ("Table 5 reports annual generation error for every estimation method",
     "Table 6 reports annual generation error for every estimation method", 1),

    # --- British English.
    ("Utilizing", "Utilising", 1),
]

CAPTION_RENUMBER = [
    ("Table 2: The three information sets",
     "Table 3: The three information sets", 1),
    ("Table 3: Financial and techno-economic assumptions",
     "Table 4: Financial and techno-economic assumptions", 1),
    ("Table 4: Series diagnostics by region (ADF, KPSS, Mann-Kendall, STL).",
     "Table 5: Series diagnostics by region: stationarity tests, and "
     "trend and seasonal strength from STL decomposition.", 1),
    ("Table 5: Annual generation error by estimation method",
     "Table 6: Annual generation error by estimation method", 1),
    ("Table 6: Payback error by estimation method",
     "Table 10: Payback error by estimation method", 1),
]


def substitute(pairs):
    """Replace within a single w:t first; fall back to whole-paragraph.

    Converting the equation objects leaves the affected sentences split
    across several runs, so a sentence that spans a former equation has no
    contiguous w:t to match. For those, rewrite the paragraph's full text and
    collapse it to its first run -- safe here because every target paragraph
    is uniform body text.
    """
    for old, new, expected in pairs:
        hits = 0
        for n in tree.getroot().iter(W + "t"):
            if n.text and old in n.text:
                hits += n.text.count(old)
                n.text = n.text.replace(old, new)
        how = "run"
        if hits == 0:
            for p in tree.getroot().iter(W + "p"):
                ts = [t for t in p.iter(W + "t")]
                joined = "".join(t.text or "" for t in ts)
                if old in joined:
                    ts[0].text = joined.replace(old, new)
                    ts[0].set(XML_SPACE, "preserve")
                    for t in ts[1:]:
                        t.text = ""
                    hits += 1
                    how = "para"
        flag = "ok  " if hits == expected else "MISMATCH"
        print(f"  [{flag}] {hits}/{expected} ({how})  {old[:52]}")


# ================================================================== run it
print("1  equations")
convert_equations()

print("2  caption renumbering (order matters: 6 before 5 before 4 ...)")
substitute(list(reversed(CAPTION_RENUMBER)))

print("3  corrections and cross-references")
substitute(SUBS)

# ------------------------------------------------ 4  new tables + captions
print("4  new tables")
T = {name: pd.read_csv(f"{SRC}/tables/{name}.csv",
                       dtype=str, keep_default_na=False)
     for name in ["data_sources", "series_diagnostics", "shap_top10",
                  "deployment", "sensitivity", "averaging_order"]}
TEMPLATE_TBL = body[tbl_after(find(lambda t: t.startswith("Table 6: Annual")) - 3)]

# Table 10 first (latest position), then work backwards so indices hold.
i = find(lambda t: t.startswith("Table 10: Payback error"))
new_para(i + 1, "Table 11: Payback sensitivity to import price and "
                "self-consumption fraction, measured generation.")
insert_table(i + 2, T["sensitivity"], TEMPLATE_TBL, col_ratio=(3, 2, 2, 2, 2, 2))

# Table 8 first, then the decomposition, so the two tables end up in
# document order: each insertion re-finds "4.7 Payback", which by then sits
# after whatever was inserted before it.
i = find(lambda t: t.startswith("4.7 Payback"))
new_para(i, "Table 8: Deployment model against measured three-year mean "
            "yield, 2013\u20132022 climatology window. Per-region errors are "
            "the published one-decimal values.")
insert_table(i + 1, T["deployment"], TEMPLATE_TBL, col_ratio=(4, 4, 4, 3, 3, 3))

i = find(lambda t: t.startswith("4.7 Payback"))
new_para(i, "The residual under-prediction is not, however, a property of the "
            "window alone. Reconstructing the deployment run from the saved "
            "model reproduces it as \u22125.9%, within 0.2 percentage points "
            "of the published figure, and holding the model and the weather "
            "record fixed while varying only what is fed to it separates that "
            "error into two roughly equal parts. Run on observed daily "
            "weather for the evaluation period the pooled model is "
            "essentially unbiased (\u22120.0% mean error). Replacing those "
            "real days with a "
            "day-of-year mean of the same period, which changes nothing but "
            "the smoothing, costs 2.8 percentage points. Shifting the window "
            "back to 2013\u20132022 costs a further 3.1. Half the deployed "
            "error therefore has nothing to do with which years are chosen: "
            "it is the arithmetic consequence of averaging the weather before "
            "modelling it rather than after.")
i = find(lambda t: t.startswith("The residual under-prediction is not"))
new_para(i + 1, "This suggests a correction that requires no new data. "
                "Running the model over each of the ten historical years "
                "separately and averaging the resulting annual yields uses "
                "exactly the same weather record and the same model, and "
                "differs only in the order of the two operations. Doing so "
                "reduces the mean absolute annual error of the deployed tool "
                "from 5.95% to 2.34%, recovering 3.85 percentage points "
                "(Table 9). The remaining error is the genuine window effect: "
                "2013\u20132022 was dimmer than the period against which the "
                "tool is being judged, which no reordering can repair.")
i = find(lambda t: t.startswith("This suggests a correction that requires"))
new_para(i + 1, "Table 9: Order of averaging. Both columns use the same "
                "2013\u20132022 weather record and the same pooled model; "
                "only the order of averaging differs.")
insert_table(i + 2, T["averaging_order"], TEMPLATE_TBL,
             col_ratio=(4, 3, 3, 2, 3, 2))

i = find(lambda t: t.startswith("4.5 The Vendor Benchmark"))
new_para(i, "Table 7: SHAP feature importance, weather-aware XGBoost, "
            "ten highest-ranked features.")
insert_table(i + 1, T["shap_top10"], TEMPLATE_TBL, col_ratio=(1, 5, 2))

i = find(lambda t: t.startswith("Table 5: Series diagnostics"))
insert_table(i + 1, T["series_diagnostics"], TEMPLATE_TBL,
             col_ratio=(5, 3, 3, 4, 3, 3, 3, 4))

i = find(lambda t: t.startswith("3.2 Target Variable"))
new_para(i, "Table 2: Data sources, coverage and role in the analysis.")
insert_table(i + 1, T["data_sources"], TEMPLATE_TBL,
             col_ratio=(4, 8, 7, 5))

# ------------------------------------------------------------ 5  figures
print("5  figures")
FIGS = os.path.join(SCRIPTS, "figures")

# Figure 5 is new: it belongs with the non-stationary-target limitation.
i = find(lambda t: t.startswith("Crucially, this regional drift almost "
                                "perfectly predicts"))
new_para(i + 1, "Figure 5: Change in capacity factor per unit irradiance, "
                "2023 to 2025, against residual annual error of the "
                "weather-aware models.")
insert_image(i + 1, f"{FIGS}/fig5_drift_vs_error.png",
             "Scatter plot of efficiency drift against residual model error")

i = find(lambda t: t.startswith("Figure 4: PVGIS estimate"))
insert_image(i, f"{FIGS}/fig4_pvgis_latitude.png",
             "PVGIS estimate and measured yield against latitude")

# The two figure captions currently share one paragraph; split them.
i = find(lambda t: t.startswith("Figure 2: Model comparison"))
set_para(body[i], "Figure 2: Model comparison — coefficient of "
                  "determination by forecast horizon, and mean absolute "
                  "annual error by method.")
new_para(i + 1, "Figure 3: Crawley, thirty-day-ahead forecast against "
                "observed daily capacity factor.")
insert_image(i + 1, f"{FIGS}/fig3_crawley_h30.png",
             "Thirty-day-ahead forecast against observed outturn, Crawley")
insert_image(i, f"{FIGS}/fig2_model_comparison.png",
             "Skill by horizon and annual generation error by method")

i = find(lambda t: t.startswith("Figure 1: STL decomposition"))
set_para(body[i], "Figure 1: Seasonal-trend decomposition (STL) of daily "
                  "capacity factor, Crawley, 2023 to 2025. All four panels "
                  "share a common vertical scale.")
insert_image(i, f"{FIGS}/fig1_stl_crawley.png",
             "STL decomposition of daily capacity factor")

# --------------------------------- 6  caption position: tables above, figures below
print("6  caption placement")
for label in ["Table 1: Hierarchy", "Table 3: The three information sets",
              "Table 6: Annual generation error", "Table 10: Payback error"]:
    ci = find(lambda t, k=label: t.startswith(k))
    ti = None
    for j in range(ci - 1, max(ci - 4, -1), -1):
        if etree.QName(body[j]).localname == "tbl":
            ti = j
            break
    if ti is None:
        print(f"  [skip] no table above {label}")
        continue
    cap = body[ci]
    body.remove(cap)
    body.insert(ti, cap)
    print(f"  moved {label[:28]} above its table")


# ------------------------------- 6b  climatology trend: figure 6 and 5.3 text
print("6b climatology trend")
i = find(lambda t: t.startswith("The documented physical driver"))
new_para(i, "Direct examination of the 2013\u20132022 reanalysis record shows "
            "why this trend cannot be pinned down more precisely, and Figure 6 "
            "sets the question beside its consequence. Fitting an ordinary "
            "least squares trend to annual mean irradiance across the six "
            "regions returns +7.1% per decade, but that estimate is not "
            "robust. Year-to-year variation has a standard deviation of 3.5% "
            "of the mean; 2022 was the brightest year in the record by a "
            "margin of 7.3%; and removing that single year halves the fitted "
            "slope to +3.8% per decade and removes its significance entirely "
            "(p = 0.28, against p = 0.06 for the full record). Only two of the "
            "six regions return a slope significant at the five per cent "
            "level. A ten-year record is therefore too short to separate a "
            "brightening trend of this magnitude from ordinary interannual "
            "variability. The window-sensitivity result is the more defensible "
            "estimate precisely because it rests on a longer baseline: "
            "regressing the four window estimates on their mid-years implies "
            "roughly 1.2% per decade, which is both an order of magnitude more "
            "plausible and consistent with the radiation literature.")
i = find(lambda t: t.startswith("Direct examination of the 2013"))
new_para(i + 1, "Figure 6: Annual mean shortwave irradiance across the six "
                "regions, 2013 to 2022, with fitted trends including and "
                "excluding 2022; and the annual generation estimate error "
                "produced by each candidate climatology window. The "
                "highlighted bar is the window selected for the deployment "
                "model.")
insert_image(i + 1, f"{FIGS}/fig6_irradiance_trend.png",
             "Annual irradiance trend and climatology window sensitivity")


# --------------------------- 6d  5.3: what the averaging order means
print("6d averaging-order discussion")
i = find(lambda t: t.startswith("This uncertainty is framed here"))
new_para(i + 1, "The order-of-averaging result in Section 4.6 sharpens this "
                "point considerably. A typical meteorological year is not "
                "merely a possibly unrepresentative sample of weather; it is a "
                "smoothed one, and smoothing is not neutral. Because the "
                "relationship between daily irradiance and delivered output is "
                "not linear across its range, running any model over an "
                "averaged year returns systematically less than averaging that "
                "model's output over the real years it was built from. In this "
                "study that ordering alone accounted for 3.85 percentage "
                "points of under-prediction, roughly half the total error of "
                "the deployed tool, and it was recovered without new data, a "
                "different model or a different window. PVGIS, the MCS "
                "standard method and every other typical-year approach share "
                "the same architecture, so they are exposed to the same bias "
                "in the same direction. That it operates opposite to the "
                "optimism this study documents elsewhere does not make it "
                "harmless: two biases of unknown relative size pointing in "
                "opposite directions leave a household no better informed "
                "than one large bias would.")


# ------------------------------------- 6c  keep captions with what they label
print("6c keepNext on captions and images")


def set_keep_next(p):
    ppr = p.find(W + "pPr")
    if ppr is None:
        ppr = etree.Element(W + "pPr")
        p.insert(0, ppr)
    if ppr.find(W + "keepNext") is not None:
        return False
    kn = etree.Element(W + "keepNext")
    style = ppr.find(W + "pStyle")
    if style is not None:
        style.addnext(kn)
    else:
        ppr.insert(0, kn)
    return True


kept = 0
for el in body:
    if etree.QName(el).localname != "p":
        continue
    # A table caption sits above its table; an image sits above its caption.
    if ptext(el).startswith("Table ") and ": " in ptext(el)[:12]:
        kept += set_keep_next(el)
    elif el.find(".//" + W + "drawing") is not None:
        kept += set_keep_next(el)
print(f"  keepNext set on {kept} paragraphs")


# ------------------------------------------------------- 7  Chapter 6 split
print("6  chapter 6")
i = find(lambda t: t == "5.5 Conclusions")
set_para(body[i], "6.1 Answers to the Research Questions")
# Clone a real chapter heading: the nearest preceding paragraph is a bullet
# from 5.4 Future Work, which would make the new heading a bullet too.
ch5 = body[find(lambda t: t == "Chapter 5: Discussion")]
head = copy.deepcopy(ch5)
set_para(head, "Chapter 6: Conclusions and Recommendations")
body.insert(i, head)

# ------------------------------------------------------------------ write
tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
_rels.write(RELS, xml_declaration=True, encoding="UTF-8", standalone=True)

ct = open(CT, encoding="utf-8").read()
if 'Extension="png"' not in ct:
    ct = ct.replace("<Default ", '<Default Extension="png" '
                    'ContentType="image/png"/><Default ', 1)
    open(CT, "w", encoding="utf-8").write(ct)
    print("  added png content type")

print("done.")
