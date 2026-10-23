# For output PDF generation
# Jackson
"""
batch_report.py  -  Batch PDF report generator + glue between model.py and inspection_ui.py

Each batch gets a 2-page PDF:
    Page 1  Part photo + expected data exactly as listed on the datasheet (and the drawing)
    Page 2  Results of this batch's analysis (conforming / non-conforming counts, statistics, charts,
            measured-vs-datasheet table, sensor signatures vs genuine reference, non-conforming item list)

HOW THE THREE FILES FIT TOGETHER
    model.py  trains/evaluates the model and provides featurize() + BoltVerifier
    batch_report.py   scans -> model verdicts -> batch.json + report.pdf in the UI's folder layout
    inspection_ui.py  watches that folder and shows the table, PDF, trends (File > Simulate batch)

    <root>/Bolts/91247A724/part.json                      (written by `setup`)
    <root>/Bolts/91247A724/reference_photo.jpg            (extracted from the datasheet)
    <root>/Bolts/91247A724/<BATCH-ID>/batch.json + report.pdf + parts.csv

QUICK START (keep all three .py files in one folder)
    pip install PySide6 numpy scipy scikit-learn joblib matplotlib reportlab pypdf pillow
    python model.py run --quick                       # train + validate (once)
    python batch_report.py setup --datasheet Schematic.pdf    # register part, extract photo/drawing
    python batch_report.py simulate --batches 3               # make 3 simulated batches + PDFs
    python inspection_ui.py                                   # browse them

REAL SCANNER DATA
    scans = [Scan(spectrum, vibration, vision_dict), ...]     # same inputs as model.featurize
    model = load_model()
    a = analyze_scans(model, scans, "91247A724")
    write_batch(root, a)                                      # PDF + batch.json, the UI picks it up
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepInFrame, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import model as bv  # noqa: E402

IDX = {n: i for i, n in enumerate(bv.FEATURE_NAMES)}
INCH = 25.4
PASS_C, FAIL_C = "#2e9e55", "#d64545"
STATUS_COLOR = {"CONFORM": PASS_C, "NON-CONFORM": FAIL_C}
FAIL_RATE = 0.10          # non-conforming threshold for batch acceptance
REVIEW_P = 0.01           # statistical threshold for batch acceptance

# --------------------------------------------------------------------------------------
# Expected data: McMaster-Carr 91247A724 datasheet
# --------------------------------------------------------------------------------------
DATASHEET = {
    "91247A724": {
        "ctype": "Bolts",
        "title": "Medium-Strength Grade 5 Steel Hex Head Screw",
        "subtitle": 'Zinc-Plated, 1/2"-13 Thread, 3" Long, Partially Threaded',
        "source": "McMaster-Carr 91247A724 datasheet",
        "key": [("Thread size", '1/2"-13'), ("Length", '3"'), ("Head width (across flats)", '3/4"'),
                ("Head height", '5/16" (0.313")'), ("Threading", "Partially threaded"),
                ("Minimum thread length", '1 1/4"'), ("Strength grade", "SAE Grade 5"),
                ("Material", "Zinc-plated steel"), ("Head markings", "3 radial lines + maker mark")],
        "rows": [("Thread direction", "Right hand"), ("Hex head profile", "Standard"),
                 ("Fastener head type", "Hex"), ("Drive style", "External hex"),
                 ("Strength rating", "Medium"), ("Material", "Zinc-plated steel"),
                 ("Thread size", '1/2"-13'), ("Length", '3"'), ("Head width", '3/4"'),
                 ("Head height", '5/16"'), ("Threading", "Partially threaded"),
                 ("Thread spacing", "Coarse"), ("Strength grade/class", "SAE Grade 5"),
                 ("Tensile strength", "120,000 psi"), ("Hardness", "Rockwell C25"),
                 ("Minimum thread length", '1 1/4"'), ("Screw size decimal equiv.", '0.5"'),
                 ("Specifications met", "ASME B18.2.1, SAE J429"), ("System of measurement", "Inch"),
                 ("Thread fit", "Unified Standard Class 2A"), ("Thread type", "UNC"),
                 ("Tip type", "Flat")],
        # (feature name, label, nominal, window lo, window hi)   inches unless noted
        # NOTE: nominal values come from the datasheet; WINDOWS ARE ASSUMED inspection limits
        # (nominal tolerance plus a measurement-uncertainty allowance). Edit to your control plan.
        "dims": [("len_mm", "Overall length (in)", 3.000, 2.970, 3.030),
                 ("af_mm", "Head width, across flats (in)", 0.750, 0.728, 0.760),
                 ("head_h_mm", "Head height (in)", 0.313, 0.298, 0.328),
                 ("shank_mm", "Shank diameter (in)", 0.500, 0.485, 0.505),
                 ("tpi", "Thread pitch (threads/in)", 13.0, 12.5, 13.5),
                 ("thread_mm", "Threaded length (in)", 1.25, 1.20, 1.60)],
        "straight_max_mm": 0.20,
        "marks": 3,
        # UI part.json (millimetres, same keys the batch.json 'dimensions' use)
        "ui_dims_mm": {"Length (mm)": 76.2, "Head AF (mm)": 19.05, "head_h_mm": 7.95,
                       "shank_mm": 12.7, "thread_mm": 31.75},
        "material": "Zinc-plated steel (SAE Grade 5)", "density": 7.85,
    }
}


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class Scan:
    spectrum: np.ndarray        # (131,) reflectance 400-1700 nm
    vibration: np.ndarray       # (8192,) tap response at bv.FS
    vision: dict                # see model.featurize
    truth: str | None = None    # simulation only: "genuine" or an impostor family name


@dataclass
class BatchAnalysis:
    part_number: str
    ctype: str
    batch_id: str
    timestamp: datetime
    X: np.ndarray
    anom: np.ndarray
    sup: np.ndarray
    suspect: np.ndarray
    reasons: list
    spectra: np.ndarray
    thr_a: float
    reference: dict                      # per-feature genuine mean / sd (from the model's scaler)
    model_info: dict
    truth: np.ndarray | None = None
    disposition: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.X)

    @property
    def n_suspect(self):
        return int(self.suspect.sum())


# --------------------------------------------------------------------------------------
# Model + analysis
# --------------------------------------------------------------------------------------
def find_model() -> Path:
    candidates = (Path.cwd() / "model.joblib", HERE / "model.joblib",
                  HERE / ".bolt_verifier_results" / "bolt_verifier.joblib", HERE / ".bolt_verifier.joblib")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return HERE / ".bolt_verifier.joblib"


def load_model(path: Path | None = None, train_if_missing: bool = True):
    """Load the trained verifier; optionally train a quick one (about a minute) if none exists."""
    path = Path(path) if path else find_model()
    if path.exists():
        return bv.BoltVerifier.load(path), path
    if not train_if_missing:
        raise FileNotFoundError(f"{path} not found. Run: python model.py run --quick")
    print("No trained model found - training a quick one (about a minute)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    Xtr, ftr = bv.make_dataset({"genuine": 2000, **{n: 80 for n in bv.FAMILY_NAMES}}, seed=1)
    Xcal, _ = bv.make_dataset({"genuine": 1500}, seed=2)
    m = bv.BoltVerifier(0.02).fit(Xtr[ftr == "genuine"], Xtr[ftr != "genuine"], Xcal)
    m.save(path)
    return m, path


def load_model_info(model_path: Path) -> dict:
    """Validation numbers from model's study (results.json next to the model), if present."""
    info = {"available": False}
    rp = Path(model_path).with_name("results.json")
    if not rp.exists():
        return info
    try:
        r = json.loads(rp.read_text())
        c = r["detectors"]["combined"]
        fam = c["detection_by_family"]
        info = {"available": True, "alpha": r["config"]["alpha"], "fsr": c["false_suspect_rate"],
                "macro_known": c["macro_detection"], "macro_unseen": r.get("unseen_macro_combined"),
                "blind": sorted([(f, v["rate"]) for f, v in fam.items() if v["rate"] < 0.6], key=lambda t: t[1])}
    except (KeyError, json.JSONDecodeError):
        pass
    return info


def next_batch_id(part_dir: Path, ts: datetime) -> str:
    prefix = f"BATCH-{ts:%Y%m%d}-"
    nums = [int(p.name[len(prefix):]) for p in part_dir.glob(prefix + "*") if p.name[len(prefix):].isdigit()] \
        if part_dir.exists() else []
    return f"{prefix}{(max(nums) + 1 if nums else 1):04d}"


def analyze_scans(model, scans, part_number: str, batch_id: str | None = None, timestamp: datetime | None = None,
                  model_info: dict | None = None, root: Path | None = None) -> BatchAnalysis:
    """Run the verifier on every scan in a batch and compute batch-level statistics."""
    ds = DATASHEET[part_number]
    ts = timestamp or datetime.now()
    feats, spectra, truth = [], [], []
    for s in scans:
        feats.append(bv.featurize(s.spectrum, s.vibration, s.vision))
        spectra.append(np.asarray(s.spectrum, float))
        truth.append(s.truth)
    X = np.vstack(feats)
    sc = model.scores(X)
    a_flag, s_flag = sc["anom"] > model.thr_a, sc["sup"] > model.thr_s
    suspect = a_flag | s_flag
    blocks = list(sc["block"])
    reasons = []
    for i in range(len(X)):
        r = [b for b in blocks if sc["block"][b][i] > model.thr_block[b]]
        if suspect[i] and not r:
            r = ["learned pattern"]
        reasons.append(r if suspect[i] else [])
    ref = {"mean": model.scaler.mean_, "sd": model.scaler.scale_}
    batch_id = batch_id or next_batch_id((root / ds["ctype"] / part_number) if root else Path("."), ts)
    a = BatchAnalysis(part_number, ds["ctype"], batch_id, ts, X, sc["anom"], sc["sup"], suspect, reasons,
                      np.vstack(spectra), model.thr_a, ref, model_info or {"available": False},
                      np.array(truth) if truth and truth[0] is not None else None)
    a.disposition = batch_disposition(a)
    return a


def batch_disposition(a: BatchAnalysis) -> dict:
    n, k = a.n, a.n_suspect
    mi = a.model_info
    p0 = mi["fsr"]["hi"] if mi.get("available") else 0.02          # conservative clean-lot suspect rate
    p_fsr = mi["fsr"]["rate"] if mi.get("available") else 0.02
    p_val = float(binom.sf(k - 1, n, p0)) if k > 0 else 1.0
    lo, hi = bv.wilson(k, n)
    rate = k / n
    if rate > FAIL_RATE or p_val < REVIEW_P:
        status, msg = "NON-CONFORM", ("One or more inspection criteria were not met. Hold the lot and investigate "
                                      "the non-conforming parts before release.")
    else:
        status, msg = "CONFORM", ("The batch meets the configured batch-level acceptance criteria. "
                                  "Any individually non-conforming parts must still be handled separately.")
    return {"status": status, "message": msg, "p_value": p_val, "rate": rate, "ci": (lo, hi),
            "expected_suspects": n * p_fsr, "excess": k - n * p_fsr, "p0": p0}


# --------------------------------------------------------------------------------------
# Simulated scanning (uses model's sensor simulator)
# --------------------------------------------------------------------------------------
def simulate_scans(n: int, contamination: float, rng: np.random.Generator, cfg=None, families=None):
    cfg = cfg or bv.SimConfig()
    families = families or bv.FAMILY_NAMES
    imp = set(rng.choice(n, int(rng.binomial(n, contamination)), replace=False).tolist()) if contamination > 0 else set()
    for i in range(n):
        fam = str(rng.choice(families)) if i in imp else "genuine"
        part = bv.genuine(rng) if fam == "genuine" else bv.FAMILIES[fam](rng)
        sp, vb, vs = bv.render(part, rng, cfg)
        yield Scan(sp, vb, vs, truth=fam)


# --------------------------------------------------------------------------------------
# Reference assets (photo / drawing)
# --------------------------------------------------------------------------------------
def extract_assets(datasheet_pdf: Path, out_dir: Path, part_number: str) -> dict:
    """Pull the product photo (page 1) and dimensional drawing (largest later image) from the datasheet PDF."""
    from pypdf import PdfReader
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(datasheet_pdf))
    photo = drawing = None
    best = 0
    for pi, page in enumerate(reader.pages):
        for im in page.images:
            w, h = im.image.size
            if pi == 0 and photo is None and w >= 600:
                photo = im.image
            elif pi > 0 and w * h > best and w >= 600:
                drawing, best = im.image, w * h
    found = {}
    for name, img in (("photo", photo), ("drawing", drawing)):
        if img is not None:
            p = out_dir / f"reference_{name}.jpg"
            img.convert("RGB").save(p, quality=92)
            found[name] = p
    return found


def find_asset(part_dir: Path, part_number: str, kind: str) -> Path | None:
    for base, stem in ((part_dir, f"reference_{kind}"), (HERE, f"{part_number}_{kind}"), (Path.cwd(), f"{part_number}_{kind}")):
        for ext in (".jpg", ".jpeg", ".png"):
            if (base / (stem + ext)).exists():
                return base / (stem + ext)
    return None


# --------------------------------------------------------------------------------------
# Part registration (creates the folders/spec the UI expects)
# --------------------------------------------------------------------------------------
def install_part(root: Path, part_number: str = "91247A724", datasheet_pdf: Path | None = None) -> Path:
    ds = DATASHEET[part_number]
    pdir = Path(root) / ds["ctype"] / part_number
    pdir.mkdir(parents=True, exist_ok=True)
    if datasheet_pdf:
        got = extract_assets(Path(datasheet_pdf), pdir, part_number)
        print(f"  extracted from datasheet: {', '.join(got) or 'nothing found'}")
    else:  # copy bundled assets if they sit next to this script
        for kind in ("photo", "drawing"):
            src = find_asset(pdir, part_number, kind)
            if src and src.parent != pdir:
                shutil.copy(src, pdir / f"reference_{kind}{src.suffix}")
    spec_path = pdir / "part.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec.update({"description": f"{ds['title']} - {ds['subtitle']}", "material": ds["material"],
                 "Density (g/cm3)": ds["density"], "dimensions": ds["ui_dims_mm"],
                 "tolerance_pct": spec.get("tolerance_pct", 1.5), "part_number": part_number,
                 "datasheet": {"source": ds["source"], "rows": ds["rows"]}})
    spec_path.write_text(json.dumps(spec, indent=2))
    return pdir


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------
def col(a: BatchAnalysis, name: str) -> np.ndarray:
    return a.X[:, IDX[name]]


def to_native(a: BatchAnalysis, name: str) -> np.ndarray:
    """Feature column converted to display units."""
    v = col(a, name)
    if name in ("len_mm", "af_mm", "head_h_mm", "shank_mm", "thread_mm"):
        return v / INCH
    return v


def ref_stat(a: BatchAnalysis, name: str):
    return a.reference["mean"][IDX[name]], a.reference["sd"][IDX[name]]


def fmt_pm(x: np.ndarray, d=4):
    return f"{np.mean(x):.{d}f} ± {np.std(x, ddof=1):.{d}f}"


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def make_charts(a: BatchAnalysis, ds: dict) -> bytes:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.3, 3.7))
    ok, bad = ~a.suspect, a.suspect
    c_ok, c_bad = "#4b7bb5", "#d64545"

    # A. anomaly score
    s = np.log10(np.maximum(a.anom, 1e-9) / a.thr_a)
    bins = np.linspace(min(s.min(), -0.8), max(s.max(), 0.5), 36)
    ax[0, 0].hist(s[ok], bins, color=c_ok, label="conform")
    ax[0, 0].hist(s[bad], bins, color=c_bad, label="non-conform")
    ax[0, 0].axvline(0, color="k", lw=0.9, ls="--")
    ax[0, 0].set_title("Anomaly score vs genuine-part model"); ax[0, 0].set_xlabel("log10 (score / threshold)")
    ax[0, 0].set_ylabel("parts"); ax[0, 0].legend(frameon=False)

    # B. length vs head width, with inspection window
    d = {k: (lo, hi, nom) for k, _, nom, lo, hi in ds["dims"]}
    L, W = to_native(a, "len_mm"), to_native(a, "af_mm")
    ax[0, 1].scatter(L[ok], W[ok], s=7, color=c_ok, alpha=.6, lw=0)
    ax[0, 1].scatter(L[bad], W[bad], s=10, color=c_bad, alpha=.9, lw=0)
    (l0, l1, _), (w0, w1, _) = d["len_mm"], d["af_mm"]
    ax[0, 1].add_patch(plt.Rectangle((l0, w0), l1 - l0, w1 - w0, fill=False, ec="k", ls="--", lw=.9))
    ax[0, 1].set_title("Length vs head width (dashed = inspection window)")
    ax[0, 1].set_xlabel("length (in)"); ax[0, 1].set_ylabel("head width (in)")

    # C. first resonance vs genuine reference band
    f1 = 10 ** col(a, "log_f1") / 1000
    m, sd = ref_stat(a, "log_f1")
    ax[1, 0].axvspan(10 ** (m - 3 * sd) / 1000, 10 ** (m + 3 * sd) / 1000, color="#2e9e55", alpha=.15,
                     label="genuine ±3 SD")
    b2 = np.linspace(max(0, min(f1.min(), 10 ** (m - 3.5 * sd) / 1000)), max(f1.max(), 10 ** (m + 3.5 * sd) / 1000), 40)
    ax[1, 0].hist(f1[ok], b2, color=c_ok); ax[1, 0].hist(f1[bad], b2, color=c_bad)
    ax[1, 0].set_title("Vibration: first resonance"); ax[1, 0].set_xlabel("kHz"); ax[1, 0].set_ylabel("parts")
    ax[1, 0].legend(frameon=False)

    # D. flags by modality
    names = ["vision", "vibration", "hyperspectral", "cross-sensor", "learned pattern"]
    counts = [sum(n in r for r in a.reasons) for n in names]
    ax[1, 1].barh(names, counts, color=c_bad); ax[1, 1].invert_yaxis()
    ax[1, 1].set_title("Non-conforming parts by detection source (a part can have several)")
    ax[1, 1].set_xlabel("parts")
    for i, c in enumerate(counts):
        ax[1, 1].text(c, i, f" {c}", va="center")
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# PDF building
# --------------------------------------------------------------------------------------
_ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=_ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3b57"))
B = ParagraphStyle("B", parent=_ss["Normal"], fontSize=8, leading=10)
SM = ParagraphStyle("SM", parent=B, fontSize=6.8, leading=8.4, textColor=colors.HexColor("#555555"))
TC = ParagraphStyle("TC", parent=B, fontSize=7.6, leading=9)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")
TH = ParagraphStyle("TH", parent=TC, fontName="Helvetica-Bold", textColor=colors.white)
BIG = ParagraphStyle("BIG", parent=B, fontSize=17, leading=19, alignment=TA_CENTER, fontName="Helvetica-Bold")
LAB = ParagraphStyle("LAB", parent=SM, alignment=TA_CENTER)


def P(t, s=TC):
    return Paragraph(str(t), s)


def grid(rows, widths, header=True, zebra=True, extra=()):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8ced6"))]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")))
    if zebra:
        st.append(("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]))
    t.setStyle(TableStyle(st + list(extra)))
    return t


class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont("Helvetica", 7)
            self.setFillColor(colors.HexColor("#777777"))
            self.drawRightString(letter[0] - 0.6 * inch, 0.38 * inch, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _placeholder_photo(w):
    from reportlab.graphics.shapes import Drawing, Polygon, Rect, String
    h = w * 0.55
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#f2f2f2"), strokeColor=colors.HexColor("#cccccc")))
    d.add(Rect(w * .12, h * .40, w * .55, h * .20, fillColor=colors.HexColor("#c9d3dc"), strokeColor=colors.grey))
    d.add(Polygon([w * .67, h * .32, w * .78, h * .32, w * .84, h * .5, w * .78, h * .68, w * .67, h * .68],
                  fillColor=colors.HexColor("#b4c0cc"), strokeColor=colors.grey))
    d.add(String(w * .5, h * .12, "Reference photo not found", textAnchor="middle", fontSize=8, fillColor=colors.grey))
    return d


def _img(path: Path, width):
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * h / w)


def page1(a: BatchAnalysis, ds: dict, photo: Path | None, drawing: Path | None, width) -> list:
    S = []
    S.append(P(f"<b>{ds['title']}</b>", ParagraphStyle("t", parent=B, fontSize=14, leading=17, textColor=colors.HexColor("#1b5e35"))))
    S.append(P(ds["subtitle"], ParagraphStyle("s", parent=B, fontSize=9.5, leading=12, textColor=colors.HexColor("#1b5e35"))))
    S.append(P(f"Part number {a.part_number}  |  Expected data from the {ds['source']}", SM))
    S.append(Spacer(1, 5))
    pw = 3.75 * inch
    left = [_img(photo, pw - 6) if photo else _placeholder_photo(pw - 6), Spacer(1, 2),
            P("Reference photo of the target part (datasheet).", SM)]
    key = grid([[P("Key expected value", TH), P("Datasheet", TH)]] + [[P(k), P(f"<b>{v}</b>")] for k, v in ds["key"]],
               [1.65 * inch, width - pw - 1.65 * inch - 6])
    top = Table([[left, key]], colWidths=[pw, width - pw])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S += [top, P("All datasheet properties", H)]

    rows = ds["rows"]
    half = math.ceil(len(rows) / 2)
    L, R = rows[:half], rows[half:] + [("", "")] * (2 * half - len(rows) - 0 if len(rows) < 2 * half else 0)
    R = R[:half] + [("", "")] * (half - len(R))
    cw = [1.45 * inch, width / 2 - 1.45 * inch]
    body = [[P("Property", TH), P("Value", TH), P("Property", TH), P("Value", TH)]]
    for (k1, v1), (k2, v2) in zip(L, R):
        body.append([P(k1), P(f"<b>{v1}</b>"), P(k2), P(f"<b>{v2}</b>")])
    S.append(grid(body, cw + cw))
    S.append(Spacer(1, 6))

    # drawing + inspection windows
    dw = 4.1 * inch
    win = [[P("Characteristic", TH), P("Nominal", TH), P("Window", TH)]]
    for _, label, nom, lo, hi in ds["dims"]:
        win.append([P(label), P(f"<b>{nom:g}</b>"), P(f"{lo:g} to {hi:g}")])
    win.append([P("Straightness (mm)"), P("-"), P(f"0 to {ds['straight_max_mm']:g}")])
    win.append([P("Head radial marks"), P(f"<b>{ds['marks']}</b>"), P(f"{ds['marks']}")])
    wt = grid(win, [1.28 * inch, 0.5 * inch, width - dw - 1.78 * inch - 8])
    right = [P("Inspection windows used in this report", H), wt, Spacer(1, 3),
             P("Nominals are from the datasheet. Windows are assumed inspection limits (nominal tolerance plus a "
               "measurement-uncertainty allowance); edit DATASHEET in batch_report.py to match your control plan.", SM)]
    if drawing:
        left2 = [P("Dimensional drawing (datasheet)", H), _img(drawing, dw - 4)]
        t = Table([[left2, right]], colWidths=[dw, width - dw])
    else:
        t = Table([[right]], colWidths=[width])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S.append(t)
    return S


def page2(a: BatchAnalysis, ds: dict, width) -> list:
    S, d = [], a.disposition
    col_s = STATUS_COLOR[d["status"]]
    n, k = a.n, a.n_suspect
    cards = [[P(str(n), BIG), P(str(n - k), BIG), P(str(k), BIG), P(f"{d['rate']:.1%}", BIG),
              P(f"{d['p_value']:.3g}" if d["p_value"] >= 0.001 else f"{d['p_value']:.1e}", BIG)],
             [P("parts scanned", LAB), P("Conforming", LAB), P("Non-conforming", LAB),
              P(f"non-conforming rate (95% CI {d['ci'][0]:.1%}-{d['ci'][1]:.1%})", LAB),
              P("clean-lot test p-value", LAB)]]
    ct = Table(cards, colWidths=[width / 5] * 5)
    ct.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8ced6")),
                            ("LINEAFTER", (0, 0), (-2, -1), 0.25, colors.HexColor("#c8ced6")),
                            ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(PASS_C)),
                            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    banner = Table([[P(f"<font color='white'><b>BATCH RESULT: {'Conform' if d['status'] == 'CONFORM' else 'Non-conform'}</b></font>",
                       ParagraphStyle("bn", parent=B, fontSize=10.5, leading=13)),
                     P(f"<font color='white'>{d['message']}</font>", ParagraphStyle("bm", parent=B, fontSize=7.6, leading=9.2))]],
                   colWidths=[2.3 * inch, width - 2.3 * inch])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(col_s)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S += [banner, Spacer(1, 4), ct,
          P(f"Expected non-conforming detections in a conforming lot of {n}: {d['expected_suspects']:.1f}. Observed: {k} "
            f"(excess {d['excess']:+.1f}). p-value = P(at least {k} detections | conforming lot, false-detection rate "
            f"{d['p0']:.1%}).", SM), Spacer(1, 3)]

    img = Image(io.BytesIO(make_charts(a, ds)), width=width, height=width * 3.7 / 7.3)
    S += [img, P("Measured vs datasheet", H)]

    # measured vs datasheet table
    rows = [[P(x, TH) for x in ("Characteristic", "Datasheet", "Window", "Batch mean ± SD", "Min - Max", "Outside window")]]
    for name, label, nom, lo, hi in ds["dims"]:
        v = to_native(a, name)
        dd = 3 if name in ("tpi",) else 4
        out = int(np.sum((v < lo) | (v > hi)))
        rows.append([P(label), P(f"{nom:g}"), P(f"{lo:g} to {hi:g}"), P(fmt_pm(v, dd)),
                     P(f"{v.min():.{dd}f} - {v.max():.{dd}f}"), P(f"{out} ({out / n:.1%})")])
    sv = col(a, "straight_mm")
    rows.append([P("Straightness (mm)"), P("-"), P(f"0 to {ds['straight_max_mm']:g}"), P(fmt_pm(sv, 3)),
                 P(f"{sv.min():.3f} - {sv.max():.3f}"), P(f"{int((sv > ds['straight_max_mm']).sum())} ({(sv > ds['straight_max_mm']).mean():.1%})")])
    mk = float(np.mean(col(a, "marks_n") == ds["marks"]))
    rows.append([P("Head radial marks read"), P(str(ds["marks"])), P(str(ds["marks"])), P(f"{mk:.1%} read as {ds['marks']}"),
                 P("-"), P(f"{int(round((1 - mk) * n))} ({1 - mk:.1%})")])
    S.append(grid(rows, [1.75 * inch, 0.65 * inch, 0.95 * inch, 1.45 * inch, 1.3 * inch, width - 6.1 * inch]))

    # sensor signatures vs genuine reference
    S.append(P("Sensor signatures vs genuine-part reference (from model training data)", H))

    def sig(label, fn, name, d=3, ref_fn=None):
        v = fn(col(a, name))
        m, sd = ref_stat(a, name)
        rm, rsd = (ref_fn(m, sd) if ref_fn else (m, sd))
        shift = (np.mean(v) - rm) / (rsd + 1e-12)
        return [P(label), P(fmt_pm(v, d)), P(f"{rm:.{d}f} ± {rsd:.{d}f}"), P(f"{shift:+.2f}")]

    lg = lambda m, sd: (10 ** m, 10 ** m * math.log(10) * sd)
    srows = [[P(x, TH) for x in ("Signature", "Batch mean ± SD", "Genuine reference", "Shift (reference SDs)")]]
    srows.append(sig("Resonance 1 (kHz)", lambda v: 10 ** v / 1000, "log_f1", 2,
                     lambda m, sd: tuple(x / 1000 for x in lg(m, sd))))
    srows.append(sig("Resonance 2 (kHz)", lambda v: 10 ** v / 1000, "log_f2", 2,
                     lambda m, sd: tuple(x / 1000 for x in lg(m, sd))))
    srows.append(sig("Damping ratio (x 1e-4)", lambda v: 10 ** v * 1e4, "log_zeta", 2,
                     lambda m, sd: tuple(x * 1e4 for x in lg(m, sd))))
    srows.append(sig("Stiffness-implied wave speed (m/s)", lambda v: 10 ** v, "log_c_est", 0, lg))
    mr = np.mean(a.X[:, [IDX[f"R_bin{i:02d}"] for i in range(20)]], axis=1)
    rm = np.mean([a.reference["mean"][IDX[f"R_bin{i:02d}"]] for i in range(20)])
    rsd = float(np.sqrt(np.mean([a.reference["sd"][IDX[f"R_bin{i:02d}"]] ** 2 for i in range(20)])))
    srows.append([P("Mean reflectance, 400-1700 nm"), P(fmt_pm(mr, 3)), P(f"{rm:.3f} ± {rsd:.3f}"), P(f"{(mr.mean() - rm) / rsd:+.2f}")])
    srows.append(sig("Coating tint (R560 / R460)", lambda v: v, "ratio_560_460", 3))
    srows.append(sig("Oil-film band depth (1210 nm)", lambda v: v, "oil_1210", 3))
    S.append(grid(srows, [2.4 * inch, 1.7 * inch, 1.7 * inch, width - 5.8 * inch]))

    # suspect list
    S.append(P("Non-conforming parts for confirmatory testing (highest anomaly first)", H))
    order = np.argsort(-a.anom / a.thr_a)
    order = [i for i in order if a.suspect[i]][:8]
    if order:
        r2 = [[P(x, TH) for x in ("Part #", "Score / thr.", "Impostor prob.", "Detection source", "Length (in)", "Head W (in)", "Res. 1 (kHz)")]]
        for i in order:
            r2.append([P(f"{i + 1}"), P(f"{a.anom[i] / a.thr_a:.1f}"), P(f"{a.sup[i]:.2f}"), P(", ".join(a.reasons[i]) or "-"),
                       P(f"{a.X[i, IDX['len_mm']] / INCH:.3f}"), P(f"{a.X[i, IDX['af_mm']] / INCH:.3f}"),
                       P(f"{10 ** a.X[i, IDX['log_f1']] / 1000:.2f}")])
        S.append(grid(r2, [0.5 * inch, 0.75 * inch, 0.85 * inch, width - 4.55 * inch, 0.8 * inch, 0.8 * inch, 0.85 * inch]))
        if k > len(order):
            S.append(P(f"... and {k - len(order)} more (see parts.csv in the batch folder).", SM))
    else:
        S.append(P("No non-conforming parts in this batch.", B))

    # model notes
    mi = a.model_info
    if mi.get("available"):
        blind = ", ".join(f"{f} ({r:.0%})" for f, r in mi["blind"]) or "none below 60%"
        unseen = f", unseen-family {mi['macro_unseen']:.1%}" if mi.get("macro_unseen") else ""
        note = (f"Model validation (simulation): false-suspect rate {mi['fsr']['rate']:.1%} "
                f"(95% CI {mi['fsr']['lo']:.1%}-{mi['fsr']['hi']:.1%}); average impostor detection "
                f"{mi['macro_known']:.1%}{unseen}. Weakest detection: {blind}. Simulated validation is not a substitute "
                f"for testing on measured parts.")
    else:
        note = ("No validation study found next to the model. Run 'python model.py run' to generate false-suspect "
                "and detection statistics for this report.")
    S += [Spacer(1, 3), P(note, SM)]
    if a.truth is not None:
        imp = a.truth != "genuine"
        caught, escaped = int((imp & a.suspect).sum()), int((imp & ~a.suspect).sum())
        fs = int((~imp & a.suspect).sum())
        esc_f = sorted({str(t) for t in a.truth[imp & ~a.suspect]})
        S.append(P(f"<b>MODEL TEST BATCH - reference labels:</b> {int(imp.sum())} impostor part(s) inserted; {caught} correctly identified, "
                   f"{escaped} not identified{' (' + ', '.join(esc_f) + ')' if esc_f else ''}; {fs} genuine part(s) incorrectly marked non-conforming.", SM))
    return S


def build_pdf(a: BatchAnalysis, out_path: Path, photo: Path | None = None, drawing: Path | None = None) -> Path:
    ds = DATASHEET[a.part_number]
    doc = BaseDocTemplate(str(out_path), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                          topMargin=0.95 * inch, bottomMargin=0.55 * inch,
                          title=f"Batch report {a.batch_id}", author="Component Inspection System",
                          subject=f"{a.part_number} batch analysis")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    titles = {1: "Reference part and expected datasheet data", 2: "Batch analysis results"}

    def on_page(c, d):
        pg = c.getPageNumber()
        c.saveState()
        c.setFillColor(colors.HexColor("#1f3b57"))
        c.rect(0, letter[1] - 0.78 * inch, letter[0], 0.78 * inch, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(0.6 * inch, letter[1] - 0.38 * inch, "COMPONENT INSPECTION REPORT")
        c.setFont("Helvetica", 8.5)
        c.drawString(0.6 * inch, letter[1] - 0.58 * inch, titles.get(pg, "Batch report"))
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(letter[0] - 0.6 * inch, letter[1] - 0.38 * inch, f"{a.batch_id}")
        c.setFont("Helvetica", 8.5)
        c.drawRightString(letter[0] - 0.6 * inch, letter[1] - 0.58 * inch,
                          f"Part {a.part_number}  |  {a.timestamp:%Y-%m-%d %H:%M}")
        c.restoreState()

    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
    h = doc.height - 2
    story = [KeepInFrame(doc.width, h, page1(a, ds, photo, drawing, doc.width), mode="shrink"), PageBreak(),
             KeepInFrame(doc.width, h, page2(a, ds, doc.width), mode="shrink")]
    doc.build(story, canvasmaker=NumberedCanvas)
    return out_path


# --------------------------------------------------------------------------------------
# Write a finished batch into the UI's folder layout (atomically)
# --------------------------------------------------------------------------------------
def batch_json(a: BatchAnalysis, ds: dict) -> dict:
    n, k = a.n, a.n_suspect
    dims = {key: round(float(np.mean(col(a, f))), 3) for key, f in
            (("Length (mm)", "len_mm"), ("Head AF (mm)", "af_mm"), ("head_h_mm", "head_h_mm"),
             ("shank_mm", "shank_mm"), ("thread_mm", "thread_mm"))}
    spec_ok = float(np.mean([("hyperspectral" not in r) for r in a.reasons]))
    mod = {m: int(sum(m in r for r in a.reasons)) for m in ("vision", "vibration", "hyperspectral", "cross-sensor", "learned pattern")}
    d = a.disposition
    meta = {
        "batch_id": a.batch_id, "timestamp": a.timestamp.isoformat(timespec="seconds"),
        "item_count": n, "pass_count": n - k, "suspect_count": k,
        "material": ds["material"] if d["status"] == "CONFORM" else "Inconsistent with datasheet",
        "material_confidence": round(spec_ok, 3),          # share of parts whose spectrum matched the genuine model
        "dimensions": dims,
        "vibration_peak_hz": round(float(np.mean(10 ** col(a, "log_f1"))), 1),
        "status": d["status"],
        "verification": {"verified": n - k, "suspect": k, "suspect_rate": round(d["rate"], 4),
                         "p_value": d["p_value"], "excess_suspects": round(d["excess"], 2), "detection_source": mod,
                         "message": d["message"]},
    }
    if a.truth is not None:
        imp = a.truth != "genuine"
        meta["simulated"] = {"impostors_inserted": int(imp.sum()), "caught": int((imp & a.suspect).sum()),
                             "escaped": int((imp & ~a.suspect).sum()), "false_suspects": int((~imp & a.suspect).sum())}
    return meta


def write_batch(root: Path, a: BatchAnalysis, photo: Path | None = None, drawing: Path | None = None) -> Path:
    ds = DATASHEET[a.part_number]
    root = Path(root)
    pdir = install_part(root, a.part_number) if not (root / ds["ctype"] / a.part_number / "part.json").exists() \
        else root / ds["ctype"] / a.part_number
    photo = photo or find_asset(pdir, a.part_number, "photo")
    drawing = drawing or find_asset(pdir, a.part_number, "drawing")
    # build in a hidden staging folder, then rename into place so the UI never sees a half-written batch
    stage = root / ".staging" / uuid.uuid4().hex
    stage.mkdir(parents=True)
    try:
        (stage / "batch.json").write_text(json.dumps(batch_json(a, ds), indent=2))
        build_pdf(a, stage / "report.pdf", photo, drawing)
        with open(stage / "parts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["part", "verdict", "anomaly_score", "anomaly_threshold", "impostor_probability", "detection_source",
                        "length_in", "head_width_in", "head_height_in", "shank_in", "tpi", "thread_in", "marks",
                        "resonance1_hz"] + (["truth"] if a.truth is not None else []))
            for i in range(a.n):
                x = a.X[i]
                w.writerow([i + 1, "NON-CONFORM" if a.suspect[i] else "CONFORM", f"{a.anom[i]:.3f}", f"{a.thr_a:.3f}",
                            f"{a.sup[i]:.4f}", ";".join(a.reasons[i]), f"{x[IDX['len_mm']] / INCH:.4f}",
                            f"{x[IDX['af_mm']] / INCH:.4f}", f"{x[IDX['head_h_mm']] / INCH:.4f}",
                            f"{x[IDX['shank_mm']] / INCH:.4f}", f"{x[IDX['tpi']]:.2f}", f"{x[IDX['thread_mm']] / INCH:.4f}",
                            int(x[IDX['marks_n']]), f"{10 ** x[IDX['log_f1']]:.1f}"] + ([a.truth[i]] if a.truth is not None else []))
        final = pdir / a.batch_id
        if final.exists():
            final = pdir / f"{a.batch_id}-{uuid.uuid4().hex[:4]}"
        os.replace(stage, final)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            (root / ".staging").rmdir()
        except OSError:
            pass
    return final


def simulate_and_write(root: Path, n: int = 250, contamination: float = 0.01, model_path: Path | None = None,
                       seed: int | None = None, part_number: str = "91247A724") -> dict:
    """One call used by the CLI and by the UI's File > Simulate batch."""
    model, mp = load_model(model_path)
    info = load_model_info(mp)
    rng = np.random.default_rng(seed)
    a = analyze_scans(model, simulate_scans(n, contamination, rng), part_number, model_info=info, root=Path(root))
    out = write_batch(Path(root), a)
    return {"dir": out, "ctype": a.ctype, "part": a.part_number, "batch_id": a.batch_id,
            "status": a.disposition["status"], "suspects": a.n_suspect, "n": a.n}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("setup", help="register the part and extract photo/drawing from the datasheet PDF")
    s1.add_argument("--datasheet", type=Path)
    s1.add_argument("--part", default="91247A724")
    s2 = sub.add_parser("simulate", help="simulate scanned batches and write PDF reports")
    s2.add_argument("--batches", type=int, default=1)
    s2.add_argument("--n", type=int, default=250)
    s2.add_argument("--contamination", type=float, default=0.01)
    s2.add_argument("--model", type=Path)
    s2.add_argument("--seed", type=int)
    for s in (s1, s2):
        s.add_argument("--root", type=Path, default=Path(os.environ.get("INSPECTION_DATA", str(HERE))))
    a = ap.parse_args()
    if a.cmd == "setup":
        p = install_part(a.root, a.part, a.datasheet)
        print(f"Part registered at {p}")
    else:
        for i in range(a.batches):
            r = simulate_and_write(a.root, a.n, a.contamination, a.model, None if a.seed is None else a.seed + i)
            print(f"{r['batch_id']}: {'Conform' if r['status'] == 'CONFORM' else 'Non-conform'}  ({r['suspects']} non-conforming of {r['n']})  -> {r['dir']}")


if __name__ == "__main__":
    main()