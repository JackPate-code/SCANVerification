# For camera reading AI model
# Emma

"""
bolt_verifier.py  -  Multi-sensor "verified / suspect" model for McMaster-Carr 91247A724
(Grade 5 zinc-plated steel hex head screw, 1/2"-13 x 3", partially threaded)

WHAT THIS FILE DOES
  1. SIMULATOR   Physics-based generator of hyperspectral spectra, vibration ring-downs and
                 machine-vision measurements for genuine parts and 20 families of impostors /
                 defective parts (wrong length, wrong thread, wrong plating, stainless,
                 Grade 8, spoofed Grade 2, cracked, bent, corroded, plus subtle near-misses).
  2. MODEL       Feature extraction per sensor + cross-sensor consistency features, then
                 (a) a genuine-only anomaly detector (Mahalanobis distance), and
                 (b) a supervised gradient-boosting classifier on genuine vs impostors.
                 A part is SUSPECT if either flags it; thresholds are calibrated to a target
                 false-suspect rate on held-out genuine parts.
  3. RELIABILITY Statistical testing on fresh simulated data: Wilson 95% intervals, per-family
                 detection, unseen-family (leave-one-family-out) detection, batch-level Monte
                 Carlo, and noise / calibration-drift sweeps. Writes report.md, results.json
                 and PNG plots.

USAGE
    pip install numpy scipy scikit-learn joblib matplotlib
    python bolt_verifier.py run              # full study  (several minutes)
    python bolt_verifier.py run --quick      # small study (about a minute)
    python bolt_verifier.py test             # classify a few simulated parts with the saved model

USING REAL DATA LATER
    Everything the model needs from a part is one call:
        x = featurize(spectrum, vibration, vision)      # see featurize() docstring
    Build X for real genuine parts (and any real/known-bad parts), then:
        v = BoltVerifier(alpha=0.02).fit(X_genuine, X_impostor, X_genuine_calibration)
        v.predict_features(x)  ->  {"label": "VERIFIED" | "SUSPECT", "reasons": [...], ...}

HONEST LIMITS
    The simulator encodes physics from the datasheet (dimensions, thread, material, strength
    grade) plus ASSUMED sensor noise and part-to-part variation (see SimConfig and genuine()).
    The statistics below show how the pipeline behaves under those assumptions. They are NOT a
    substitute for validation on measured parts. Replace the assumptions with real data and
    re-run this same evaluation to get numbers you can defend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, dump, load
from scipy.signal import find_peaks, hilbert
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Target part (from the McMaster-Carr datasheet, 91247A724)
# --------------------------------------------------------------------------------------
TARGET = dict(
    part_number="91247A724", length_in=3.0, head_af_in=0.75, head_height_in=0.313,
    major_dia_in=0.5, tpi=13, min_thread_len_in=1.25, head_marks=3,
    material="zinc-plated medium-carbon steel", grade="SAE Grade 5",
)

# Sensor settings (change to match your hardware)
FS = 250_000                         # vibration sample rate, Hz
N_SAMP = 8192                        # samples per tap (~33 ms)
T = np.arange(N_SAMP) / FS
WIN = np.hanning(N_SAMP)
WL = np.arange(400.0, 1701.0, 10.0)  # hyperspectral bands, nm (131 bands, VNIR+SWIR)
X_WL = (WL - 400.0) / 1300.0


@dataclass(frozen=True)
class SimConfig:
    noise_mult: float = 1.0          # scales ALL sensor noise (1.0 = baseline assumption)
    spec_offset: float = 0.0         # additive reflectance calibration drift


# --------------------------------------------------------------------------------------
# Part model and impostor families
# --------------------------------------------------------------------------------------
@dataclass
class Part:
    L: float; af: float; h: float; d: float; tpi: float; thread: float   # inches
    marks: int; E: float; rho: float; zeta: float; surface: str
    straight: float = 0.04       # mm
    crack: float = 0.0
    oil: float = 0.0
    rust: float = 0.0
    k_head: float = 0.90         # head-mass frequency correction
    tex: float = 0.006           # spectral speckle (surface texture)


def genuine(rng: np.random.Generator) -> Part:
    """Genuine 91247A724 with ASSUMED manufacturing scatter. Replace with measured scatter."""
    return Part(
        L=rng.normal(3.0, 0.010), af=rng.normal(0.745, 0.003), h=rng.normal(0.313, 0.004),
        d=rng.normal(0.497, 0.0015), tpi=13.0, thread=max(1.25, rng.normal(1.30, 0.05)),
        marks=3, E=rng.normal(200, 3), rho=rng.normal(7850, 25),
        zeta=float(rng.lognormal(np.log(8e-4), 0.25)), surface="zinc_clear",
        straight=abs(rng.normal(0, 0.04)),
        oil=float(rng.uniform(0.005, 0.02)) if rng.random() < 0.08 else 0.0,
        k_head=rng.normal(0.90, 0.005),
    )


def _g(rng, **kw) -> Part:
    return replace(genuine(rng), **kw)


FAMILIES = {
    "short_length": lambda r: _g(r, L=r.choice([2.5, 2.75]) + r.normal(0, 0.01)),
    "long_length": lambda r: _g(r, L=r.choice([3.25, 3.5, 4.0]) + r.normal(0, 0.01)),
    "dia_3_8_16": lambda r: _g(r, d=0.372 + r.normal(0, 0.0015), af=0.5625 - 0.004 + r.normal(0, 0.003),
                               h=0.234 + r.normal(0, 0.004), tpi=16.0, thread=1.0 + abs(r.normal(0, .04))),
    "dia_5_8_11": lambda r: _g(r, d=0.622 + r.normal(0, 0.002), af=0.9375 - 0.005 + r.normal(0, 0.003),
                               h=0.39 + r.normal(0, 0.004), tpi=11.0, thread=1.5 + abs(r.normal(0, .04))),
    "metric_M12": lambda r: _g(r, L=2.992 + r.normal(0, 0.01), d=0.4685 + r.normal(0, 0.0015),
                               af=0.748 + r.normal(0, 0.003), h=0.295 + r.normal(0, 0.004),
                               tpi=14.51, thread=1.18 + abs(r.normal(0, .04)), marks=0),
    "fine_thread_UNF": lambda r: _g(r, tpi=20.0),
    "fully_threaded": lambda r: (lambda p: replace(p, thread=p.L))(genuine(r)),
    "heavy_hex_head": lambda r: _g(r, af=0.875 - 0.005 + r.normal(0, 0.003), h=0.344 + r.normal(0, 0.004),
                                   k_head=r.normal(0.86, 0.005)),
    "plain_oiled": lambda r: _g(r, surface="plain", oil=float(r.uniform(0.08, 0.3))),
    "black_oxide": lambda r: _g(r, surface="black"),
    "yellow_zinc": lambda r: _g(r, surface="yellow"),
    "hot_dip_galv": lambda r: _g(r, surface="hdg", d=0.5 + r.normal(0, 0.0015), tex=0.03),
    "stainless_18_8": lambda r: _g(r, E=r.normal(193, 3), rho=r.normal(8000, 25), marks=0,
                                   zeta=float(r.lognormal(np.log(1.2e-3), 0.25)), surface="stainless"),
    "grade_8": lambda r: _g(r, marks=6, surface=str(r.choice(["zinc_clear", "yellow"])),
                            zeta=float(r.lognormal(np.log(7.2e-4), 0.25))),
    "grade_2_spoofed": lambda r: _g(r, zeta=float(r.lognormal(np.log(9.2e-4), 0.25))),
    "cracked": lambda r: (lambda p: replace(p, crack=float(r.uniform(0.03, 0.12)),
                                            zeta=p.zeta * float(r.uniform(3, 10))))(genuine(r)),
    "bent": lambda r: (lambda p: replace(p, straight=float(r.uniform(0.25, 1.2)),
                                         k_head=p.k_head * (1 + r.normal(0, 0.012)), zeta=p.zeta * 1.3))(genuine(r)),
    "corroded": lambda r: (lambda p: replace(p, rust=float(r.uniform(0.2, 0.9)), zeta=p.zeta * 1.2))(genuine(r)),
    # ---- subtle / near-miss families (these are the ones that stress the model) ----
    "marginal_length": lambda r: _g(r, L=3.0 + r.choice([-1, 1]) * r.uniform(0.03, 0.06)),
    "zinc_nickel_plating": lambda r: _g(r, surface="zinc_nickel"),
    "undersize_shank": lambda r: _g(r, d=0.490 + r.normal(0, 0.0015)),
    "early_crack": lambda r: (lambda p: replace(p, crack=float(r.uniform(0.003, 0.015)),
                                                zeta=p.zeta * float(r.uniform(1.3, 2.5))))(genuine(r)),
}
FAMILY_NAMES = list(FAMILIES)


# --------------------------------------------------------------------------------------
# Sensor simulation
# --------------------------------------------------------------------------------------
def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


def _gauss(c, w):
    return np.exp(-((WL - c) / w) ** 2)


def surface_reflectance(p: Part, rng) -> np.ndarray:
    s = p.surface
    if s == "zinc_clear":
        R = 0.60 + 0.05 * X_WL + 0.035 * _gauss(440, 50)
    elif s == "zinc_nickel":
        R = 0.50 + 0.07 * X_WL
    elif s == "yellow":
        R = 0.20 + 0.40 * _sig((WL - 520) / 18) + 0.04 * X_WL
    elif s == "plain":
        R = 0.46 + 0.10 * X_WL
    elif s == "black":
        R = 0.06 + 0.03 * X_WL
    elif s == "hdg":
        R = 0.40 + 0.12 * X_WL
    elif s == "stainless":
        R = 0.52 + 0.10 * X_WL
    else:
        raise ValueError(s)
    if s in ("zinc_clear", "yellow", "zinc_nickel"):   # thin-film interference ripple
        R = R + 0.012 * np.sin(2 * np.pi * WL / rng.uniform(300, 600) + rng.uniform(0, 6.28))
    if p.rust > 0:
        rust = 0.12 + 0.30 * _sig((WL - 590) / 45)
        R = (1 - p.rust) * R + p.rust * rust
    if p.oil > 0:                       # C-H overtone absorption of an oil film
        R = R * (1 - p.oil * (_gauss(1210, 25) + 0.8 * _gauss(1690, 40)))
    return R


def render(p: Part, rng, cfg: SimConfig):
    """Return (spectrum[131], vibration[N_SAMP], vision dict) for one part."""
    nm = cfg.noise_mult
    # ---- hyperspectral ------------------------------------------------------------
    R = surface_reflectance(p, rng)
    gain = rng.lognormal(0, 0.07)
    drift = rng.normal(0, 0.006) + rng.normal(0, 0.006) * X_WL + rng.normal(0, 0.004) * X_WL ** 2
    spectrum = gain * R * (1 + rng.normal(0, p.tex * nm, WL.size)) + drift + cfg.spec_offset \
        + rng.normal(0, 0.004 * nm, WL.size)

    # ---- vibration (free-free beam modes + longitudinal) --------------------------------
    L, d = p.L * 0.0254, p.d * 0.0254
    th = min(p.thread / p.L, 1.0)
    c = math.sqrt(p.E * 1e9 / p.rho)
    d_eff = d * (1 - 0.06 * th)
    f1 = 3.5608 * (d_eff / 4) / L ** 2 * c * p.k_head * (1 - p.crack)
    f2 = f1 * 2.757
    f3 = c / (2 * L) * 0.93 * (1 - 0.3 * p.crack)
    amps = np.array([1.0, rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.5)]) * rng.uniform(0.6, 1.4)
    x = np.zeros(N_SAMP)
    for a, f in zip(amps, (f1, f2, f3)):
        z = p.zeta * (f / f1) ** 0.3
        x += a * np.exp(-z * 2 * np.pi * f * T) * np.sin(2 * np.pi * f * T + rng.uniform(0, 6.28))
    ref = amps[0]
    x += rng.normal(0, ref * 10 ** (-rng.normal(32, 3) / 20) * nm, N_SAMP)
    for _ in range(3):                   # conveyor / machinery rumble (below the analysis band)
        x += rng.uniform(0.2, 1.5) * ref * nm * np.sin(2 * np.pi * rng.uniform(40, 900) * T + rng.uniform(0, 6.28))
    if rng.random() < 0.03 * nm:         # shock from a neighbouring part
        s0, ln = rng.integers(0, N_SAMP - 400), rng.integers(100, 400)
        x[s0:s0 + ln] += rng.normal(0, ref * rng.uniform(0.3, 1.0), ln)

    # ---- machine vision (what a calibrated pipeline would report) ------------------------
    sd = 0.0035 * nm
    mark_read = p.marks if rng.random() > 0.015 else max(0, p.marks + int(rng.choice([-1, 1])))
    vision = {
        "length_mm": (p.L + rng.normal(0, sd)) * 25.4,
        "head_af_mm": (p.af + rng.normal(0, sd)) * 25.4,
        "head_h_mm": (p.h + rng.normal(0, sd)) * 25.4,
        "shank_mm": (p.d + rng.normal(0, sd)) * 25.4,
        "tpi": p.tpi + rng.normal(0, 0.12 * nm),
        "thread_mm": (p.thread + rng.normal(0, 0.03 * nm)) * 25.4,
        "straight_mm": abs(p.straight + rng.normal(0, 0.02 * nm)),
        "marks_n": mark_read,
    }
    return spectrum, x, vision


# --------------------------------------------------------------------------------------
# Feature extraction (works on simulated OR real sensor data)
# --------------------------------------------------------------------------------------
_BIN_IDX = np.array_split(np.arange(WL.size), 20)
_FR = np.fft.rfftfreq(2 * N_SAMP, 1 / FS)
_FR0 = np.fft.rfftfreq(N_SAMP, 1 / FS)
_DF = _FR[1]


def _wl(nm):
    return int(round((nm - 400) / 10))


VISION_F = ["len_mm", "af_mm", "head_h_mm", "shank_mm", "tpi", "thread_mm", "straight_mm",
            "marks_n", "marks_ok", "af_over_d", "len_over_d", "thread_over_len"]
VIB_F = ["log_f1", "log_f2", "log_f3", "a2_a1", "a3_a1", "log_zeta", "peak_snr_db", "n_peaks"]
SPEC_F = [f"R_bin{i:02d}" for i in range(20)] + [f"S_bin{i:02d}" for i in range(20)] + \
         ["slope", "ratio_560_460", "oil_1210", "oil_1650", "texture"]
FUSION_F = ["log_c_est", "f2_over_f1", "f3_over_f1"]
FEATURE_NAMES = VISION_F + VIB_F + SPEC_F + FUSION_F
BLOCKS = {
    "vision": list(range(0, len(VISION_F))),
    "vibration": list(range(len(VISION_F), len(VISION_F) + len(VIB_F))),
    "hyperspectral": list(range(len(VISION_F) + len(VIB_F), len(VISION_F) + len(VIB_F) + len(SPEC_F))),
    "cross-sensor": list(range(len(FEATURE_NAMES) - len(FUSION_F), len(FEATURE_NAMES))),
}


def vib_features(x: np.ndarray):
    x = np.asarray(x, float) - np.mean(x)
    spec = np.abs(np.fft.rfft(x * WIN, n=2 * N_SAMP))
    s = np.where(_FR >= 3000, spec, 0.0)
    floor = np.median(spec[_FR >= 3000]) + 1e-12
    pk, _ = find_peaks(s, height=10 * floor, distance=int(1500 / _DF))
    top = np.sort(pk[np.argsort(s[pk])[::-1]][:3])
    freqs, mags = [], []
    for i in top:
        y0, y1, y2 = np.log(s[i - 1:i + 2] + 1e-12)
        den = y0 - 2 * y1 + y2
        freqs.append(_FR[i] + (0.5 * (y0 - y2) / den if den != 0 else 0.0) * _DF)
        mags.append(s[i])
    n_pk = len(freqs)
    freqs += [0.0] * (3 - n_pk)
    mags += [0.0] * (3 - n_pk)
    strongest = max(mags) + 1e-12
    # damping from the strongest mode: band-pass -> Hilbert envelope -> log-linear fit
    f0 = freqs[int(np.argmax(mags))]
    log_zeta = -2.0
    if f0 > 0:
        mask = np.abs(_FR0 - f0) < max(0.04 * f0, 300)
        env = np.abs(hilbert(np.fft.irfft(np.fft.rfft(x) * mask, n=N_SAMP)))
        start = int(np.argmax(env[: N_SAMP // 3]))
        below = np.where(env[start:] < 0.2 * env[start])[0]
        end = start + int(below[0]) if below.size else N_SAMP - int(0.05 * N_SAMP)
        if end - start > 30:
            slope = np.polyfit(T[start:end], np.log(env[start:end] + 1e-12), 1)[0]
            log_zeta = float(np.log10(np.clip(-slope / (2 * np.pi * f0), 1e-5, 1e-1)))
    lf = [math.log10(max(f, 1.0)) for f in freqs]
    feats = [*lf, mags[1] / strongest, mags[2] / strongest, log_zeta,
             20 * math.log10(strongest / floor), float(n_pk)]
    return feats, freqs


def spec_features(sp: np.ndarray):
    sp = np.asarray(sp, float)
    snv = (sp - sp.mean()) / (sp.std() + 1e-9)
    raw_b = [sp[i].mean() for i in _BIN_IDX]
    snv_b = [snv[i].mean() for i in _BIN_IDX]
    slope = np.polyfit(X_WL, sp, 1)[0]
    r560_460 = sp[_wl(550):_wl(570) + 1].mean() / (sp[_wl(450):_wl(470) + 1].mean() + 1e-6)

    def dip(c):
        lo, hi, mid = _wl(c - 80), min(_wl(c + 80), WL.size - 1), _wl(c)
        return (sp[lo] + sp[hi]) / 2 - sp[mid - 1:mid + 2].mean()

    resid = sp - np.convolve(sp, np.ones(7) / 7, mode="same")
    texture = resid[4:-4].std() / (sp.mean() + 1e-6)
    return [*raw_b, *snv_b, slope, r560_460, dip(1210), dip(1650), texture]


def featurize(spectrum, vibration, vision: dict) -> np.ndarray:
    """
    spectrum  : (131,) calibrated reflectance, 400-1700 nm in 10 nm steps (white-referenced)
    vibration : (8192,) tap response sampled at FS = 250 kHz
    vision    : dict with length_mm, head_af_mm, head_h_mm, shank_mm, tpi, thread_mm,
                straight_mm, marks_n (number of radial lines read on the head)
    """
    v = vision
    vis = [v["length_mm"], v["head_af_mm"], v["head_h_mm"], v["shank_mm"], v["tpi"], v["thread_mm"],
           v["straight_mm"], v["marks_n"], float(v["marks_n"] == TARGET["head_marks"]),
           v["head_af_mm"] / v["shank_mm"], v["length_mm"] / v["shank_mm"], v["thread_mm"] / v["length_mm"]]
    vib, f = vib_features(vibration)
    f1, f2, f3 = f
    # fusion: effective wave speed implied by f1 and the *measured* geometry. It should
    # equal the steel value regardless of length/diameter, so it ties vibration to vision.
    c_est = f1 * 4 * (v["length_mm"] / 1000) ** 2 / (3.5608 * v["shank_mm"] / 1000) if f1 > 0 else 0.0
    fusion = [math.log10(max(c_est, 1.0)), f2 / f1 if f1 > 0 else 0.0, f3 / f1 if f1 > 0 else 0.0]
    return np.array(vis + vib + spec_features(spectrum) + fusion, dtype=float)


assert len(FEATURE_NAMES) == len(featurize(
    np.full(WL.size, 0.6), np.random.default_rng(0).normal(size=N_SAMP),
    dict(length_mm=76, head_af_mm=19, head_h_mm=8, shank_mm=12.6, tpi=13, thread_mm=33,
         straight_mm=0.1, marks_n=3)))


# --------------------------------------------------------------------------------------
# Dataset generation (parallel, never stores raw signals)
# --------------------------------------------------------------------------------------
def _gen_chunk(family: str, n: int, seed, cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = genuine if family == "genuine" else FAMILIES[family]
    out = np.empty((n, len(FEATURE_NAMES)))
    for i in range(n):
        sp, vb, vs = render(fn(rng), rng, cfg)
        out[i] = featurize(sp, vb, vs)
    return out


def make_dataset(plan: dict[str, int], seed: int, cfg: SimConfig = SimConfig(), n_jobs: int = -1, chunk=250):
    jobs, fams = [], []
    for fam, n in plan.items():
        for s in range(0, n, chunk):
            jobs.append((fam, min(chunk, n - s)))
    seeds = np.random.SeedSequence(seed).spawn(len(jobs))
    parts = Parallel(n_jobs=n_jobs)(delayed(_gen_chunk)(f, n, sd, cfg) for (f, n), sd in zip(jobs, seeds))
    X = np.vstack(parts)
    fam = np.concatenate([[f] * n for f, n in jobs])
    return X, fam


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------
class BoltVerifier:
    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha          # target false-suspect rate on genuine parts

    def _fit_sup(self, X_gen, X_imp):
        X = np.vstack([X_gen, X_imp])
        y = np.r_[np.zeros(len(X_gen)), np.ones(len(X_imp))]
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=24,
                                              l2_regularization=1.0, random_state=0).fit(X, y)

    def fit(self, X_gen, X_imp, X_cal_gen):
        self.scaler = StandardScaler().fit(X_gen)
        Z = self.scaler.transform(X_gen)
        self.cov = LedoitWolf().fit(Z)
        self.block_cov = {k: LedoitWolf().fit(Z[:, ix]) for k, ix in BLOCKS.items()}
        self.clf = self._fit_sup(X_gen, X_imp)
        self.calibrate(X_cal_gen)
        return self

    # --- scoring ---------------------------------------------------------------------
    def scores(self, X):
        X = np.atleast_2d(X)
        Z = self.scaler.transform(X)
        return {"anom": self.cov.mahalanobis(Z), "sup": self.clf.predict_proba(X)[:, 1],
                "block": {k: self.block_cov[k].mahalanobis(Z[:, ix]) for k, ix in BLOCKS.items()}}

    def calibrate(self, X_cal_gen):
        s = self.scores(X_cal_gen)
        q = 1 - self.alpha / 2          # two detectors share the false-suspect budget
        self.thr_a = float(np.quantile(s["anom"], q))
        self.thr_s = float(np.quantile(s["sup"], q))
        self.thr_block = {k: float(np.quantile(v, 0.995)) for k, v in s["block"].items()}

    def suspect(self, X, which="combined", thr_s=None):
        s = self.scores(X)
        a, p = s["anom"] > self.thr_a, s["sup"] > (self.thr_s if thr_s is None else thr_s)
        return {"anomaly": a, "supervised": p, "combined": a | p}[which]

    # --- single-part API ----------------------------------------------------------
    def predict_features(self, x) -> dict:
        s = self.scores(x)
        a, p = bool(s["anom"][0] > self.thr_a), bool(s["sup"][0] > self.thr_s)
        reasons = [k for k, v in s["block"].items() if v[0] > self.thr_block[k]]
        if p and not reasons:
            reasons = ["learned impostor pattern"]
        return {"label": "SUSPECT" if (a or p) else "VERIFIED", "reasons": reasons,
                "anomaly_score": float(s["anom"][0]), "anomaly_threshold": self.thr_a,
                "impostor_probability": float(s["sup"][0])}

    def predict(self, spectrum, vibration, vision) -> dict:
        return self.predict_features(featurize(spectrum, vibration, vision))

    def save(self, path):
        dump(self, path)

    @staticmethod
    def load(path) -> "BoltVerifier":
        return load(path)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(flags):
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def batch_monte_carlo(fam_k, fam_n, fsr_k, fsr_n, B, q, draws, rng):
    """Propagate test-set uncertainty (Beta posteriors) into per-batch outcomes."""
    fam_k, fam_n = np.asarray(fam_k), np.asarray(fam_n)
    miss = rng.beta(fam_n - fam_k + 1, fam_k + 1, size=(draws, len(fam_k))).mean(axis=1)  # P(escape)
    fsr = rng.beta(fsr_k + 1, fsr_n - fsr_k + 1, size=draws)
    n_imp = rng.binomial(B, q, size=draws)
    esc = rng.binomial(n_imp, miss)
    extra = rng.binomial(B - n_imp, fsr)
    has = n_imp >= 1
    p_any = float(np.mean(esc[has] > 0)) if has.any() else float("nan")
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"P_batch_has_impostor": 1 - (1 - q) ** B, "mean_escapes_per_batch": float(esc.mean()),
            "escapes_95": ci(esc), "P_escape_given_impostor_present": p_any,
            "mean_extra_tests_per_batch": float(extra.mean()), "extra_tests_95": ci(extra)}


# --------------------------------------------------------------------------------------
# Full study
# --------------------------------------------------------------------------------------
def run_study(out: Path, quick: bool, alpha: float, seed: int):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    S = dict(g_train=1500, f_train=80, g_cal=1500, g_test=2500, f_test=250, sweep_g=800, sweep_f=60) if quick else \
        dict(g_train=6000, f_train=300, g_cal=3000, g_test=6000, f_test=600, sweep_g=1500, sweep_f=150)
    plan = lambda g, f: {"genuine": g, **{n: f for n in FAMILY_NAMES}}

    print("Generating training / calibration / test sets (independent seeds)...")
    Xtr, ftr = make_dataset(plan(S["g_train"], S["f_train"]), seed + 1)
    Xcal, _ = make_dataset({"genuine": S["g_cal"]}, seed + 2)
    Xte, fte = make_dataset(plan(S["g_test"], S["f_test"]), seed + 3)
    print(f"  done in {time.time() - t0:.0f}s")

    gen_tr, imp_tr = Xtr[ftr == "genuine"], Xtr[ftr != "genuine"]
    model = BoltVerifier(alpha).fit(gen_tr, imp_tr, Xcal)
    model.save(out / "bolt_verifier.joblib")
    print("  model trained and calibrated")

    is_gen = fte == "genuine"
    res = {"config": {"alpha": alpha, "seed": seed, "sizes": S, "target": TARGET}}

    # ---- A. headline metrics for each detector -----------------------------------------
    sc = model.scores(Xte)
    res["detectors"] = {}
    for which in ("anomaly", "supervised", "combined"):
        flag = model.suspect(Xte, which)
        fsr = rate(flag[is_gen])
        fam = {f: rate(flag[fte == f]) for f in FAMILY_NAMES}
        det = [fam[f]["rate"] for f in FAMILY_NAMES]
        res["detectors"][which] = {"false_suspect_rate": fsr, "detection_by_family": fam,
                                   "macro_detection": float(np.mean(det)), "worst_family": min(fam, key=lambda f: fam[f]["rate"]),
                                   "worst_detection": float(min(det))}
    # ROC AUC (continuous scores) per family, combined score = max of rank-normalised scores
    a_rank = lambda v: np.searchsorted(np.sort(sc["anom"][is_gen]), v) / is_gen.sum()
    s_rank = lambda v: np.searchsorted(np.sort(sc["sup"][is_gen]), v) / is_gen.sum()
    comb = np.maximum(a_rank(sc["anom"]), s_rank(sc["sup"]))
    res["auc"] = {}
    for f in FAMILY_NAMES:
        m = (fte == f) | is_gen
        y = (fte[m] != "genuine").astype(int)
        res["auc"][f] = {"anomaly": float(roc_auc_score(y, sc["anom"][m])),
                         "supervised": float(roc_auc_score(y, sc["sup"][m])),
                         "combined": float(roc_auc_score(y, comb[m]))}
    res["auc_overall_combined"] = float(roc_auc_score((~is_gen).astype(int), comb))

    # ---- B. unseen-family generalisation (leave-one-family-out) ---------------------------
    print("Leave-one-family-out (unseen impostor types)...")
    loo = {}
    flag_a = model.suspect(Xte, "anomaly")
    for f in FAMILY_NAMES:
        keep = ftr != f
        clf = model._fit_sup(gen_tr, Xtr[(ftr != "genuine") & keep])
        thr_s = float(np.quantile(clf.predict_proba(Xcal)[:, 1], 1 - alpha / 2))
        p = clf.predict_proba(Xte[fte == f])[:, 1] > thr_s
        loo[f] = {"anomaly_only": rate(flag_a[fte == f]), "combined_unseen": rate(flag_a[fte == f] | p)}
    res["unseen"] = loo
    res["unseen_macro_combined"] = float(np.mean([loo[f]["combined_unseen"]["rate"] for f in FAMILY_NAMES]))
    res["unseen_macro_anomaly"] = float(np.mean([loo[f]["anomaly_only"]["rate"] for f in FAMILY_NAMES]))

    # ---- C. batch-level Monte Carlo ----------------------------------------------------------
    print("Batch-level Monte Carlo...")
    rng = np.random.default_rng(seed + 9)
    comb_flag = model.suspect(Xte, "combined")
    fsr = res["detectors"]["combined"]["false_suspect_rate"]
    known = res["detectors"]["combined"]["detection_by_family"]
    esc_k = [known[f]["n"] - known[f]["k"] for f in FAMILY_NAMES]
    n_k = [known[f]["n"] for f in FAMILY_NAMES]
    esc_u = [loo[f]["combined_unseen"]["n"] - loo[f]["combined_unseen"]["k"] for f in FAMILY_NAMES]
    n_u = [loo[f]["combined_unseen"]["n"] for f in FAMILY_NAMES]
    # batch_monte_carlo takes "k" as the count of *flagged*; convert to escape counts via complement
    res["batch"] = []
    for q in (0.001, 0.005, 0.01, 0.02, 0.05):
        row = {"contamination": q, "batch_size": 250}
        row["known_families"] = batch_monte_carlo([n - e for n, e in zip(n_k, esc_k)], n_k, fsr["k"], fsr["n"], 250, q, 20000, rng)
        row["unseen_families"] = batch_monte_carlo([n - e for n, e in zip(n_u, esc_u)], n_u, fsr["k"], fsr["n"], 250, q, 20000, rng)
        res["batch"].append(row)

    # ---- D. robustness sweeps (model trained at baseline, tested under shift) --------------------
    print("Noise / calibration-drift sweeps...")
    sweeps = []
    cases = [("noise x0.5", SimConfig(noise_mult=0.5)), ("baseline", SimConfig()), ("noise x2", SimConfig(noise_mult=2.0)),
             ("noise x3", SimConfig(noise_mult=3.0)), ("spectral drift +0.02", SimConfig(spec_offset=0.02)),
             ("spectral drift +0.05", SimConfig(spec_offset=0.05))]
    for i, (name, cfg) in enumerate(cases):
        Xs, fs = make_dataset(plan(S["sweep_g"], S["sweep_f"]), seed + 100 + i, cfg)
        g = fs == "genuine"
        flag = model.suspect(Xs, "combined")
        half = np.where(g)[0]
        cal_idx, eval_idx = half[: len(half) // 2], half[len(half) // 2:]
        m2 = BoltVerifier(alpha); m2.__dict__.update({k: v for k, v in model.__dict__.items()})
        m2.calibrate(Xs[cal_idx])                                   # recalibrated on genuine under the shifted condition
        flag2 = m2.suspect(Xs, "combined")
        imp = ~g
        sweeps.append({"case": name,
                       "fsr_fixed_thresholds": rate(flag[g])["rate"],
                       "macro_detection_fixed": float(np.mean([rate(flag[fs == f])["rate"] for f in FAMILY_NAMES])),
                       "fsr_recalibrated": rate(flag2[eval_idx])["rate"],
                       "macro_detection_recalibrated": float(np.mean([rate(flag2[fs == f])["rate"] for f in FAMILY_NAMES]))})
        print(f"  {name}: done")
    res["sweeps"] = sweeps
    res["runtime_s"] = time.time() - t0

    (out / "results.json").write_text(json.dumps(res, indent=2))
    write_report(out, res)
    try:
        make_plots(out, res, sc, is_gen, comb)
    except ImportError:
        print("matplotlib not installed - skipping plots")
    print(f"\nFinished in {res['runtime_s']:.0f}s.  Output folder: {out.resolve()}")
    c = res["detectors"]["combined"]
    print(f"Combined model: false-suspect {c['false_suspect_rate']['rate']:.2%}, macro detection "
          f"{c['macro_detection']:.1%}, unseen-family macro detection {res['unseen_macro_combined']:.1%}")


# --------------------------------------------------------------------------------------
# Report + plots
# --------------------------------------------------------------------------------------
def pct(r):
    return f"{r['rate']:.1%} ({r['lo']:.1%}-{r['hi']:.1%})"


def write_report(out: Path, r: dict):
    d = r["detectors"]
    L = ["# Bolt verification: simulation reliability report", "",
         f"Target part: **{TARGET['part_number']}** - {TARGET['grade']}, {TARGET['material']}, "
         f"1/2\"-13 x {TARGET['length_in']}\", partially threaded (min thread {TARGET['min_thread_len_in']}\"), "
         f"head {TARGET['head_af_in']}\" AF x {TARGET['head_height_in']}\", {TARGET['head_marks']} radial head marks.", "",
         "> All numbers come from a physics-based **simulation** with assumed sensor noise and part scatter. "
         "They show how the pipeline behaves under those assumptions; validate on measured parts before relying on them.", "",
         f"Target false-suspect rate alpha = {r['config']['alpha']:.1%}. Intervals are Wilson 95%.", "",
         "## 1. Headline (fresh held-out test set)", "",
         "| Detector | False-suspect rate (genuine) | Macro detection | Worst family | Worst detection |", "|---|---|---|---|---|"]
    for k in ("anomaly", "supervised", "combined"):
        L.append(f"| {k} | {pct(d[k]['false_suspect_rate'])} | {d[k]['macro_detection']:.1%} | "
                 f"{d[k]['worst_family']} | {d[k]['worst_detection']:.1%} |")
    L += ["", f"Overall ROC AUC (combined score, genuine vs all impostors): **{r['auc_overall_combined']:.4f}**", "",
          "## 2. Detection by impostor family", "",
          "'Known' = family was in training. 'Unseen' = family removed from training (leave-one-family-out), "
          "so only the genuine-only anomaly detector plus a classifier that never saw that family can catch it.", "",
          "| Family | Known: combined | Unseen: anomaly only | Unseen: combined | AUC (combined) |", "|---|---|---|---|---|"]
    for f in FAMILY_NAMES:
        L.append(f"| {f} | {pct(d['combined']['detection_by_family'][f])} | {pct(r['unseen'][f]['anomaly_only'])} | "
                 f"{pct(r['unseen'][f]['combined_unseen'])} | {r['auc'][f]['combined']:.3f} |")
    L += ["", f"Macro detection: known {d['combined']['macro_detection']:.1%}, "
              f"unseen {r['unseen_macro_combined']:.1%} (anomaly-only {r['unseen_macro_anomaly']:.1%}).", "",
          "## 3. Batch-level outcomes (250 parts per batch, Monte Carlo with test-set uncertainty)", "",
          "Escapes = impostors that reach 'verified'. Extra tests = genuine parts sent to further testing.", "",
          "| Contamination | P(batch has impostor) | Known fams: mean escapes (95%) | P(escape given impostor present) | "
          "Unseen fams: mean escapes (95%) | P(escape given impostor present) | Extra tests / batch (95%) |", "|---|---|---|---|---|---|---|"]
    for b in r["batch"]:
        k, u = b["known_families"], b["unseen_families"]
        L.append(f"| {b['contamination']:.1%} | {k['P_batch_has_impostor']:.1%} | {k['mean_escapes_per_batch']:.3f} "
                 f"({k['escapes_95'][0]:.0f}-{k['escapes_95'][1]:.0f}) | {k['P_escape_given_impostor_present']:.1%} | "
                 f"{u['mean_escapes_per_batch']:.3f} ({u['escapes_95'][0]:.0f}-{u['escapes_95'][1]:.0f}) | "
                 f"{u['P_escape_given_impostor_present']:.1%} | {k['mean_extra_tests_per_batch']:.1f} "
                 f"({k['extra_tests_95'][0]:.0f}-{k['extra_tests_95'][1]:.0f}) |")
    L += ["", "## 4. Robustness to sensor noise and calibration drift", "",
          "Model trained at baseline noise. 'Fixed' keeps the original thresholds; 'recalibrated' re-sets the "
          "thresholds using genuine parts measured under the shifted condition.", "",
          "| Condition | FSR (fixed) | Detection (fixed) | FSR (recal.) | Detection (recal.) |", "|---|---|---|---|---|"]
    for s in r["sweeps"]:
        L.append(f"| {s['case']} | {s['fsr_fixed_thresholds']:.1%} | {s['macro_detection_fixed']:.1%} | "
                 f"{s['fsr_recalibrated']:.1%} | {s['macro_detection_recalibrated']:.1%} |")
    L += ["", "## 5. How to read this", "",
          "- A low detection rate for a family means the sensors, as simulated, carry little information separating it "
          "from the genuine part (for example a spoofed Grade 2 bolt is physically near-identical in stiffness, density and "
          "surface). Fix by adding a sensor or test that measures the difference (e.g. hardness), not by tuning the model.",
          "- 'Unseen' rows are the honest estimate for new kinds of counterfeits.",
          "- Drift rows show why thresholds must be recalibrated on known-good parts (white reference, periodic genuine checks).",
          "- Replace the simulator's assumed scatter and noise with measured values, then re-run the same study on real data."]
    (out / "report.md").write_text("\n".join(L))


def make_plots(out: Path, r: dict, sc, is_gen, comb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(FAMILY_NAMES))
    known = r["detectors"]["combined"]["detection_by_family"]
    k_rate = [known[f]["rate"] for f in FAMILY_NAMES]
    k_err = [[known[f]["rate"] - known[f]["lo"] for f in FAMILY_NAMES], [known[f]["hi"] - known[f]["rate"] for f in FAMILY_NAMES]]
    u = r["unseen"]
    u_rate = [u[f]["combined_unseen"]["rate"] for f in FAMILY_NAMES]
    ax.barh(y - 0.2, k_rate, 0.38, xerr=k_err, label="known family", color="#2e7dd7")
    ax.barh(y + 0.2, u_rate, 0.38, label="unseen family (LOFO)", color="#e0892f")
    ax.set_yticks(y); ax.set_yticklabels(FAMILY_NAMES); ax.invert_yaxis()
    ax.set_xlim(0, 1.02); ax.set_xlabel("Fraction flagged SUSPECT"); ax.legend(loc="lower right")
    ax.set_title("Detection by impostor family (95% Wilson CI on known)")
    fig.tight_layout(); fig.savefig(out / "detection_by_family.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve((~is_gen).astype(int), comb)
    ax.plot(fpr, tpr, lw=2); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False-suspect rate (genuine)"); ax.set_ylabel("Detection rate (impostors)")
    ax.set_title(f"ROC, AUC = {r['auc_overall_combined']:.4f}"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [s["case"] for s in r["sweeps"]]
    ax.plot(names, [s["macro_detection_fixed"] for s in r["sweeps"]], "o-", label="detection, fixed thresholds")
    ax.plot(names, [s["macro_detection_recalibrated"] for s in r["sweeps"]], "s-", label="detection, recalibrated")
    ax.plot(names, [s["fsr_fixed_thresholds"] for s in r["sweeps"]], "^--", label="false-suspect, fixed")
    ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=.3); plt.setp(ax.get_xticklabels(), rotation=20)
    ax.set_title("Robustness to sensor noise / calibration drift")
    fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------------------
# Sample model check
# --------------------------------------------------------------------------------------
def test_sample(model_path: Path):
    model = BoltVerifier.load(model_path)
    rng = np.random.default_rng(12345)
    cfg = SimConfig()
    cases = [("genuine", genuine)] * 4 + [(n, FAMILIES[n]) for n in ("stainless_18_8", "cracked", "grade_2_spoofed",
                                                                    "fine_thread_UNF", "yellow_zinc", "short_length")]
    print(f"{'truth':<18}{'verdict':<10}reasons")
    for name, fn in cases:
        sp, vb, vs = render(fn(rng), rng, cfg)
        out = model.predict(sp, vb, vs)
        print(f"{name:<18}{out['label']:<10}{', '.join(out['reasons']) or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "test"])
    ap.add_argument("--quick", action="store_true", help="smaller study")
    ap.add_argument("--alpha", type=float, default=0.02, help="target false-suspect rate")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / ".bolt_verifier_results")
    a = ap.parse_args()
    if a.command == "run":
        run_study(a.out, a.quick, a.alpha, a.seed)
    else:
        test_sample(a.out / "bolt_verifier.joblib")


if __name__ == "__main__":
    main()