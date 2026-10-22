# For Sensor Reading AI Model
# Jackson
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # works without a display; remove if you want pop-up windows
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay,
                             confusion_matrix, mean_absolute_error)
from sklearn.model_selection import train_test_split
 
RNG = np.random.default_rng(42)
OUT_DIR = "demo_output"
os.makedirs(OUT_DIR, exist_ok=True)
 
# =============================================================================
# 1. VIBRATION SENSOR SIMULATION
# =============================================================================
FS = 5000          # sampling rate (Hz)
DURATION = 1.0     # seconds per reading
SHAFT_HZ = 30      # motor shaft speed (30 Hz = 1800 RPM)
VIB_CLASSES = ["Healthy", "Imbalance", "Bearing Fault", "Looseness"]
 
 
def simulate_vibration(condition: str, fs=FS, duration=DURATION, shaft_hz=SHAFT_HZ):
    """Return one simulated accelerometer reading (in g) for a machine condition."""
    t = np.arange(0, duration, 1 / fs)
    # Baseline: small 1x shaft component + sensor noise
    sig = 0.20 * np.sin(2 * np.pi * shaft_hz * t) + RNG.normal(0, 0.08, t.size)
 
    if condition == "Imbalance":
        # Strong 1x component
        sig += RNG.uniform(0.8, 1.3) * np.sin(2 * np.pi * shaft_hz * t + RNG.uniform(0, 6.28))
    elif condition == "Bearing Fault":
        # Periodic impacts exciting a high-frequency structural resonance
        fault_hz = shaft_hz * RNG.uniform(3.4, 3.7)   # outer-race defect frequency
        res_hz = RNG.uniform(1200, 1600)
        impact_times = np.arange(0, duration, 1 / fault_hz)
        for it in impact_times:
            idx = (t >= it)
            decay = np.exp(-(t[idx] - it) * 900)
            sig[idx] += RNG.uniform(0.6, 1.0) * decay * np.sin(2 * np.pi * res_hz * (t[idx] - it))
    elif condition == "Looseness":
        # Harmonics of shaft speed (2x, 3x, 4x ...) + a little clipping-type distortion
        for h in range(2, 7):
            sig += RNG.uniform(0.25, 0.5) / (h ** 0.5) * np.sin(2 * np.pi * shaft_hz * h * t + RNG.uniform(0, 6.28))
        sig += 0.35 * np.sin(2 * np.pi * shaft_hz * 0.5 * t)   # sub-harmonic
    return t, sig
 
 
def vibration_features(sig, fs=FS, shaft_hz=SHAFT_HZ):
    """Classic condition-monitoring features: time domain + frequency domain."""
    rms = np.sqrt(np.mean(sig ** 2))
    peak = np.max(np.abs(sig))
    crest = peak / rms
    kurt = kurtosis(sig)
    f, psd = welch(sig, fs=fs, nperseg=1024)
    total = psd.sum()
 
    def band(lo, hi):
        return psd[(f >= lo) & (f < hi)].sum() / total
 
    def at(freq, w=4):
        return psd[(f >= freq - w) & (f <= freq + w)].sum() / total
 
    return [rms, peak, crest, kurt,
            at(shaft_hz), at(2 * shaft_hz), at(3 * shaft_hz), at(0.5 * shaft_hz),
            band(0, 100), band(100, 500), band(500, 1000), band(1000, 2000)]
 
 
VIB_FEATURE_NAMES = ["RMS", "Peak", "Crest factor", "Kurtosis",
                     "1x shaft", "2x shaft", "3x shaft", "0.5x shaft",
                     "Band 0-100Hz", "Band 100-500Hz", "Band 0.5-1kHz", "Band 1-2kHz"]
 
 
def build_vibration_dataset(n_per_class=150):
    X, y = [], []
    for ci, cond in enumerate(VIB_CLASSES):
        for _ in range(n_per_class):
            _, s = simulate_vibration(cond)
            X.append(vibration_features(s))
            y.append(ci)
    return np.array(X), np.array(y)
 
 
# =============================================================================
# 2. HYPERSPECTRAL SENSOR SIMULATION
# =============================================================================
WAVELENGTHS = np.linspace(400, 1000, 120)   # nm, 120 spectral bands
QUALITY_CLASSES = ["Good", "Marginal", "Defective"]
 
 
def gauss(wl, center, width, amp):
    return amp * np.exp(-0.5 * ((wl - center) / width) ** 2)
 
 
def simulate_spectrum(quality: str, wl=WAVELENGTHS):
    """
    Simulated reflectance spectrum of a product (think produce, grain, or a coated part).
    Physical-ish story: healthy material absorbs strongly near 680nm (pigment) and 970nm
    (water), with a high 'NIR plateau'. Degradation weakens these features and adds
    surface/discoloration effects.
    """
    # Hidden quality score (ground truth): 0-100
    if quality == "Good":
        q = RNG.uniform(75, 100)
    elif quality == "Marginal":
        q = RNG.uniform(45, 75)
    else:
        q = RNG.uniform(5, 45)
    s = q / 100.0   # 0..1
 
    base = 0.08 + 0.45 / (1 + np.exp(-(wl - 720) / 18))          # red-edge step into NIR plateau
    base *= (0.75 + 0.35 * s)                                    # healthier = higher NIR plateau
    base -= gauss(wl, 680, 22, 0.20 * s)                         # pigment absorption (weaker when degraded)
    base -= gauss(wl, 970, 25, 0.12 * s)                         # water absorption
    base += gauss(wl, 600, 40, 0.22 * (1 - s))                   # browning / discoloration bump
    base += gauss(wl, 500, 30, 0.05 * (1 - s))
 
    # Illumination drift + sensor noise
    base *= RNG.normal(1.0, 0.03)
    spectrum = base + RNG.normal(0, 0.008, wl.size)
    return np.clip(spectrum, 0, 1), q
 
 
def build_spectral_dataset(n_per_class=200):
    X, y_cls, y_score = [], [], []
    for ci, qual in enumerate(QUALITY_CLASSES):
        for _ in range(n_per_class):
            sp, q = simulate_spectrum(qual)
            X.append(sp)
            y_cls.append(ci)
            y_score.append(q)
    return np.array(X), np.array(y_cls), np.array(y_score)
 
 
# =============================================================================
# 3. TRAIN THE AI MODELS
# =============================================================================
def train_models():
    print("=" * 70)
    print(" TRAINING AI MODELS ON SIMULATED SENSOR DATA")
    print("=" * 70)
 
    # ---- Vibration -> machine health classifier
    Xv, yv = build_vibration_dataset()
    Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(Xv, yv, test_size=0.25, stratify=yv, random_state=1)
    vib_model = RandomForestClassifier(n_estimators=200, random_state=1).fit(Xv_tr, yv_tr)
    vib_pred = vib_model.predict(Xv_te)
    vib_acc = accuracy_score(yv_te, vib_pred)
    print(f"[Vibration]    Machine-health classifier accuracy : {vib_acc * 100:5.1f}%  "
          f"({len(yv_te)} test readings)")
 
    # ---- Hyperspectral -> quality class + quality score
    Xs, ys_cls, ys_score = build_spectral_dataset()
    pca = PCA(n_components=8).fit(Xs)
    Xs_p = pca.transform(Xs)
    (Xs_tr, Xs_te, yc_tr, yc_te, ysc_tr, ysc_te) = train_test_split(
        Xs_p, ys_cls, ys_score, test_size=0.25, stratify=ys_cls, random_state=1)
    qual_clf = RandomForestClassifier(n_estimators=200, random_state=1).fit(Xs_tr, yc_tr)
    qual_reg = RandomForestRegressor(n_estimators=200, random_state=1).fit(Xs_tr, ysc_tr)
    qc_pred = qual_clf.predict(Xs_te)
    qs_pred = qual_reg.predict(Xs_te)
    qual_acc = accuracy_score(yc_te, qc_pred)
    qual_mae = mean_absolute_error(ysc_te, qs_pred)
    print(f"[Hyperspectral] Quality classifier accuracy       : {qual_acc * 100:5.1f}%  "
          f"({len(yc_te)} test samples)")
    print(f"[Hyperspectral] Quality-score error (MAE, 0-100)  : {qual_mae:5.2f} points")
    print(f"[Hyperspectral] PCA: 120 bands -> 8 components keeping "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}% of variance\n")
 
    return dict(vib_model=vib_model, qual_clf=qual_clf, qual_reg=qual_reg, pca=pca,
                vib_eval=(yv_te, vib_pred, vib_acc),
                qual_eval=(yc_te, qc_pred, qual_acc, ysc_te, qs_pred, qual_mae))
 
 
# =============================================================================
# 4. SENSOR FUSION: ONE DECISION FOR THE LINE
# =============================================================================
def fuse(machine_state, machine_conf, quality_score, quality_class):
    """Simple, explainable rule layer on top of the two AI outputs."""
    if machine_state == "Bearing Fault" and machine_conf > 0.6:
        return "STOP", "Bearing fault detected - schedule maintenance now"
    if machine_state != "Healthy" and quality_class != "Good":
        return "STOP", "Machine anomaly AND quality drop - likely process-induced defects"
    if machine_state != "Healthy":
        return "INSPECT", f"Machine anomaly ({machine_state}) - quality still OK, plan maintenance"
    if quality_class == "Defective":
        return "INSPECT", "Machine healthy but product defective - check upstream material"
    if quality_class == "Marginal":
        return "INSPECT", "Product quality trending down - monitor closely"
    return "RUN", "All systems nominal"
 
 
def live_stream_demo(models, n_steps=12):
    """Simulate a stream of readings where the machine slowly degrades over time."""
    print("=" * 70)
    print(" LIVE SIMULATION: machine degrades over time, AI watches both sensors")
    print("=" * 70)
    print(f"{'Step':>4} | {'Machine state (conf)':<27} | {'Quality (score)':<20} | Decision")
    print("-" * 95)
 
    # Scenario script: healthy -> imbalance creeps in -> bearing fault -> quality drops
    vib_script = ["Healthy"] * 4 + ["Imbalance"] * 2 + ["Looseness"] * 2 + ["Bearing Fault"] * 4
    qual_script = ["Good"] * 5 + ["Marginal"] * 3 + ["Defective"] * 4
 
    history = dict(step=[], rms=[], score=[], decision=[])
    last_vib = last_spec = None
    for i in range(n_steps):
        _, vsig = simulate_vibration(vib_script[i])
        vfeat = np.array(vibration_features(vsig)).reshape(1, -1)
        probs = models["vib_model"].predict_proba(vfeat)[0]
        m_idx = int(np.argmax(probs))
        m_state, m_conf = VIB_CLASSES[m_idx], probs[m_idx]
 
        spec, _ = simulate_spectrum(qual_script[i])
        sp_p = models["pca"].transform(spec.reshape(1, -1))
        q_class = QUALITY_CLASSES[int(models["qual_clf"].predict(sp_p)[0])]
        q_score = float(models["qual_reg"].predict(sp_p)[0])
 
        decision, reason = fuse(m_state, m_conf, q_score, q_class)
        print(f"{i + 1:>4} | {m_state:<14} ({m_conf * 100:3.0f}%)      | "
              f"{q_class:<9} ({q_score:4.0f})    | {decision:<7} {reason}")
 
        history["step"].append(i + 1)
        history["rms"].append(vfeat[0, 0])
        history["score"].append(q_score)
        history["decision"].append(decision)
        last_vib, last_spec = (vsig, vib_script[i]), (spec, qual_script[i])
 
    print()
    return history
 
 
# =============================================================================
# 5. PRESENTATION CHARTS
# =============================================================================
COLORS = {"Healthy": "#2e9e5b", "Imbalance": "#e0a21b", "Bearing Fault": "#d64545",
          "Looseness": "#7b5cc4", "Good": "#2e9e5b", "Marginal": "#e0a21b", "Defective": "#d64545",
          "RUN": "#2e9e5b", "INSPECT": "#e0a21b", "STOP": "#d64545"}
 
 
def chart_vibration_signals():
    fig, axes = plt.subplots(4, 2, figsize=(13, 9))
    fig.suptitle("Vibration Sensor: Raw Signal (left) and Frequency Spectrum (right)",
                 fontsize=15, fontweight="bold")
    for row, cond in enumerate(VIB_CLASSES):
        t, s = simulate_vibration(cond)
        axes[row, 0].plot(t[:600] * 1000, s[:600], color=COLORS[cond], lw=1)
        axes[row, 0].set_ylabel(f"{cond}\n(g)", fontsize=10)
        f, psd = welch(s, fs=FS, nperseg=1024)
        axes[row, 1].semilogy(f, psd, color=COLORS[cond], lw=1.2)
        axes[row, 1].set_xlim(0, 2000)
        axes[row, 0].grid(alpha=.3); axes[row, 1].grid(alpha=.3)
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT_DIR}/1_vibration_signals.png", dpi=150)
    plt.close(fig)
 
 
def chart_spectra():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hyperspectral Sensor: Reflectance Spectra by Quality Grade",
                 fontsize=15, fontweight="bold")
    for qual in QUALITY_CLASSES:
        for k in range(12):
            sp, _ = simulate_spectrum(qual)
            ax[0].plot(WAVELENGTHS, sp, color=COLORS[qual], alpha=.35, lw=1,
                       label=qual if k == 0 else None)
    ax[0].axvline(680, ls="--", c="gray", lw=.8); ax[0].text(684, 0.02, "pigment\n680nm", fontsize=8)
    ax[0].axvline(970, ls="--", c="gray", lw=.8); ax[0].text(900, 0.02, "water\n970nm", fontsize=8)
    ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Reflectance")
    ax[0].legend(); ax[0].grid(alpha=.3)
 
    # Simulated "hyperspectral image": a 2D scan with a defect region, colored by predicted quality
    size = 60
    yy, xx = np.mgrid[0:size, 0:size]
    score_map = 88 + RNG.normal(0, 2.5, (size, size))
    score_map -= 50 * np.exp(-(((xx - 38) ** 2 + (yy - 22) ** 2) / 60))   # defect blob
    score_map -= 25 * np.exp(-(((xx - 14) ** 2 + (yy - 44) ** 2) / 35))   # smaller marginal area
    im = ax[1].imshow(np.clip(score_map, 0, 100), cmap="RdYlGn", vmin=20, vmax=100)
    ax[1].set_title("AI quality map across a scanned item\n(each pixel = one spectrum)")
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=.046, label="Quality score")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT_DIR}/2_hyperspectral.png", dpi=150)
    plt.close(fig)
 
 
def chart_model_performance(models):
    yv_te, vib_pred, vib_acc = models["vib_eval"]
    yc_te, qc_pred, qual_acc, ysc_te, qs_pred, qual_mae = models["qual_eval"]
 
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("AI Model Performance on Held-out Test Data", fontsize=15, fontweight="bold")
 
    ConfusionMatrixDisplay(confusion_matrix(yv_te, vib_pred), display_labels=VIB_CLASSES
                           ).plot(ax=ax[0], cmap="Blues", colorbar=False, xticks_rotation=25)
    ax[0].set_title(f"Machine health (vibration)\nAccuracy {vib_acc * 100:.1f}%")
 
    ConfusionMatrixDisplay(confusion_matrix(yc_te, qc_pred), display_labels=QUALITY_CLASSES
                           ).plot(ax=ax[1], cmap="Greens", colorbar=False)
    ax[1].set_title(f"Product quality (hyperspectral)\nAccuracy {qual_acc * 100:.1f}%")
 
    ax[2].scatter(ysc_te, qs_pred, s=14, alpha=.6, color="#3b6fd4")
    ax[2].plot([0, 100], [0, 100], "k--", lw=1)
    ax[2].set_xlabel("True quality score"); ax[2].set_ylabel("AI-predicted score")
    ax[2].set_title(f"Quality score regression\nMean abs. error {qual_mae:.1f} pts")
    ax[2].grid(alpha=.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT_DIR}/3_model_performance.png", dpi=150)
    plt.close(fig)
 
 
def chart_dashboard(history):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Fused Sensor Dashboard: Machine Health + Product Quality -> Line Decision",
                 fontsize=15, fontweight="bold")
    steps = history["step"]
    ax[0].plot(steps, history["rms"], "-o", color="#d64545")
    ax[0].set_ylabel("Vibration RMS (g)"); ax[0].grid(alpha=.3)
    ax[1].plot(steps, history["score"], "-o", color="#2e9e5b")
    ax[1].axhspan(0, 45, color="#d64545", alpha=.08)
    ax[1].axhspan(45, 75, color="#e0a21b", alpha=.08)
    ax[1].axhspan(75, 100, color="#2e9e5b", alpha=.08)
    ax[1].set_ylabel("Quality score"); ax[1].set_ylim(0, 100); ax[1].grid(alpha=.3)
    for s, d in zip(steps, history["decision"]):
        ax[2].bar(s, 1, color=COLORS[d], width=.85)
        ax[2].text(s, .5, d, ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                   rotation=90)
    ax[2].set_yticks([]); ax[2].set_ylabel("AI decision"); ax[2].set_xlabel("Time step")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT_DIR}/4_fused_dashboard.png", dpi=150)
    plt.close(fig)
 
 
# =============================================================================
if __name__ == "__main__":
    print("\n  SENSOR READING AI MODEL - DEMO (all data is simulated)\n")
    models = train_models()
    history = live_stream_demo(models)
 
    print("Generating presentation charts ...")
    chart_vibration_signals()
    chart_spectra()
    chart_model_performance(models)
    chart_dashboard(history)
    print(f"Done. Charts saved in ./{OUT_DIR}/")
 