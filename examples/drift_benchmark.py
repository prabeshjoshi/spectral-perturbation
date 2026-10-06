"""Drift-monitoring benchmark on synthetic NIR data.

    python examples/drift_benchmark.py

1. Builds a PLS model on synthetic absorbance spectra.
2. Shows one run of a lamp-aging scenario: theta(t), the Q statistic with a
   calibrated EWMA alarm, and the true prediction bias against the harm
   threshold (examples/drift_timeline.png).
3. Benchmarks T2 / Q x Shewhart / EWMA / CUSUM / Page-Hinkley / martingale on
   three scenarios at matched pre-onset false-alarm probability, and prints the
   summary.

Needs: pip install 'specperturb[monitor]' matplotlib
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from specperturb.drift import DriftScenario, harm_onset, model_truth
from specperturb.drift import mechanisms as M
from specperturb.drift import profiles as T
from specperturb.drift.benchmark import MonitoringBenchmark
from specperturb.drift.monitor import EWMA, calibrate, q_residuals

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# --- synthetic absorbance spectra: 3 constituents + noise --------------------
x = np.linspace(1100, 2500, 200)  # nm
bands = np.vstack([np.exp(-0.5 * ((x - c) / w) ** 2) for c, w in [(1450, 40), (1940, 50), (2100, 60)]])


def spectra(n):
    C = rng.uniform(0.2, 1.0, (n, 3))
    return C @ bands + 0.1 + rng.normal(0, 2e-3, (n, len(x))), C[:, 0]


X_cal, y_cal = spectra(100)
X_incontrol, _ = spectra(4000)   # NOT used for the model: control limits
X_pool, _ = spectra(4000)        # NOT used for the model: stream source
model = PLSRegression(3, scale=False).fit(X_cal, y_cal)

ONSET, N_STEPS, HARM = 150, 600, 0.03
film = 0.6 * np.exp(-0.5 * ((x - 1700) / 35) ** 2) + 0.2 * np.exp(-0.5 * ((x - 2300) / 40) ** 2)
scenarios = {
    "no drift": None,
    "probe fouling (ramp)": DriftScenario([(M.Fouling(film), T.Ramp(ONSET, 2e-4))]),
    "lamp aging (saturating)": DriftScenario([(M.Gain(), T.Saturating(ONSET, -0.25, tau=200)),
                                              (M.NoiseIncrease(), T.Ramp(ONSET, 5e-6))]),
    "temperature, OH band": DriftScenario([(M.TemperatureDrift(shift_per_degree=-0.4, broadening_per_degree=0.8,
                                                                region=(1380, 1520), taper=20),
                                            T.Ramp(ONSET, 0.02))]),
}

# --- 1. one run, drawn ------------------------------------------------------
stream = X_pool[rng.integers(0, len(X_pool), N_STEPS)]
res = scenarios["lamp aging (saturating)"].simulate(stream, x=x, rng=1)
truth = model_truth(model, res.X_clean, res.X, res.X_systematic)
h_on = harm_onset(truth.bias_systematic, HARM)

q = q_residuals(model).fit(X_cal)
half = len(X_incontrol) // 2
ewma = EWMA(0.2).setup(q(X_incontrol[:half]))
knob = calibrate(ewma, q(X_incontrol[half:]), criterion="fap", window=ONSET, fap=0.05, n_sim=400, rng=0)
s = q(res.X)
path = ewma.path(s)
alarm = ewma.first_alarm(s, knob)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
t = res.t
fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True, gridspec_kw=dict(hspace=0.35))
for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)

ax = axes[0]
ax.plot(t, 100 * res.theta["Gain"], color=C1, lw=2)
ax.set_title("Drift parameter: lamp intensity change (%)  [noise increase also ramps, not shown]",
             loc="left", fontsize=10, color=INK)

ax = axes[1]
ax.scatter(t, s, s=8, color=C2, alpha=0.35, lw=0, label="Q per spectrum")
ax.plot(t, path, color=C2, lw=2, label="EWMA of Q (λ = 0.2)")
ax.axhline(knob, color=MUTED, lw=1, ls="--")
ax.text(t[-1], knob * 1.15, "limit (5 % false-alarm probability before onset)", va="bottom", ha="right", fontsize=8, color=MUTED)
ax.set_yscale("log")
ax.set_ylim(s.min() * 0.7, max(path.max(), knob) * 3)
ax.set_title("Monitor: Q residual (log scale) with a calibrated EWMA chart", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))

ax = axes[2]
ax.plot(t, truth.bias_systematic, color=C3, lw=2)
ax.axhspan(-HARM, HARM, color=GRID, alpha=0.6, lw=0)
ax.text(t[5], HARM, f" acceptable bias ±{HARM}", va="bottom", fontsize=8, color=MUTED)
ax.set_title("Ground truth: prediction bias of the PLS model", loc="left", fontsize=10, color=INK)
ax.set_xlabel("spectrum index (time)", fontsize=9, color=MUTED)

events = [("drift onset", res.onset), ("alarm", alarm), ("harm onset", h_on)]
for ax in axes:
    for label, idx in events:
        if idx is not None:
            ax.axvline(idx, color=INK if label == "alarm" else MUTED, lw=1, ls="-" if label == "alarm" else ":")
lo, hi = axes[0].get_ylim()
placement = {"drift onset": ("right", 0.30), "alarm": ("left", 0.55), "harm onset": ("left", 0.80)}
for label, idx in events:
    if idx is not None:
        ha, frac = placement[label]
        pad = " " if ha == "left" else ""
        axes[0].text(idx, hi - frac * (hi - lo), f"{pad}{label} (t={idx}){'' if pad else ' '}", ha=ha, va="center",
                     fontsize=8, color=INK, bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1))

lead = None if (alarm is None or h_on is None) else h_on - alarm
fig.suptitle(f"Alarm vs harm: EWMA on Q warned {lead} spectra before the bias became unacceptable"
             if lead is not None and lead > 0 else "Alarm vs harm", x=0.02, ha="left", fontsize=11, color=INK)
out = Path(__file__).with_name("drift_timeline.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out, "| onset", res.onset, "alarm", alarm, "harm", h_on)

# --- 2. benchmark -----------------------------------------------------------
bench = MonitoringBenchmark(model, harm_threshold=HARM, criterion="fap", window=ONSET, fap=0.05,
                           n_sim=300, x=x, random_state=0).fit(X_cal, X_incontrol)
print("realized pre-onset false-alarm probability (target 0.05):")
for k, v in bench.verify_calibration(X_pool, n_runs=300).items():
    print(f"  {k[0]:>2} {k[1]:<12} {v:.3f}")
result = bench.run(scenarios, X_pool, n_steps=N_STEPS, n_runs=40)

cols = ["scenario", "statistic", "detector", "false_alarm_rate", "median_delay", "harm_rate",
        "warned_before_harm", "median_lead", "nuisance_alarm_rate"]
print("\n('no drift' counts alarms over the whole run; the others count alarms before onset)")
print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
for r in result.summary():
    print("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
