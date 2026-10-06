"""Render one panel per registered perturbation on a synthetic NIR spectrum.

    python examples/gallery.py   ->  examples/gallery.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import specperturb as sp

wn = np.linspace(4000, 10000, 700)


def band(c, w, a):
    return a * np.exp(-0.5 * ((wn - c) / w) ** 2)


x0 = band(5200, 120, 0.8) + band(6900, 90, 0.5) + band(8300, 200, 0.35) + band(4600, 70, 0.4) + 0.05
X = np.tile(x0, (3, 1))

_reps = np.vstack([x0 + 0.04 * k * (wn - wn.mean()) / np.ptp(wn) + 0.03 * np.sin(k)
                   for k in range(6)])
extra = {
    "Temperature": dict(delta_T=(5, 10), shift_per_degree=-8.0, broadening_per_degree=10.0,
                        region=(6500, 7400), taper=100),
    "StrayLight": dict(fraction=(0.05, 0.1)),
    "ReplicateNoise": dict(replicates=_reps, groups=np.repeat([0, 1, 2], 2), scale=2.0),
    "AddInterferent": dict(interferents=band(7200, 40, 1.0), coef=(0.1, 0.3)),
    "PowerLawScatter": dict(axis_unit="cm-1"),
    "AxisShift": dict(shift=(40, 80)),
    "AxisWarp": dict(amplitude=(30, 60)),
    "AxisStretch": dict(stretch=(0.01, 0.02)),
    "Broadening": dict(fwhm=(15, 30)),
    "GaussianNoise": dict(std=(0.01, 0.02)),
    "ColoredNoise": dict(std=(0.02, 0.03)),
    "DeadPixels": dict(frac=(0.05, 0.1), mode="zero"),
    "Quantization": dict(bits=(4, 5)),
    "PoissonNoise": dict(counts=(200, 500)),
}

names = sp.available()
cols = 4
rows = int(np.ceil((len(names) + 1) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True)
axes = axes.ravel()

for ax, name in zip(axes, names):
    out = sp.get(name, **extra.get(name, {}))(X, x=wn, rng=3)
    ax.plot(wn, x0, color="0.7", lw=1.4)
    for row in out:
        ax.plot(wn, row, lw=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_yticks([])

pipe = sp.Compose([
    sp.EMSCScatter(order=2),
    sp.RandomApply(sp.SineBaseline(), p=0.5),
    sp.AxisShift(shift=(-20, 20)),
    sp.GaussianNoise(std=0.004),
])
ax = axes[len(names)]
ax.plot(wn, x0, color="0.7", lw=1.4)
for row in pipe(X, x=wn, rng=3):
    ax.plot(wn, row, lw=0.8)
ax.set_title("Compose pipeline", fontsize=10, fontweight="bold")
ax.set_yticks([])

for ax in axes[len(names) + 1:]:
    ax.axis("off")
for ax in axes[-cols:]:
    ax.set_xlabel("wavenumber (cm$^{-1}$)", fontsize=8)

fig.tight_layout()
out_path = Path(__file__).with_name("gallery.png")
fig.savefig(out_path, dpi=100)
print("wrote", out_path)
