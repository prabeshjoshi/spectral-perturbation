"""x-axis distortions: wavelength / wavenumber calibration errors between
instruments or over time. Spectra are resampled back onto the original grid
with edge clamping, so no extrapolation artifacts appear at the ends."""
from __future__ import annotations

import numpy as np

from ..base import Perturbation, register, resample, sample, unit_axis

__all__ = ["AxisShift", "AxisStretch", "AxisWarp"]


@register
class AxisShift(Perturbation):
    """Rigid shift of the x-axis. ``shift`` is in axis units (nm or cm-1)."""

    def __init__(self, shift=(-2.0, 2.0), **kw):
        super().__init__(**kw)
        self.shift = shift

    def _apply(self, X, x, rng):
        d = sample(rng, self.shift, X.shape[0], self.magnitude)
        for i in range(X.shape[0]):
            X[i] = resample(x, X[i], x - d[i])
        return X


@register
class AxisStretch(Perturbation):
    """Linear dispersion error: x -> x_c + (1 + s)(x - x_c) about the axis
    centre. ``stretch`` is the fractional scale error (1e-3 = 0.1 %)."""

    def __init__(self, stretch=(-1e-3, 1e-3), **kw):
        super().__init__(**kw)
        self.stretch = stretch

    def _apply(self, X, x, rng):
        s = sample(rng, self.stretch, X.shape[0], self.magnitude)
        xc = 0.5 * (x.min() + x.max())
        for i in range(X.shape[0]):
            X[i] = resample(x, X[i], xc + (x - xc) / (1.0 + s[i]))
        return X


@register
class AxisWarp(Perturbation):
    """Smooth non-linear warp, x -> x + A sin(2 pi f u + phi), u in [0, 1].
    ``amplitude`` is in axis units; ``cycles`` sets how wavy the error is."""

    def __init__(self, amplitude=(0.5, 3.0), cycles=(0.5, 2.0), **kw):
        super().__init__(**kw)
        self.amplitude, self.cycles = amplitude, cycles

    def _apply(self, X, x, rng):
        n = X.shape[0]
        u = unit_axis(x)
        A = sample(rng, self.amplitude, n, self.magnitude)
        f = sample(rng, self.cycles, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        for i in range(n):
            X[i] = resample(x, X[i], x - A[i] * np.sin(2 * np.pi * f[i] * u + ph[i]))
        return X
