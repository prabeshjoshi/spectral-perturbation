"""Additive background / baseline effects.

These are additive, so they are most natural in the absorbance (or Raman
intensity) domain. All shapes are defined on the x-axis normalised to [-1, 1]
or [0, 1], so coefficients mean the same thing regardless of units.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..base import Perturbation, norm_axis, register, sample, sample_int, unit_axis

__all__ = [
    "ConstantOffset", "LinearBaseline", "PolynomialBaseline", "ExponentialBaseline",
    "SineBaseline", "SplineBaseline", "GaussianHump",
]


@register
class ConstantOffset(Perturbation):
    """Flat additive offset (the MSC intercept term)."""

    def __init__(self, offset=(-0.05, 0.05), **kw):
        super().__init__(**kw)
        self.offset = offset

    def _apply(self, X, x, rng):
        return X + sample(rng, self.offset, (X.shape[0], 1), self.magnitude)


@register
class LinearBaseline(Perturbation):
    """Sloped baseline: offset + slope * w, with w in [-1, 1].
    ``slope`` is the half-rise across the full range."""

    def __init__(self, offset=(-0.05, 0.05), slope=(-0.05, 0.05), **kw):
        super().__init__(**kw)
        self.offset, self.slope = offset, slope

    def _apply(self, X, x, rng):
        n = X.shape[0]
        w = norm_axis(x)[None, :]
        a = sample(rng, self.offset, (n, 1), self.magnitude)
        b = sample(rng, self.slope, (n, 1), self.magnitude)
        return X + a + b * w


@register
class PolynomialBaseline(Perturbation):
    """Random polynomial sum_k c_k * w**k, w in [-1, 1].

    ``coef`` may be one range for every order, or a list of ranges, one per
    order (length order+1), e.g. to keep the constant term small while
    allowing curvature.
    """

    def __init__(self, order=3, coef=(-0.05, 0.05), **kw):
        super().__init__(**kw)
        self.order, self.coef = int(order), coef

    def _apply(self, X, x, rng):
        n = X.shape[0]
        V = np.vander(norm_axis(x), self.order + 1, increasing=True)  # (m, k)
        if isinstance(self.coef, (list, tuple)) and len(self.coef) and not np.isscalar(self.coef[0]):
            if len(self.coef) != self.order + 1:
                raise ValueError("per-order coef list must have length order+1")
            C = np.column_stack([sample(rng, r, n, self.magnitude) for r in self.coef])
        else:
            C = sample(rng, self.coef, (n, self.order + 1), self.magnitude)
        return X + C @ V.T


@register
class ExponentialBaseline(Perturbation):
    """Exponential background, the typical shape of Raman fluorescence.

    Normalised so the background runs from 0 to ``amplitude`` across the
    range; ``rate`` controls curvature (larger = more sharply rising).
    ``direction``: 'up' rises with x, 'down' decays with x, 'random' either.
    """

    def __init__(self, amplitude=(0.05, 0.5), rate=(1.0, 6.0), direction="random", **kw):
        super().__init__(**kw)
        self.amplitude, self.rate, self.direction = amplitude, rate, direction

    def _apply(self, X, x, rng):
        n = X.shape[0]
        u = unit_axis(x)[None, :]
        a = sample(rng, self.amplitude, (n, 1), self.magnitude)
        k = sample(rng, self.rate, (n, 1))
        if self.direction == "up":
            flip = np.zeros((n, 1), bool)
        elif self.direction == "down":
            flip = np.ones((n, 1), bool)
        else:
            flip = rng.random((n, 1)) < 0.5
        u = np.where(flip, 1.0 - u, u)
        shape = np.expm1(k * u) / np.expm1(k)  # 0 -> 1, exactly
        return X + a * shape


@register
class SineBaseline(Perturbation):
    """Sinusoidal ripple: etalon / interference fringes, common in NIR
    transmission through thin films or cuvette windows."""

    def __init__(self, amplitude=(0.005, 0.05), cycles=(2, 20), **kw):
        super().__init__(**kw)
        self.amplitude, self.cycles = amplitude, cycles

    def _apply(self, X, x, rng):
        n = X.shape[0]
        u = unit_axis(x)[None, :]
        a = sample(rng, self.amplitude, (n, 1), self.magnitude)
        f = sample(rng, self.cycles, (n, 1))
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        return X + a * np.sin(2 * np.pi * f * u + ph)


@register
class SplineBaseline(Perturbation):
    """Smooth wandering baseline through randomly placed knots. The most
    general 'something drifted' background."""

    def __init__(self, n_knots=(4, 8), amplitude=(0.02, 0.1), **kw):
        super().__init__(**kw)
        self.n_knots, self.amplitude = n_knots, amplitude

    def _apply(self, X, x, rng):
        n = X.shape[0]
        w = norm_axis(x)
        amps = sample(rng, self.amplitude, n, self.magnitude)
        for i in range(n):
            k = max(2, int(sample_int(rng, self.n_knots)))
            xk = np.linspace(-1, 1, k)
            X[i] += CubicSpline(xk, rng.uniform(-amps[i], amps[i], k), bc_type="natural")(w)
        return X


@register
class GaussianHump(Perturbation):
    """Broad Gaussian hump under the spectrum (fluorescence envelope, broad
    water or matrix band). Width is a fraction of the axis span."""

    def __init__(self, amplitude=(0.05, 0.3), width_frac=(0.1, 0.4), **kw):
        super().__init__(**kw)
        self.amplitude, self.width_frac = amplitude, width_frac

    def _apply(self, X, x, rng):
        n = X.shape[0]
        span = x.max() - x.min()
        a = sample(rng, self.amplitude, (n, 1), self.magnitude)
        s = sample(rng, self.width_frac, (n, 1)) * span
        c = rng.uniform(x.min(), x.max(), (n, 1))
        return X + a * np.exp(-0.5 * ((x[None, :] - c) / s) ** 2)
