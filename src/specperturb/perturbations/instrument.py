"""Instrument and sample-environment effects that need a physical model:
stray light, lamp intensity with a stale reference, and temperature.

All three are defined for ABSORBANCE input (log10 units). Each also serves as a
drift mechanism in :mod:`specperturb.drift`, where its single physical
parameter follows a time profile.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..base import Perturbation, register, resample, sample

__all__ = ["StrayLight", "LampIntensity", "Temperature", "region_weights"]

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@register
class StrayLight(Perturbation):
    """Stray light: a fraction ``s`` of unabsorbed light reaches the detector.

        T_obs = (T + s) / (1 + s),    A_obs = -log10((10**-A + s) / (1 + s))

    Negligible at low absorbance; compresses (flattens) high-absorbance bands,
    which makes the response non-linear in concentration.

    fraction : stray-light fraction of the incident intensity, e.g. 1e-4 to 1e-2.
    """

    def __init__(self, fraction=(0.0, 0.01), **kw):
        super().__init__(**kw)
        self.fraction = fraction

    def _apply(self, X, x, rng):
        s = np.clip(sample(rng, self.fraction, (X.shape[0], 1), self.magnitude), 0.0, None)
        return -np.log10((10.0 ** (-X) + s) / (1.0 + s))


@register
class LampIntensity(Perturbation):
    """Change in source intensity by a factor (1 + change).

    domain='absorbance': the reference (I0) was collected before the change, so
        A_obs = A - log10(1 + change), i.e. a flat offset in absorbance.
    domain='intensity': raw intensity / single-beam data is scaled by (1 + change).

    change : fractional intensity change, e.g. -0.1 for a 10 % drop.
    """

    def __init__(self, change=(-0.1, 0.1), domain="absorbance", **kw):
        super().__init__(**kw)
        if domain not in ("absorbance", "intensity"):
            raise ValueError("domain must be 'absorbance' or 'intensity'")
        self.change, self.domain = change, domain

    def _apply(self, X, x, rng):
        g = sample(rng, self.change, (X.shape[0], 1), self.magnitude)
        if np.any(g <= -1):
            raise ValueError("change must be > -1")
        return X - np.log10(1.0 + g) if self.domain == "absorbance" else X * (1.0 + g)


def region_weights(x, region=None, taper=0.0):
    """Channel weights in [0, 1]: 1 inside ``region=(lo, hi)`` (axis units), 0
    outside, with cosine tapers of width ``taper`` at the edges. None -> all 1."""
    x = np.asarray(x, float)
    if region is None:
        return np.ones_like(x)
    lo, hi = sorted(region)
    w = ((x >= lo) & (x <= hi)).astype(float)
    if taper > 0:
        for edge, sign in ((lo, -1), (hi, 1)):
            d = sign * (x - edge)  # distance outside the region
            band = (d > 0) & (d < taper)
            w[band] = 0.5 * (1 + np.cos(np.pi * d[band] / taper))
    return w


@register
class Temperature(Perturbation):
    """Temperature change: band shift plus broadening, optionally limited to a
    region (e.g. the OH / water bands).

        X_T = X + w * (broaden(shift(X)) - X)

    with shift = shift_per_degree * dT (axis units) and extra Gaussian FWHM =
    broadening_per_degree * dT (axis units, applied only for dT > 0; band
    narrowing on cooling is not modelled).

    The coefficients are deliberately required: they depend on the matrix and
    band and should come from your own temperature series or the literature.

    delta_T : temperature change range (degrees).
    shift_per_degree : band shift per degree in axis units (sign matters; water
        bands move to shorter wavelength on heating).
    broadening_per_degree : extra FWHM per degree in axis units (>= 0).
    region, taper : where the effect applies (see :func:`region_weights`).
    """

    def __init__(self, delta_T=(-2.0, 2.0), shift_per_degree=None, broadening_per_degree=0.0,
                 region=None, taper=0.0, **kw):
        super().__init__(**kw)
        if shift_per_degree is None:
            raise ValueError("shift_per_degree is required (axis units per degree)")
        if broadening_per_degree < 0:
            raise ValueError("broadening_per_degree must be >= 0")
        self.delta_T, self.shift_per_degree = delta_T, shift_per_degree
        self.broadening_per_degree, self.region, self.taper = broadening_per_degree, region, taper

    def _apply(self, X, x, rng):
        dT = sample(rng, self.delta_T, X.shape[0], self.magnitude)
        w = region_weights(x, self.region, self.taper)
        spacing = np.median(np.abs(np.diff(x))) if len(x) > 1 else 1.0
        for i, d in enumerate(dT):
            if d == 0:
                continue
            row = resample(x, X[i], x - self.shift_per_degree * d)
            fwhm = self.broadening_per_degree * d
            if fwhm > 0:
                row = gaussian_filter1d(row, fwhm * _FWHM_TO_SIGMA / spacing, mode="nearest")
            X[i] = X[i] + w * (row - X[i])
        return X
