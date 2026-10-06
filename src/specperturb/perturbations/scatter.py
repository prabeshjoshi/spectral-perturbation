"""Scatter and path-length effects: the forward models of MSC / EMSC and
wavelength-dependent scattering envelopes.

Multiplicative terms are physical for reflectance, transmittance, or
pseudo-absorbance (log 1/R) where scatter acts as a gain. The forward MSC /
EMSC models are written exactly as the corrections assume them, so applying
MSC / EMSC afterwards should approximately undo them, which is a useful test.
"""
from __future__ import annotations

import numpy as np

from ..base import Perturbation, norm_axis, register, sample, sample_gain

__all__ = ["MSCScatter", "EMSCScatter", "PathLength", "PowerLawScatter"]


@register
class MSCScatter(Perturbation):
    """Forward MSC model: X -> b * X + a, one (a, b) per spectrum."""

    def __init__(self, gain=(0.8, 1.25), offset=(-0.05, 0.05), **kw):
        super().__init__(**kw)
        self.gain, self.offset = gain, offset

    def _apply(self, X, x, rng):
        n = X.shape[0]
        b = sample_gain(rng, self.gain, (n, 1), self.magnitude)
        a = sample(rng, self.offset, (n, 1), self.magnitude)
        return b * X + a


@register
class EMSCScatter(Perturbation):
    """Forward EMSC model:

        X -> b * X + sum_{k=0..order} c_k * w**k  [+ sum_j h_j * interferent_j]

    with w the axis normalised to [-1, 1]. order=0 reduces to MSC.
    ``interferents`` is an optional (n_components, n_channels) array of
    constituent spectra that EMSC would model as known interferents.
    """

    def __init__(self, order=2, gain=(0.9, 1.1), coef=(-0.05, 0.05),
                 interferents=None, interferent_coef=(-0.1, 0.1), **kw):
        super().__init__(**kw)
        self.order, self.gain, self.coef = int(order), gain, coef
        self.interferents = None if interferents is None else np.atleast_2d(np.asarray(interferents, float))
        self.interferent_coef = interferent_coef

    def _apply(self, X, x, rng):
        n = X.shape[0]
        V = np.vander(norm_axis(x), self.order + 1, increasing=True)
        b = sample_gain(rng, self.gain, (n, 1), self.magnitude)
        C = sample(rng, self.coef, (n, self.order + 1), self.magnitude)
        out = b * X + C @ V.T
        if self.interferents is not None:
            if self.interferents.shape[1] != X.shape[1]:
                raise ValueError("interferents must have the same number of channels as X")
            H = sample(rng, self.interferent_coef, (n, self.interferents.shape[0]), self.magnitude)
            out = out + H @ self.interferents
        return out


@register
class PathLength(Perturbation):
    """Pure multiplicative scaling (Beer-Lambert path length or packing
    density). Equivalent to MSCScatter with zero offset."""

    def __init__(self, gain=(0.8, 1.2), **kw):
        super().__init__(**kw)
        self.gain = gain

    def _apply(self, X, x, rng):
        return X * sample_gain(rng, self.gain, (X.shape[0], 1), self.magnitude)


@register
class PowerLawScatter(Perturbation):
    """Wavelength-dependent scattering envelope,

        X -> X * (1 + a * (lambda / lambda_ref) ** -n)

    n ~ 4 is Rayleigh, n ~ 0-2 covers the Mie regime for larger particles.
    ``axis_unit`` tells it whether x is wavelength ('nm') or wavenumber
    ('cm-1'); the latter is converted so the scattering still grows toward
    short wavelengths.
    """

    def __init__(self, amplitude=(0.05, 0.3), exponent=(0.0, 4.0), axis_unit="nm", **kw):
        super().__init__(**kw)
        if axis_unit not in ("nm", "cm-1"):
            raise ValueError("axis_unit must be 'nm' or 'cm-1'")
        self.amplitude, self.exponent, self.axis_unit = amplitude, exponent, axis_unit

    def _apply(self, X, x, rng):
        n = X.shape[0]
        if np.any(x <= 0):
            raise ValueError("PowerLawScatter needs a strictly positive physical axis")
        lam = x if self.axis_unit == "nm" else 1e7 / x
        rel = (lam / np.median(lam))[None, :]
        a = sample(rng, self.amplitude, (n, 1), self.magnitude)
        e = sample(rng, self.exponent, (n, 1))
        return X * (1.0 + a * rel ** (-e))
