"""Band-level effects: spurious peaks, intensity changes, and line broadening."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..base import Perturbation, register, row_span, sample, sample_gain, sample_int

__all__ = ["AddPeaks", "IntensityScale", "Broadening", "gaussian", "lorentzian", "pseudo_voigt"]

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian(x, centre, fwhm):
    return np.exp(-0.5 * ((x - centre) / (fwhm * _FWHM_TO_SIGMA)) ** 2)


def lorentzian(x, centre, fwhm):
    g = fwhm / 2.0
    return g ** 2 / ((x - centre) ** 2 + g ** 2)


def pseudo_voigt(x, centre, fwhm, eta):
    """eta=0 pure Gaussian, eta=1 pure Lorentzian. All shapes peak at 1."""
    return eta * lorentzian(x, centre, fwhm) + (1.0 - eta) * gaussian(x, centre, fwhm)


@register
class AddPeaks(Perturbation):
    """Inject spurious bands (contaminant, unmodelled constituent).

    amplitude : relative to each spectrum's peak-to-peak range
    fwhm_frac : FWHM as a fraction of the axis span
    shape     : 'gaussian' | 'lorentzian' | 'voigt' (pseudo-Voigt, eta sampled)
    centres   : optional (low, high) in axis units to restrict where peaks land
    """

    def __init__(self, n_peaks=(1, 3), amplitude=(0.05, 0.3), fwhm_frac=(0.005, 0.03),
                 shape="voigt", eta=(0.0, 1.0), centres=None, **kw):
        super().__init__(**kw)
        if shape not in ("gaussian", "lorentzian", "voigt"):
            raise ValueError("shape must be 'gaussian', 'lorentzian' or 'voigt'")
        self.n_peaks, self.amplitude, self.fwhm_frac = n_peaks, amplitude, fwhm_frac
        self.shape, self.eta, self.centres = shape, eta, centres

    def _apply(self, X, x, rng):
        n = X.shape[0]
        span_x = x.max() - x.min()
        lo, hi = self.centres if self.centres is not None else (x.min(), x.max())
        span_y = row_span(X)[:, 0]
        for i in range(n):
            k = int(sample_int(rng, self.n_peaks))
            if k == 0:
                continue
            c = rng.uniform(lo, hi, (k, 1))
            fwhm = sample(rng, self.fwhm_frac, (k, 1)) * span_x
            a = sample(rng, self.amplitude, (k, 1), self.magnitude) * span_y[i]
            if self.shape == "gaussian":
                shp = gaussian(x[None, :], c, fwhm)
            elif self.shape == "lorentzian":
                shp = lorentzian(x[None, :], c, fwhm)
            else:
                shp = pseudo_voigt(x[None, :], c, fwhm, sample(rng, self.eta, (k, 1)))
            X[i] += (a * shp).sum(0)
        return X


@register
class IntensityScale(Perturbation):
    """Global intensity gain (laser power, integration time, concentration)."""

    def __init__(self, gain=(0.8, 1.2), **kw):
        super().__init__(**kw)
        self.gain = gain

    def _apply(self, X, x, rng):
        return X * sample_gain(rng, self.gain, (X.shape[0], 1), self.magnitude)


@register
class Broadening(Perturbation):
    """Convolution with a Gaussian instrument line shape (resolution loss,
    thermal broadening). ``fwhm`` is in channels."""

    def __init__(self, fwhm=(1.0, 5.0), **kw):
        super().__init__(**kw)
        self.fwhm = fwhm

    def _apply(self, X, x, rng):
        sig = sample(rng, self.fwhm, X.shape[0], self.magnitude) * _FWHM_TO_SIGMA
        for i, s in enumerate(sig):
            if s > 1e-3:
                X[i] = gaussian_filter1d(X[i], s, mode="nearest")
        return X
