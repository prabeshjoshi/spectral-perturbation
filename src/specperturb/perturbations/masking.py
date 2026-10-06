"""Constituent and masking effects: adding known interferent spectra, and
blanking out bands to probe which regions a model depends on."""
from __future__ import annotations

import numpy as np

from ..base import Perturbation, register, sample, sample_int

__all__ = ["AddInterferent", "BandMask"]


@register
class AddInterferent(Perturbation):
    """Add scaled reference spectra (water, CO2, solvent, packaging film).

    interferents : (n_components, n_channels) array on the same axis as X
    coef         : range for each component's weight
    """

    def __init__(self, interferents, coef=(0.0, 0.2), **kw):
        super().__init__(**kw)
        self.interferents = np.atleast_2d(np.asarray(interferents, float))
        self.coef = coef

    def _apply(self, X, x, rng):
        if self.interferents.shape[1] != X.shape[1]:
            raise ValueError("interferents must have the same number of channels as X")
        H = sample(rng, self.coef, (X.shape[0], self.interferents.shape[0]), self.magnitude)
        return X + H @ self.interferents


@register
class BandMask(Perturbation):
    """Replace contiguous bands with a fill value ('mean' uses the spectrum
    mean, 'interp' draws a straight line across the gap, or a float).
    width_frac is the band width as a fraction of the channel count."""

    def __init__(self, n_bands=(1, 3), width_frac=(0.02, 0.08), fill="interp", **kw):
        super().__init__(**kw)
        self.n_bands, self.width_frac, self.fill = n_bands, width_frac, fill

    def _apply(self, X, x, rng):
        n, m = X.shape
        for i in range(n):
            for _ in range(int(sample_int(rng, self.n_bands))):
                w = int(round(sample(rng, self.width_frac, 1, self.magnitude)[0] * m))
                if w < 1:
                    continue
                s = int(rng.integers(0, max(1, m - w)))
                e = min(m, s + w)
                if self.fill == "interp":
                    left = X[i, s - 1] if s > 0 else X[i, e % m]
                    right = X[i, e] if e < m else left
                    X[i, s:e] = np.linspace(left, right, e - s + 2)[1:-1]
                elif self.fill == "mean":
                    X[i, s:e] = X[i].mean()
                else:
                    X[i, s:e] = float(self.fill)
        return X
