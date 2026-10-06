"""Stochastic noise models: detector, shot, drift, and digitisation effects."""
from __future__ import annotations

import numpy as np

from ..base import Perturbation, register, row_rms, row_span, sample, sample_int

__all__ = [
    "GaussianNoise", "HeteroscedasticNoise", "PoissonNoise", "MultiplicativeNoise",
    "ColoredNoise", "CosmicSpikes", "Quantization", "DeadPixels",
]


@register
class GaussianNoise(Perturbation):
    """Additive white Gaussian noise.

    std : absolute standard deviation range, or relative to each spectrum's
          RMS when ``relative=True``.
    """

    def __init__(self, std=(0.001, 0.01), relative=False, **kw):
        super().__init__(**kw)
        self.std, self.relative = std, relative

    def _apply(self, X, x, rng):
        s = sample(rng, self.std, (X.shape[0], 1), self.magnitude)
        if self.relative:
            s = s * row_rms(X)
        return X + rng.standard_normal(X.shape) * s


@register
class HeteroscedasticNoise(Perturbation):
    """Signal-dependent Gaussian noise, std = k * |X|**gamma.
    gamma=0.5 approximates shot-noise scaling without a counts model;
    gamma=1 gives proportional (gain) noise."""

    def __init__(self, k=(0.005, 0.03), gamma=0.5, **kw):
        super().__init__(**kw)
        self.k, self.gamma = k, gamma

    def _apply(self, X, x, rng):
        k = sample(rng, self.k, (X.shape[0], 1), self.magnitude)
        return X + rng.standard_normal(X.shape) * k * np.abs(X) ** self.gamma


@register
class PoissonNoise(Perturbation):
    """Photon shot noise. Each spectrum is scaled so its maximum equals
    ``counts`` photons, Poisson-sampled, and scaled back.

    Only physically meaningful for non-negative intensity data (raw counts,
    transmittance, reflectance), not absorbance. Negative values are clipped.
    ``magnitude`` works by reducing the photon budget: magnitude=0 means an
    effectively infinite budget (no noise).
    """

    def __init__(self, counts=(1e3, 1e5), **kw):
        super().__init__(**kw)
        self.counts = counts

    def _apply(self, X, x, rng):
        if self.magnitude == 0:
            return X
        lo, hi = (self.counts, self.counts) if np.isscalar(self.counts) else self.counts
        c = rng.uniform(lo, hi, (X.shape[0], 1)) / self.magnitude ** 2
        peak = np.clip(X.max(axis=1, keepdims=True), 1e-12, None)
        scale = c / peak
        return rng.poisson(np.clip(X, 0, None) * scale) / scale


@register
class MultiplicativeNoise(Perturbation):
    """Per-channel gain jitter: X * (1 + eps), eps ~ N(0, std)."""

    def __init__(self, std=(0.002, 0.02), **kw):
        super().__init__(**kw)
        self.std = std

    def _apply(self, X, x, rng):
        s = sample(rng, self.std, (X.shape[0], 1), self.magnitude)
        return X * (1.0 + rng.standard_normal(X.shape) * s)


@register
class ColoredNoise(Perturbation):
    """Correlated 1/f**alpha noise (0 white, 1 pink, 2 Brownian drift),
    normalised to the requested std."""

    def __init__(self, std=(0.002, 0.02), alpha=1.0, **kw):
        super().__init__(**kw)
        self.std, self.alpha = std, alpha

    def _apply(self, X, x, rng):
        n, m = X.shape
        f = np.fft.rfftfreq(m)
        f[0] = f[1] if m > 2 else 1.0
        spec = np.fft.rfft(rng.standard_normal((n, m)), axis=1) * f ** (-self.alpha / 2.0)
        noise = np.fft.irfft(spec, n=m, axis=1)
        noise = (noise - noise.mean(1, keepdims=True)) / (noise.std(1, keepdims=True) + 1e-12)
        return X + noise * sample(rng, self.std, (n, 1), self.magnitude)


@register
class CosmicSpikes(Perturbation):
    """Sparse, narrow, positive spikes (cosmic-ray hits on CCD detectors).
    Amplitude is relative to each spectrum's peak-to-peak range."""

    def __init__(self, n_spikes=(1, 5), amplitude=(0.2, 1.0), width=(1, 2), **kw):
        super().__init__(**kw)
        self.n_spikes, self.amplitude, self.width = n_spikes, amplitude, width

    def _apply(self, X, x, rng):
        n, m = X.shape
        span = row_span(X)[:, 0]
        idx = np.arange(m)
        for i in range(n):
            k = int(sample_int(rng, self.n_spikes))
            if k == 0:
                continue
            centres = rng.integers(0, m, k)
            widths = sample_int(rng, self.width, k)
            amps = sample(rng, self.amplitude, k, self.magnitude) * span[i]
            # boxcar spikes, vectorised over k
            mask = np.abs(idx[None, :] - centres[:, None]) <= widths[:, None] - 1
            X[i] += (mask * amps[:, None]).sum(0)
        return X


@register
class Quantization(Perturbation):
    """ADC quantisation to 2**bits levels across each spectrum's range.
    Not severity-scalable in a continuous way; magnitude=0 disables it."""

    def __init__(self, bits=(8, 12), **kw):
        super().__init__(**kw)
        self.bits = bits

    def _apply(self, X, x, rng):
        if self.magnitude == 0:
            return X
        lo = X.min(axis=1, keepdims=True)
        span = row_span(X)
        levels = 2.0 ** sample_int(rng, self.bits, (X.shape[0], 1)) - 1
        return lo + np.round((X - lo) / span * levels) / levels * span


@register
class DeadPixels(Perturbation):
    """Knock out a fraction of channels. mode='interp' repairs them by linear
    interpolation (what most software does); 'nan' leaves NaNs; 'zero' zeros."""

    def __init__(self, frac=(0.001, 0.01), mode="interp", **kw):
        super().__init__(**kw)
        if mode not in ("interp", "nan", "zero"):
            raise ValueError("mode must be 'interp', 'nan' or 'zero'")
        self.frac, self.mode = frac, mode

    def _apply(self, X, x, rng):
        n, m = X.shape
        fracs = sample(rng, self.frac, n, self.magnitude)
        all_idx = np.arange(m)
        for i in range(n):
            k = int(round(fracs[i] * m))
            if k == 0:
                continue
            bad = rng.choice(m, size=min(k, m - 2), replace=False)
            if self.mode == "nan":
                X[i, bad] = np.nan
            elif self.mode == "zero":
                X[i, bad] = 0.0
            else:
                good = np.setdiff1d(all_idx, bad)
                X[i, bad] = np.interp(bad, good, X[i, good])
        return X
