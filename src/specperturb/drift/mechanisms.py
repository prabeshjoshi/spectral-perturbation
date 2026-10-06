"""Drift mechanisms: each has ONE physical parameter theta, with theta = 0 the
identity. A mechanism applies a specperturb perturbation with its parameter
fixed at theta, so a time profile theta(t) gives exact ground truth per spectrum.

Mechanisms carry a physical ``stage``; :class:`~specperturb.drift.DriftScenario`
applies them in stage order (sample -> interface -> optics -> stray light ->
baseline/source -> detector noise), because several are non-linear and the
order changes the result.

All absorbance-domain unless stated.
"""
from __future__ import annotations

import numpy as np

from .. import perturbations as P

__all__ = [
    "Mechanism", "Offset", "Slope", "Curvature", "Gain", "WavelengthShift",
    "WavelengthStretch", "Bandwidth", "NoiseIncrease", "StrayLightDrift",
    "TemperatureDrift", "Fouling", "Interferent", "ScatterChange", "STAGES",
]

STAGES = {"sample": 0, "interface": 1, "optics": 2, "stray_light": 3, "source": 4, "detector": 5}


class Mechanism:
    """Base class. Subclasses implement ``perturbation(theta)`` returning a
    specperturb Perturbation that applies exactly ``theta``, or override
    ``apply`` directly.

    Attributes
    ----------
    name : str          used as the key for theta and ground truth
    parameter : str     what theta means, with units
    stage : str         one of STAGES
    stochastic : bool   True if the effect is random given theta (noise); such
                        mechanisms are excluded from the "systematic" stream used
                        for true prediction bias
    surrogate : bool    True for empirically derived drift (see empirical.py)
    """

    parameter = "theta"
    stage = "source"
    stochastic = False
    surrogate = False

    def __init__(self, name=None):
        self.name = name or type(self).__name__

    def perturbation(self, theta):
        raise NotImplementedError

    def apply(self, X, theta, x=None, rng=None):
        """Apply to rows of X with per-row parameter ``theta`` (shape (n,))."""
        X = np.array(X, dtype=float)
        theta = np.broadcast_to(np.asarray(theta, float), (X.shape[0],))
        rng = np.random.default_rng(rng)
        active = theta != 0
        for value in np.unique(theta[active]):
            rows = np.where(theta == value)[0]
            X[rows] = self.perturbation(float(value))(X[rows], x=x, rng=rng)
        return X

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, theta={self.parameter!r})"


class Offset(Mechanism):
    """Flat baseline offset."""
    parameter = "offset (absorbance units)"

    def perturbation(self, theta):
        return P.ConstantOffset(offset=theta)


class Slope(Mechanism):
    """Linear baseline tilt, pivoting at the axis centre."""
    parameter = "half-rise across the axis (absorbance units)"

    def perturbation(self, theta):
        return P.LinearBaseline(offset=0.0, slope=theta)


class Curvature(Mechanism):
    """Quadratic baseline bow: theta * w**2, w in [-1, 1] (zero at centre, theta at the ends)."""
    parameter = "bow at the axis ends relative to centre (absorbance units)"

    def perturbation(self, theta):
        return P.PolynomialBaseline(order=2, coef=[(0.0, 0.0), (0.0, 0.0), (theta, theta)])


class Gain(Mechanism):
    """Source intensity change. In absorbance with a stale reference this is a
    flat offset of -log10(1 + theta); set domain='intensity' for raw intensity."""
    parameter = "fractional intensity change (e.g. -0.1 = 10 % drop)"

    def __init__(self, domain="absorbance", name=None):
        super().__init__(name)
        self.domain = domain

    def perturbation(self, theta):
        return P.LampIntensity(change=theta, domain=self.domain)


class WavelengthShift(Mechanism):
    """Rigid wavelength / wavenumber shift."""
    parameter = "shift (axis units)"
    stage = "optics"

    def perturbation(self, theta):
        return P.AxisShift(shift=theta)


class WavelengthStretch(Mechanism):
    """Linear dispersion change about the axis centre."""
    parameter = "fractional stretch (1e-3 = 0.1 %)"
    stage = "optics"

    def perturbation(self, theta):
        return P.AxisStretch(stretch=theta)


class Bandwidth(Mechanism):
    """Resolution loss: extra Gaussian instrument line shape. theta >= 0."""
    parameter = "extra FWHM (channels)"
    stage = "optics"

    def perturbation(self, theta):
        if theta < 0:
            raise ValueError("Bandwidth theta must be >= 0 (resolution can only be lost)")
        return P.Broadening(fwhm=theta)


class NoiseIncrease(Mechanism):
    """Additional white detector noise. Random given theta."""
    parameter = "added noise sd (absorbance units)"
    stage = "detector"
    stochastic = True

    def perturbation(self, theta):
        return P.GaussianNoise(std=abs(theta))


class StrayLightDrift(Mechanism):
    """Growing stray light: non-linear compression at high absorbance."""
    parameter = "stray-light fraction of incident intensity"
    stage = "stray_light"

    def perturbation(self, theta):
        return P.StrayLight(fraction=theta)


class TemperatureDrift(Mechanism):
    """Sample temperature change: shift + broadening, optionally in a region
    (e.g. OH / water bands). Coefficients are required; see
    :class:`specperturb.Temperature`."""
    parameter = "temperature change (degrees)"
    stage = "sample"

    def __init__(self, shift_per_degree, broadening_per_degree=0.0, region=None, taper=0.0, name=None):
        super().__init__(name)
        self.kw = dict(shift_per_degree=shift_per_degree, broadening_per_degree=broadening_per_degree,
                       region=region, taper=taper)

    def perturbation(self, theta):
        return P.Temperature(delta_T=theta, **self.kw)


class _AddSpectrum(Mechanism):
    def __init__(self, spectrum, name=None):
        super().__init__(name)
        self.spectrum = np.asarray(spectrum, float).ravel()

    def perturbation(self, theta):
        return P.AddInterferent(self.spectrum, coef=theta)


class Fouling(_AddSpectrum):
    """Probe / window fouling: a film spectrum (absorbance per unit thickness)
    added in proportion to film thickness."""
    parameter = "film thickness (units of the film spectrum)"
    stage = "interface"


class Interferent(_AddSpectrum):
    """A new constituent: its pure spectrum added in proportion to concentration."""
    parameter = "concentration (units of the pure spectrum)"
    stage = "sample"


class ScatterChange(Mechanism):
    """Change in effective path length / scatter: X -> (1 + theta) X, the
    multiplicative term MSC/EMSC correct."""
    parameter = "relative effective path-length change"
    stage = "sample"

    def perturbation(self, theta):
        return P.PathLength(gain=1.0 + theta)
