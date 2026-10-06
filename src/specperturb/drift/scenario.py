"""Drift scenarios: mechanisms paired with time profiles, applied to a stream."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mechanisms import STAGES, Mechanism
from .profiles import Profile, _as_profile

__all__ = ["DriftScenario", "DriftResult"]


@dataclass
class DriftResult:
    """A simulated stream with its ground truth.

    X            : drifted spectra (n_steps, n_channels)
    X_clean      : the same spectra without drift
    X_systematic : drift applied without stochastic mechanisms (noise); the
                   reference for true prediction bias
    t            : time of each spectrum
    theta        : {mechanism name: theta per spectrum}
    onset_time   : {mechanism name: declared onset time or None}
    onset        : index of the first spectrum at/after the earliest onset (None
                   if every component is in-control variation)
    surrogate    : {mechanism name: bool}
    source       : {mechanism name: label}
    """

    X: np.ndarray
    X_clean: np.ndarray
    X_systematic: np.ndarray
    t: np.ndarray
    theta: dict
    onset_time: dict
    onset: int | None
    surrogate: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def delta(self):
        """Drift actually added to each spectrum (X - X_clean)."""
        return self.X - self.X_clean

    @property
    def size(self):
        """Euclidean norm of the drift per spectrum."""
        return np.linalg.norm(self.delta, axis=1)


class DriftScenario:
    """A set of (mechanism, profile) pairs.

    Parameters
    ----------
    components : list of (Mechanism, Profile or number)
    order : 'physical' (sort by mechanism stage; stable within a stage) or 'given'
    name : label used in benchmark reports

    Examples
    --------
    >>> from specperturb.drift import mechanisms as M, profiles as T
    >>> lamp_aging = DriftScenario([
    ...     (M.Gain(), T.Saturating(onset=100, size=-0.15, tau=300)),
    ...     (M.NoiseIncrease(), T.Ramp(onset=100, rate=2e-6)),
    ... ], name="lamp aging")
    >>> res = lamp_aging.simulate(X_stream, t=np.arange(len(X_stream)), x=wavelengths, rng=0)
    """

    def __init__(self, components, order="physical", name=None):
        comps = [(m, _as_profile(p)) for m, p in components]
        for m, _ in comps:
            if not isinstance(m, Mechanism):
                raise TypeError(f"{m!r} is not a Mechanism")
            if m.stage not in STAGES:
                raise ValueError(f"{m!r} has unknown stage {m.stage!r}")
        if order == "physical":
            comps = sorted(comps, key=lambda c: STAGES[c[0].stage])
        elif order != "given":
            raise ValueError("order must be 'physical' or 'given'")
        names, seen = [], {}
        for m, _ in comps:
            k = seen.get(m.name, 0)
            seen[m.name] = k + 1
            names.append(m.name if k == 0 else f"{m.name}_{k + 1}")
        self.components = comps
        self.names = names
        self.name = name or " + ".join(names)

    @property
    def onset_time(self):
        o = [p.onset for _, p in self.components if p.onset is not None]
        return min(o) if o else None

    def simulate(self, X_clean, t=None, x=None, rng=None):
        rng = np.random.default_rng(rng)
        X_clean = np.atleast_2d(np.asarray(X_clean, float))
        n = len(X_clean)
        t = np.arange(n, dtype=float) if t is None else np.asarray(t, float)
        if len(t) != n:
            raise ValueError("t must have one entry per spectrum")

        thetas = {nm: np.asarray(p(t, rng), float) for nm, (_, p) in zip(self.names, self.components)}
        X = X_clean.copy()
        X_sys = X_clean.copy()
        for nm, (m, _) in zip(self.names, self.components):
            th = thetas[nm]
            if not np.any(th):
                continue
            sub = np.random.default_rng(rng.integers(2 ** 63))
            X = m.apply(X, th, x=x, rng=sub)
            if not m.stochastic:
                X_sys = m.apply(X_sys, th, x=x, rng=np.random.default_rng(0))

        onset_t = self.onset_time
        onset = None
        if onset_t is not None:
            idx = np.where(t >= onset_t)[0]
            onset = int(idx[0]) if len(idx) else None
        return DriftResult(
            X=X, X_clean=X_clean, X_systematic=X_sys, t=t, theta=thetas,
            onset_time={nm: p.onset for nm, (_, p) in zip(self.names, self.components)},
            onset=onset,
            surrogate={nm: m.surrogate for nm, (m, _) in zip(self.names, self.components)},
            source={nm: getattr(m, "source", "physical") for nm, (m, _) in zip(self.names, self.components)},
        )
