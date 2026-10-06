"""Time profiles theta(t). Composable with + and * (and scalar multiplication).

Each profile knows its ``onset`` (time the drift starts) or None if it is
in-control variation present from the start (e.g. AR(1) noise, an always-on
day/night cycle). Composite onsets: a sum starts at the earliest component
onset; a product at the latest.

    theta = Saturating(onset=50, size=0.02, tau=200) + AR1(phi=0.9, sd=0.001)
    theta = Step(100, 1.0) * Periodic(amplitude=0.5, period=24, offset=1.0)
"""
from __future__ import annotations

import numpy as np

__all__ = ["Profile", "Constant", "Step", "Ramp", "Saturating", "Periodic",
           "RandomWalk", "Intermittent", "AR1"]


class Profile:
    onset = None

    def __call__(self, t, rng=None):
        t = np.asarray(t, float)
        return self._eval(t, np.random.default_rng(rng))

    def _eval(self, t, rng):
        raise NotImplementedError

    def __add__(self, other):
        return _Sum(self, _as_profile(other))

    __radd__ = __add__

    def __mul__(self, other):
        return _Prod(self, _as_profile(other))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_profile(other))


def _as_profile(v):
    return v if isinstance(v, Profile) else Constant(float(v))


def _min_onset(*ps):
    o = [p.onset for p in ps if p.onset is not None]
    return min(o) if o else None


class _Sum(Profile):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.onset = _min_onset(a, b)

    def _eval(self, t, rng):
        return self.a._eval(t, rng) + self.b._eval(t, rng)

    def __repr__(self):
        return f"({self.a!r} + {self.b!r})"


class _Prod(Profile):
    def __init__(self, a, b):
        self.a, self.b = a, b
        o = [p.onset for p in (a, b) if p.onset is not None]
        self.onset = max(o) if o else None

    def _eval(self, t, rng):
        return self.a._eval(t, rng) * self.b._eval(t, rng)

    def __repr__(self):
        return f"({self.a!r} * {self.b!r})"


class Constant(Profile):
    def __init__(self, value):
        self.value = value

    def _eval(self, t, rng):
        return np.full_like(t, self.value)

    def __repr__(self):
        return repr(self.value)


class Step(Profile):
    """0 before ``onset``, ``size`` from onset on."""

    def __init__(self, onset, size):
        self.onset, self.size = onset, size

    def _eval(self, t, rng):
        return np.where(t >= self.onset, float(self.size), 0.0)

    def __repr__(self):
        return f"Step(onset={self.onset}, size={self.size})"


class Ramp(Profile):
    """Linear from ``onset`` at ``rate`` per time unit; held at ``cap`` if given."""

    def __init__(self, onset, rate, cap=None):
        self.onset, self.rate, self.cap = onset, rate, cap

    def _eval(self, t, rng):
        v = np.clip(t - self.onset, 0, None) * self.rate
        if self.cap is not None:
            v = np.clip(v, -abs(self.cap), abs(self.cap))
        return v

    def __repr__(self):
        return f"Ramp(onset={self.onset}, rate={self.rate}, cap={self.cap})"


class Saturating(Profile):
    """``size * (1 - exp(-(t - onset) / tau))`` after onset (e.g. lamp aging)."""

    def __init__(self, onset, size, tau):
        self.onset, self.size, self.tau = onset, size, tau

    def _eval(self, t, rng):
        d = np.clip(t - self.onset, 0, None)
        return self.size * -np.expm1(-d / self.tau)

    def __repr__(self):
        return f"Saturating(onset={self.onset}, size={self.size}, tau={self.tau})"


class Periodic(Profile):
    """``offset + amplitude * sin(2 pi (t - phase) / period)``.

    onset=None: present from the start (in-control cycle, e.g. day/night).
    With an onset, the cycle is zero before it."""

    def __init__(self, amplitude, period, phase=0.0, offset=0.0, onset=None):
        self.amplitude, self.period, self.phase = amplitude, period, phase
        self.offset, self.onset = offset, onset

    def _eval(self, t, rng):
        v = self.offset + self.amplitude * np.sin(2 * np.pi * (t - self.phase) / self.period)
        return v if self.onset is None else np.where(t >= self.onset, v, 0.0)

    def __repr__(self):
        return f"Periodic(amplitude={self.amplitude}, period={self.period}, onset={self.onset})"


class RandomWalk(Profile):
    """Gaussian random walk starting at 0 at ``onset``; ``step_sd`` per unit time
    (increments scale with sqrt(dt) for uneven sampling)."""

    def __init__(self, onset, step_sd):
        self.onset, self.step_sd = onset, step_sd

    def _eval(self, t, rng):
        v = np.zeros_like(t)
        after = t >= self.onset
        if after.any():
            ta = t[after]
            dt = np.diff(np.concatenate([[self.onset], ta]))
            v[after] = np.cumsum(rng.normal(0, self.step_sd, len(ta)) * np.sqrt(np.clip(dt, 0, None)))
        return v

    def __repr__(self):
        return f"RandomWalk(onset={self.onset}, step_sd={self.step_sd})"


class Intermittent(Profile):
    """On/off episodes after ``onset`` (bubbles, intermittent contact, a valve).

    Episodes start as a Poisson process with ``rate`` per time unit and last
    ``duration``; theta = ``size`` during an episode, 0 otherwise."""

    def __init__(self, onset, rate, duration, size):
        self.onset, self.rate, self.duration, self.size = onset, rate, duration, size

    def _eval(self, t, rng):
        v = np.zeros_like(t)
        if not len(t) or t.max() < self.onset:
            return v
        horizon = t.max() - self.onset
        n = rng.poisson(self.rate * horizon) if horizon > 0 else 0
        starts = self.onset + rng.uniform(0, horizon, n)
        for s in starts:
            v[(t >= s) & (t < s + self.duration)] = self.size
        return v

    def __repr__(self):
        return f"Intermittent(onset={self.onset}, rate={self.rate}, duration={self.duration}, size={self.size})"


class AR1(Profile):
    """Stationary AR(1) noise, x_k = phi x_{k-1} + e_k, with marginal sd ``sd``.
    In-control variation (onset None). Indexed by sample, not time."""

    def __init__(self, phi, sd):
        if not -1 < phi < 1:
            raise ValueError("phi must be in (-1, 1)")
        self.phi, self.sd = phi, sd

    def _eval(self, t, rng):
        n = len(t)
        e = rng.normal(0, self.sd * np.sqrt(1 - self.phi ** 2), n)
        v = np.empty(n)
        prev = rng.normal(0, self.sd)
        for k in range(n):
            prev = self.phi * prev + e[k]
            v[k] = prev
        return v

    def __repr__(self):
        return f"AR1(phi={self.phi}, sd={self.sd})"
