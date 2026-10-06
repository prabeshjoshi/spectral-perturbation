"""Core machinery: the Perturbation base class, parameter sampling, and a
registry so new perturbations are discoverable by name.

Adding a new perturbation
-------------------------
Subclass ``Perturbation``, implement ``_apply(self, X, x, rng)``, and decorate
with ``@register``::

    from specperturb.base import Perturbation, register, sample

    @register
    class MyEffect(Perturbation):
        def __init__(self, strength=(0.0, 0.1), **kw):
            super().__init__(**kw)
            self.strength = strength

        def _apply(self, X, x, rng):
            s = sample(rng, self.strength, (X.shape[0], 1), self.magnitude)
            return X + s

``_apply`` always receives a float copy of X with shape (n_samples, n_channels),
the x-axis as a 1-D float array, and a numpy Generator. It must return an
array of the same shape. Everything else (1-D input, per-sample probability,
seeding) is handled by the base class.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np

Range = Union[float, Tuple[float, float]]
RNGLike = Union[None, int, np.random.Generator]

_REGISTRY: Dict[str, Type["Perturbation"]] = {}


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
def register(cls: Type["Perturbation"]) -> Type["Perturbation"]:
    """Class decorator: make a perturbation discoverable via ``get``/``available``."""
    name = cls.__name__
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"A perturbation named {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def available() -> list[str]:
    """Names of all registered perturbations, grouped by module."""
    return sorted(_REGISTRY, key=lambda n: (_REGISTRY[n].__module__, n))


def get(name: str, **params) -> "Perturbation":
    """Instantiate a registered perturbation by class name."""
    try:
        return _REGISTRY[name](**params)
    except KeyError:
        raise KeyError(f"Unknown perturbation {name!r}. Available: {available()}") from None


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def as_range(r: Range) -> Tuple[float, float]:
    """Accept a scalar (fixed value) or a (low, high) pair."""
    if np.isscalar(r):
        return float(r), float(r)
    lo, hi = r
    if lo > hi:
        raise ValueError(f"Range low > high: {r}")
    return float(lo), float(hi)


def sample(rng: np.random.Generator, r: Range, size, magnitude: float = 1.0) -> np.ndarray:
    """Draw uniformly from ``r`` with both bounds multiplied by ``magnitude``.

    Use this for *severity* parameters (amplitudes, std, coefficients) where
    magnitude=0 should mean "no effect". For parameters that describe shape
    rather than severity (frequencies, exponents, widths), call with
    magnitude=1.0.
    """
    lo, hi = as_range(r)
    return rng.uniform(lo * magnitude, hi * magnitude, size=size)


def sample_gain(rng: np.random.Generator, r: Range, size, magnitude: float = 1.0) -> np.ndarray:
    """Like ``sample`` but for multiplicative factors centred on 1:
    magnitude=0 gives exactly 1, magnitude=1 gives the full range."""
    lo, hi = as_range(r)
    lo, hi = 1 + (lo - 1) * magnitude, 1 + (hi - 1) * magnitude
    return rng.uniform(lo, hi, size=size)


def sample_int(rng: np.random.Generator, r, size=None) -> np.ndarray:
    lo, hi = (int(r), int(r)) if np.isscalar(r) else (int(r[0]), int(r[1]))
    return rng.integers(lo, hi + 1, size=size)


def norm_axis(x: np.ndarray) -> np.ndarray:
    """Map x to [-1, 1] (orientation-independent) for stable polynomial terms."""
    x0, x1 = x.min(), x.max()
    return 2.0 * (x - x0) / (x1 - x0 + 1e-12) - 1.0


def unit_axis(x: np.ndarray) -> np.ndarray:
    """Map x to [0, 1]."""
    return (norm_axis(x) + 1.0) / 2.0


def row_rms(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(X ** 2, axis=1, keepdims=True)) + 1e-12


def row_span(X: np.ndarray) -> np.ndarray:
    return np.ptp(X, axis=1, keepdims=True) + 1e-12


def resample(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Monotone (PCHIP) resampling with edge *clamping* instead of extrapolation,
    so axis shifts never create spurious spikes at the ends of the range."""
    from scipy.interpolate import PchipInterpolator

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    x_c = np.clip(x_new, xs[0], xs[-1])
    return PchipInterpolator(xs, ys, extrapolate=False)(x_c)


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #
class Perturbation:
    """Base class for all perturbations.

    Parameters
    ----------
    p : float
        Per-spectrum probability of applying the effect (independent per row).
    magnitude : float in [0, 1]
        Global severity scaler. 0 is the identity; 1 uses the full ranges.
    rng : int | Generator | None
        Default random source; a generator passed at call time overrides it.
    """

    def __init__(self, p: float = 1.0, magnitude: float = 1.0, rng: RNGLike = None):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if magnitude < 0:
            raise ValueError("magnitude must be >= 0")
        self.p = float(p)
        self.magnitude = float(magnitude)
        self.rng = np.random.default_rng(rng)

    # -- to implement ------------------------------------------------------- #
    def _apply(self, X: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    # -- public API --------------------------------------------------------- #
    def __call__(self, X, x=None, rng: RNGLike = None) -> np.ndarray:
        rng = self._resolve_rng(rng)
        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 1
        X2 = np.atleast_2d(X)
        if X2.ndim != 2:
            raise ValueError("X must be 1-D or 2-D (n_samples, n_channels)")
        x = self._resolve_axis(x, X2.shape[1])

        out = self._apply(X2.copy(), x, rng)
        if out.shape != X2.shape:
            raise RuntimeError(f"{type(self).__name__} changed the shape of X")

        if self.p < 1.0:
            keep = rng.random(X2.shape[0]) >= self.p
            out[keep] = X2[keep]
        return out[0] if squeeze else out

    def with_magnitude(self, magnitude: float) -> "Perturbation":
        """Return a shallow copy at a different severity (for sweeps)."""
        import copy

        new = copy.copy(self)
        new.magnitude = float(magnitude)
        return new

    # -- scikit-learn style parameter API ------------------------------------ #
    @classmethod
    def _param_names(cls) -> list:
        """Constructor parameter names, following ``**kw`` forwarding up the MRO
        (a subclass's ``**kw`` reaches ``Perturbation.__init__``)."""
        import inspect

        names = []
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None or klass is object:
                continue
            forwards = False
            for name, prm in inspect.signature(init).parameters.items():
                if prm.kind is prm.VAR_KEYWORD:
                    forwards = True
                elif name != "self" and prm.kind is not prm.VAR_POSITIONAL and name not in names:
                    names.append(name)
            if not forwards:
                break
        return names

    def get_params(self, deep: bool = True) -> dict:
        """Constructor parameters, as scikit-learn's ``clone`` and grid search expect."""
        out = {}
        for name in self._param_names():
            value = getattr(self, name)
            out[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{name}__{k}": v for k, v in value.get_params(deep=True).items()})
        return out

    def set_params(self, **params) -> "Perturbation":
        """Set parameters, including nested ones (``transform__magnitude``)."""
        valid = set(self._param_names())
        nested: dict = {}
        for key, value in params.items():
            head, _, rest = key.partition("__")
            if head not in valid:
                raise ValueError(f"Invalid parameter {head!r} for {type(self).__name__}")
            if rest:
                nested.setdefault(head, {})[rest] = value
            else:
                setattr(self, head, value)
        for head, sub in nested.items():
            getattr(self, head).set_params(**sub)
        return self

    def __repr__(self) -> str:
        kv = ", ".join(f"{k}={v!r}" for k, v in self.get_params(deep=False).items() if k != "rng")
        return f"{type(self).__name__}({kv})"

    # -- internals ---------------------------------------------------------- #
    def _resolve_rng(self, rng: RNGLike) -> np.random.Generator:
        if rng is None:
            return self.rng
        return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    @staticmethod
    def _resolve_axis(x, n_channels: int) -> np.ndarray:
        if x is None:
            return np.arange(n_channels, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.shape != (n_channels,):
            raise ValueError(f"x has shape {x.shape}, expected ({n_channels},)")
        return x
