"""Combinators for building perturbation pipelines. All of them thread a single
random generator through their children, so one seed reproduces the chain."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import Perturbation, RNGLike


class Compose(Perturbation):
    """Apply perturbations in order.

    ``magnitude`` on a Compose multiplies each child's own magnitude, so a
    whole pipeline can be swept from 0 (identity) to 1 with one knob.
    """

    def __init__(self, transforms: Sequence[Perturbation], magnitude: float = 1.0, rng: RNGLike = None):
        super().__init__(magnitude=magnitude, rng=rng)
        self.transforms = transforms  # stored as given (sklearn clone contract)

    def __call__(self, X, x=None, rng: RNGLike = None):
        rng = self._resolve_rng(rng)
        for t in self.transforms:
            t = t.with_magnitude(t.magnitude * self.magnitude) if self.magnitude != 1.0 else t
            X = t(X, x=x, rng=rng)
        return X

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n], magnitude={self.magnitude})"


class OneOf(Perturbation):
    """Pick one child at random (optionally weighted) per call, or apply
    nothing with probability 1 - p."""

    def __init__(self, transforms: Sequence[Perturbation], weights: Optional[Sequence[float]] = None,
                 p: float = 1.0, rng: RNGLike = None):
        super().__init__(p=p, rng=rng)
        self.transforms = transforms
        self.weights = weights

    def __call__(self, X, x=None, rng: RNGLike = None):
        rng = self._resolve_rng(rng)
        if rng.random() >= self.p:
            return np.asarray(X, dtype=float).copy()
        w = None if self.weights is None else np.asarray(self.weights, float) / np.sum(self.weights)
        idx = rng.choice(len(self.transforms), p=w)
        return self.transforms[idx](X, x=x, rng=rng)


class SomeOf(Perturbation):
    """Apply a random subset of k children (k drawn from ``n``), in the
    original order."""

    def __init__(self, transforms: Sequence[Perturbation], n=(1, 2), rng: RNGLike = None):
        super().__init__(rng=rng)
        self.transforms = transforms
        self.n = n

    def __call__(self, X, x=None, rng: RNGLike = None):
        rng = self._resolve_rng(rng)
        lo, hi = (self.n, self.n) if np.isscalar(self.n) else self.n
        k = int(rng.integers(lo, min(hi, len(self.transforms)) + 1))
        chosen = np.sort(rng.choice(len(self.transforms), size=k, replace=False))
        X = np.asarray(X, dtype=float).copy()
        for i in chosen:
            X = self.transforms[i](X, x=x, rng=rng)
        return X


class RandomApply(Perturbation):
    """Apply the child to the whole batch with probability p (contrast with the
    ``p`` argument on individual perturbations, which is per spectrum)."""

    def __init__(self, transform: Perturbation, p: float = 0.5, rng: RNGLike = None):
        super().__init__(p=p, rng=rng)
        self.transform = transform

    def __call__(self, X, x=None, rng: RNGLike = None):
        rng = self._resolve_rng(rng)
        if rng.random() < self.p:
            return self.transform(X, x=x, rng=rng)
        return np.asarray(X, dtype=float).copy()
