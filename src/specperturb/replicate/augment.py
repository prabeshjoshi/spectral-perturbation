"""Augmentation with real replicate differences: ``x_new = x + scale * d``.

Two interfaces to the same operation:

* :class:`ReplicateAugmenter` - scikit-learn transformer, fitted on replicate scans.
* :class:`ReplicateNoise`     - a specperturb ``Perturbation``, so it composes with
  the physics-based perturbations (``Compose([ReplicateNoise(...), GaussianNoise()])``)
  and plugs into :class:`specperturb.AugmentedEstimator`.

Rules
-----
1. Augment RAW spectra, then preprocess the augmented set (and test spectra with the
   same preprocessing). The library must be in the same representation as the
   spectra you perturb.
2. Cross-validate group-aware after augmentation: copies of one sample must stay in
   the same fold. :class:`specperturb.AugmentedEstimator` does this automatically by
   augmenting only inside ``fit``.
3. When validating, the replicate-library samples must be disjoint from the test
   samples.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import Perturbation, register
from .core import ReplicateNoiseModel

__all__ = ["ReplicateAugmenter", "ReplicateNoise"]


class ReplicateAugmenter(TransformerMixin, BaseEstimator):
    """Augment spectra with real replicate differences: ``x_new = x + scale * d``.

    Parameters
    ----------
    scale : float, default=1.0
        Multiplier on the sampled differences.
    exclude_outliers : bool, default=True
        Drop gross bad scans when building the library.
    random_state : int, Generator or None
        Seeds the generator created at ``fit``; successive ``transform`` calls
        continue that stream.

    Notes
    -----
    ``transform`` perturbs every row it is given, so do not put this class in a
    Pipeline that is also used for prediction: it would perturb test spectra too.
    Use :meth:`augment` for a manual workflow, or
    :class:`specperturb.AugmentedEstimator` with :class:`ReplicateNoise` inside
    model selection.

    Examples
    --------
    >>> aug = ReplicateAugmenter(random_state=0).fit(X_rep, groups=rep_ids)
    >>> Xa, ya, ga = aug.augment(X_cal, y_cal, n_copies=15)
    >>> # preprocess Xa, then cross-validate with GroupKFold(groups=ga)
    """

    def __init__(self, scale=1.0, exclude_outliers=True, random_state=None):
        self.scale = scale
        self.exclude_outliers = exclude_outliers
        self.random_state = random_state

    def fit(self, X, y=None, groups=None):
        """Fit the library on replicate scans. ``groups`` is required."""
        self.noise_ = ReplicateNoiseModel(self.exclude_outliers).fit(X, groups=groups)
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = self.noise_.n_features_in_
        return self

    def transform(self, X):
        """Return ``X`` with one random library difference added to each row."""
        check_is_fitted(self, "noise_")
        X = np.asarray(X, float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} channels; library has {self.n_features_in_}")
        return X + self.scale * self.noise_.sample(len(X), self._rng)

    def augment(self, X, y, n_copies=15, groups=None):
        """Originals plus ``n_copies`` perturbed copies.

        Returns
        -------
        X_aug : array of shape (n * (n_copies + 1), n_channels)
            Originals first, then each block of copies.
        y_aug : labels tiled to match.
        groups_aug : group IDs tiled to match (row index if ``groups`` is None).
            Pass these to a group-aware splitter.
        """
        X = np.asarray(X, float)
        y = np.asarray(y)
        g = np.arange(len(X)) if groups is None else np.asarray(groups)
        Xa = [X] + [self.transform(X) for _ in range(n_copies)]
        return np.vstack(Xa), np.tile(y, n_copies + 1), np.tile(g, n_copies + 1)


@register
class ReplicateNoise(Perturbation):
    """Perturbation that adds a real replicate difference to each spectrum.

    The library is supplied either in the constructor (``replicates`` + ``groups``)
    or later through :meth:`fit`, which :class:`specperturb.AugmentedEstimator`
    calls on each training fold when it receives ``groups``.

    Parameters
    ----------
    replicates : array (n_scans, n_channels), optional
        Replicate scans, in the same representation as the spectra to perturb.
    groups : array (n_scans,), optional
        Physical-sample ID of each replicate scan.
    scale : float, default=1.0
        Multiplier on the differences; ``magnitude`` multiplies it further.
    exclude_outliers : bool, default=True
    p, magnitude, rng : see :class:`specperturb.Perturbation`.

    Examples
    --------
    >>> aug = sp.Compose([sp.ReplicateNoise(X_rep, rep_ids), sp.GaussianNoise(std=1e-4)])
    >>> X_new = aug(X_cal, rng=0)
    """

    def __init__(self, replicates=None, groups=None, scale=1.0, exclude_outliers=True, **kw):
        super().__init__(**kw)
        self.replicates = replicates
        self.groups = groups
        self.scale = scale
        self.exclude_outliers = exclude_outliers
        self._fitted_model = None
        self._cache = (None, None)

    def fit(self, X, y=None, groups=None):
        """Build the library from replicate scans ``X`` with sample IDs ``groups``.

        Used when the library comes from the data (e.g. per training fold inside
        :class:`specperturb.AugmentedEstimator`); not allowed when a library was
        already given in the constructor.
        """
        if self.replicates is not None:
            raise ValueError("Library already given in the constructor; do not also call fit(groups=...).")
        self._fitted_model = ReplicateNoiseModel(self.exclude_outliers).fit(X, groups=groups)
        return self

    @property
    def noise_model_(self):
        """The fitted :class:`ReplicateNoiseModel`, or None if there is no library."""
        if self._fitted_model is not None:
            return self._fitted_model
        if self.replicates is None:
            return None
        key = (id(self.replicates), id(self.groups), self.exclude_outliers)
        if self._cache[0] != key:
            if self.groups is None:
                raise ValueError("groups is required when replicates is given.")
            self._cache = (key, ReplicateNoiseModel(self.exclude_outliers).fit(self.replicates, groups=self.groups))
        return self._cache[1]

    def _apply(self, X, x, rng):
        model = self.noise_model_
        if model is None:
            raise RuntimeError("ReplicateNoise has no library: pass replicates/groups or call fit().")
        D = model.differences_
        if D.shape[1] != X.shape[1]:
            raise ValueError(f"X has {X.shape[1]} channels; replicate library has {D.shape[1]}")
        return X + (self.scale * self.magnitude) * D[rng.integers(0, len(D), X.shape[0])]
