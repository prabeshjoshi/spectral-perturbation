"""Replicate library: within-sample scan differences as an empirical nuisance model.

Scanning the same physical sample several times (repack, re-present, rescan on one
instrument) leaves the chemistry unchanged, so the differences between those scans
are a direct, label-free sample of measurement nuisance: packing, scatter, path
length, drift and noise.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

__all__ = ["replicate_differences", "ReplicateNoiseModel"]


def replicate_differences(X, groups, exclude_outliers=True, outlier_factor=5.0):
    """All ordered within-sample pairwise differences ``x_a - x_b`` (a != b).

    Parameters
    ----------
    X : array of shape (n_scans, n_channels)
        Replicate spectra.
    groups : array of shape (n_scans,)
        Physical-sample ID of each scan. Samples need >= 2 scans to contribute.
    exclude_outliers : bool, default=True
        Drop gross bad scans: a scan is excluded when its distance to its sample's
        median spectrum exceeds ``outlier_factor`` times the median of all such
        (non-zero) distances.
    outlier_factor : float, default=5.0

    Returns
    -------
    D : array of shape (n_pairs, n_channels)
        Both ``x_a - x_b`` and ``x_b - x_a`` are included, so D is symmetric about
        zero. A sample with m scans contributes m(m - 1) rows, so samples with more
        replicates carry more weight.

    Raises
    ------
    ValueError
        If no sample has at least two (non-outlier) scans.
    """
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError(f"groups has {len(groups)} entries but X has {len(X)} rows")
    keep = np.ones(len(X), bool)
    if exclude_outliers:
        dist = np.empty(len(X))
        for g in np.unique(groups):
            i = np.where(groups == g)[0]
            dist[i] = np.linalg.norm(X[i] - np.median(X[i], 0), axis=1)
        keep = dist <= outlier_factor * np.median(dist[dist > 0]) if np.any(dist > 0) else keep
    D = []
    for g in np.unique(groups):
        i = np.where((groups == g) & keep)[0]
        for a in i:
            for b in i:
                if a != b:
                    D.append(X[a] - X[b])
    if not D:
        raise ValueError("Need >= 2 (non-outlier) scans for at least one sample.")
    return np.asarray(D)


class ReplicateNoiseModel(BaseEstimator):
    """Empirical nuisance model: the set of real replicate differences.

    Parameters
    ----------
    exclude_outliers : bool, default=True
    outlier_factor : float, default=5.0
        See :func:`replicate_differences`.

    Attributes
    ----------
    differences_ : array of shape (n_pairs, n_channels)
        The replicate differences.
    covariance_ : array of shape (n_channels, n_channels)
        Second moment ``S = D'D / n_pairs``. Because D holds differences of two
        measurements, S is twice the single-measurement noise covariance.
    n_features_in_ : int

    Examples
    --------
    >>> model = ReplicateNoiseModel().fit(X_rep, groups=sample_ids)
    >>> d = model.sample(10, random_state=0)   # 10 real differences
    """

    def __init__(self, exclude_outliers=True, outlier_factor=5.0):
        self.exclude_outliers = exclude_outliers
        self.outlier_factor = outlier_factor

    def fit(self, X, y=None, groups=None):
        """Build the library from replicate scans. ``y`` is ignored; ``groups`` is required."""
        if groups is None:
            raise ValueError("groups (physical-sample ID per scan) is required; pass it as a keyword.")
        self.differences_ = replicate_differences(X, groups, self.exclude_outliers, self.outlier_factor)
        self.covariance_ = self.differences_.T @ self.differences_ / len(self.differences_)
        self.n_features_in_ = self.differences_.shape[1]
        return self

    def sample(self, n, random_state=None):
        """Draw ``n`` real differences with replacement.

        ``random_state`` may be a seed or a ``numpy.random.Generator`` (which is
        advanced in place, so repeated calls give fresh draws).
        """
        check_is_fitted(self, "differences_")
        rng = np.random.default_rng(random_state)
        return self.differences_[rng.integers(0, len(self.differences_), n)]
