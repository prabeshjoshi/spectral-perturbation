"""Preprocessing filters that remove only replicate (nuisance) directions.

Both filters eigendecompose the replicate second moment ``S = D'D / n`` (see
:class:`ReplicateNoiseModel`) and act on spectra as a right-multiplication
``X -> X F``.

* :class:`ReplicateGLSW` down-weights replicate directions (generalized
  least-squares weighting).
* :class:`ReplicateEPO` projects out the top-k replicate directions (external
  parameter orthogonalization).

Supplying the replicate library
-------------------------------
The library is usually a separate set of samples scanned several times, with no
reference values. Pass it in the constructor (``replicates``, ``replicate_groups``);
``fit(X, y)`` then only needs the calibration data, which keeps the filter usable as
an ordinary Pipeline step and inside ``GridSearchCV``.

If your calibration set itself contains replicates, leave ``replicates=None`` and
pass ``groups`` to ``fit`` instead.

Representation (rule 2)
-----------------------
The library must be in the same representation as the filter's input. If the
filter follows a preprocessing step, pass that step as ``preprocessor``: it is
fitted on the calibration spectra given to ``fit`` (so MSC/EMSC references come
from calibration spectra only), applied to the replicate library before the
filter is estimated, and applied to every spectrum in ``transform``.

References
----------
Martens, H., Høy, M., Wise, B. M., Bro, R., Brockhoff, P. B. (2003). Pre-whitening
of data by covariance-weighted pre-processing. J. Chemometrics 17, 153-165. (GLSW)

Roger, J.-M., Chauchard, F., Bellon-Maurel, V. (2003). EPO-PLS external parameter
orthogonalisation of PLS application to temperature-independent measurement of
sugar content of intact fruits. Chemom. Intell. Lab. Syst. 66, 191-204. (EPO)
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from .core import ReplicateNoiseModel

__all__ = ["ReplicateGLSW", "ReplicateEPO"]


class _ReplicateFilterBase(TransformerMixin, BaseEstimator):
    def _fit_eig(self, X, groups):
        S = ReplicateNoiseModel(self.exclude_outliers).fit(X, groups=groups).covariance_
        lam, V = np.linalg.eigh(S)
        self.eigenvalues_, self.eigenvectors_ = lam[::-1].clip(0), V[:, ::-1]

    def fit(self, X, y=None, groups=None):
        """Estimate the filter.

        Parameters
        ----------
        X : array (n_samples, n_channels)
            Calibration spectra (used to fit ``preprocessor`` and to check the
            channel count). When no library was given in the constructor, ``X``
            must itself be replicate scans and ``groups`` is required.
        y : ignored
        groups : array (n_samples,), optional
            Sample IDs for ``X``; only used when ``replicates`` is None.
        """
        X = np.asarray(X, float)
        n_raw = X.shape[1]
        if self.replicates is not None:
            if self.replicate_groups is None:
                raise ValueError("replicate_groups is required when replicates is given.")
            if groups is not None:
                raise ValueError("Pass the library either in the constructor or as fit(groups=...), not both.")
            R, g = np.asarray(self.replicates, float), self.replicate_groups
        else:
            if groups is None:
                raise ValueError("No replicate library: pass replicates/replicate_groups to the "
                                 "constructor, or groups to fit() when X contains replicates.")
            R, g = X, groups

        if self.preprocessor is not None:
            self.preprocessor_ = clone(self.preprocessor).fit(X)
            X = self.preprocessor_.transform(X)
            R = self.preprocessor_.transform(R)
        else:
            self.preprocessor_ = None

        if R.shape[1] != X.shape[1]:
            raise ValueError(f"Replicate library has {R.shape[1]} channels but X has {X.shape[1]} "
                             "(after preprocessing).")
        self._fit_eig(R, g)
        self._build()
        self.n_features_in_ = n_raw
        return self

    def transform(self, X):
        check_is_fitted(self, "filter_")
        X = np.asarray(X, float)
        if self.preprocessor_ is not None:
            X = self.preprocessor_.transform(X)
        if X.shape[1] != self.filter_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} channels; filter expects {self.filter_.shape[0]}")
        return X @ self.filter_


class ReplicateGLSW(_ReplicateFilterBase):
    """Generalized least-squares weighting estimated from replicate differences.

    ``G = V diag(1 / sqrt(lambda / (alpha * lambda_max) + 1)) V'`` and
    ``transform(X) = X G``, with (lambda, V) the eigenpairs of the replicate second
    moment. Smaller ``alpha`` down-weights replicate directions more strongly; as
    ``alpha -> inf``, ``G -> I``.

    Parameters
    ----------
    alpha : float, default=1e-3
        Select by cross-validation over roughly 1e-6 to 1e-1. On wheat protein,
        first-derivative SG followed by GLSW with alpha ~1e-4 was best (the edge of
        the tested grid, so search below it too).
    replicates, replicate_groups : arrays, optional
        Replicate library (raw representation; see module docs).
    preprocessor : transformer, optional
        Step applied before the filter, fitted on calibration spectra.
    exclude_outliers : bool, default=True

    Attributes
    ----------
    filter_ : array (n_channels, n_channels)
    eigenvalues_, eigenvectors_ : replicate eigenpairs, descending
    preprocessor_ : fitted clone of ``preprocessor`` or None

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> model = make_pipeline(
    ...     ReplicateGLSW(alpha=1e-4, replicates=X_rep, replicate_groups=rep_ids,
    ...                   preprocessor=SavitzkyGolay(deriv=1)),
    ...     PLSRegression(n_components=8),
    ... )
    >>> model.fit(X_cal, y_cal).predict(X_test)
    """

    def __init__(self, alpha=1e-3, replicates=None, replicate_groups=None,
                 preprocessor=None, exclude_outliers=True):
        self.alpha = alpha
        self.replicates = replicates
        self.replicate_groups = replicate_groups
        self.preprocessor = preprocessor
        self.exclude_outliers = exclude_outliers

    def _build(self):
        lam, V = self.eigenvalues_, self.eigenvectors_
        w = 1.0 / np.sqrt(lam / (self.alpha * lam[0]) + 1.0)
        self.filter_ = (V * w) @ V.T


class ReplicateEPO(_ReplicateFilterBase):
    """External parameter orthogonalization: remove the top-k replicate directions.

    ``transform(X) = X (I - V_k V_k')``, with V_k the leading k eigenvectors of the
    replicate second moment.

    Parameters
    ----------
    n_components : int, default=5
        Number of replicate directions removed. Select by cross-validation.
    replicates, replicate_groups, preprocessor, exclude_outliers :
        As in :class:`ReplicateGLSW`.
    """

    def __init__(self, n_components=5, replicates=None, replicate_groups=None,
                 preprocessor=None, exclude_outliers=True):
        self.n_components = n_components
        self.replicates = replicates
        self.replicate_groups = replicate_groups
        self.preprocessor = preprocessor
        self.exclude_outliers = exclude_outliers

    def _build(self):
        Vk = self.eigenvectors_[:, :self.n_components]
        self.filter_ = np.eye(len(Vk)) - Vk @ Vk.T
