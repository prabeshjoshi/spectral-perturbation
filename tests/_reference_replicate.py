"""
replicate_perturbation.py
=========================
Data-driven spectral perturbation learned from REPLICATE scans of the same samples
(repacks / re-presentations / rescans on one instrument). Replicate differences contain
no chemistry change, so they are a direct, label-free sample of measurement nuisance
(packing, scatter, path length, drift, noise).

One replicate library -> four uses:
  1. ReplicateNoiseModel        : store/sampled real replicate differences ("transplant")
  2. ReplicateAugmenter         : data augmentation (x + real replicate difference)
  3. ReplicateGLSW / ReplicateEPO: preprocessing that removes ONLY replicate directions
  4. prediction_repeatability   : per-sample repeatability (sd) of any model's prediction,
     scatter_share               from a single scan (analytic delta method or Monte Carlo);
                                   diagnostic: share of prediction error caused by scatter

Conventions: X is (n_scans, n_wavelengths); `groups` gives the physical-sample ID of
each scan (>= 2 scans per sample needed). Fit the library on spectra in the SAME
representation the downstream step uses (e.g. fit GLSW on SG-derivative replicates if
GLSW follows an SG step). Dependencies: numpy, scikit-learn.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["replicate_differences", "ReplicateNoiseModel", "ReplicateAugmenter",
           "ReplicateGLSW", "ReplicateEPO", "prediction_repeatability", "scatter_share"]


# --------------------------------------------------------------------------- core
def replicate_differences(X, groups, exclude_outliers=True, outlier_factor=5.0):
    """All ordered within-sample pairwise differences x_a - x_b (a != b).

    Parameters
    ----------
    X : (n_scans, p) array.  groups : (n_scans,) sample IDs.
    exclude_outliers : drop scans whose distance to their sample's median spectrum
        exceeds `outlier_factor` x the median such distance (gross bad scans).
    Returns
    -------
    D : (n_pairs, p) array of replicate differences.
    """
    X = np.asarray(X, float); groups = np.asarray(groups)
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

    sample() draws real differences ('transplant'); covariance_ is their second
    moment S = D'D / n (used by GLSW/EPO and analytic repeatability).
    """
    def __init__(self, exclude_outliers=True, outlier_factor=5.0):
        self.exclude_outliers = exclude_outliers; self.outlier_factor = outlier_factor

    def fit(self, X, groups):
        self.differences_ = replicate_differences(X, groups, self.exclude_outliers, self.outlier_factor)
        self.covariance_ = self.differences_.T @ self.differences_ / len(self.differences_)
        self.n_features_in_ = self.differences_.shape[1]
        return self

    def sample(self, n, random_state=None):
        check_is_fitted(self, "differences_")
        rng = np.random.default_rng(random_state)
        return self.differences_[rng.integers(0, len(self.differences_), n)]


# --------------------------------------------------------------------------- 1. augmentation
class ReplicateAugmenter(BaseEstimator, TransformerMixin):
    """Augment spectra with real replicate differences: x_new = x + d, d ~ library.

    transform(X): perturbs each row once (sklearn-style, keeps n_rows; use in training only).
    augment(X, y, n_copies): returns originals + n_copies perturbed copies, labels and
        group IDs (use group-aware CV afterwards so copies of a sample stay together).
    Apply BEFORE preprocessing (perturb raw spectra, then preprocess the augmented set).
    """
    def __init__(self, scale=1.0, exclude_outliers=True, random_state=None):
        self.scale = scale; self.exclude_outliers = exclude_outliers; self.random_state = random_state

    def fit(self, X, groups):
        self.noise_ = ReplicateNoiseModel(self.exclude_outliers).fit(X, groups)
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = self.noise_.n_features_in_
        return self

    def transform(self, X):
        check_is_fitted(self, "noise_"); X = np.asarray(X, float)
        return X + self.scale * self.noise_.sample(len(X), self._rng)

    def augment(self, X, y, n_copies=15, groups=None):
        X = np.asarray(X, float); y = np.asarray(y)
        g = np.arange(len(X)) if groups is None else np.asarray(groups)
        Xa = [X] + [self.transform(X) for _ in range(n_copies)]
        return np.vstack(Xa), np.tile(y, n_copies + 1), np.tile(g, n_copies + 1)


# --------------------------------------------------------------------------- 2. preprocessing
class _ReplicateFilterBase(BaseEstimator, TransformerMixin):
    def _fit_eig(self, X, groups):
        S = ReplicateNoiseModel(self.exclude_outliers).fit(X, groups).covariance_
        lam, V = np.linalg.eigh(S)
        self.eigenvalues_, self.eigenvectors_ = lam[::-1].clip(0), V[:, ::-1]
        self.n_features_in_ = S.shape[0]

    def fit(self, X, y=None, groups=None):
        """Fit on REPLICATE spectra with `groups`. (y ignored; groups required.)"""
        if groups is None:
            raise ValueError("groups (sample IDs of replicate scans) is required.")
        self._fit_eig(X, groups); self._build(); return self


class ReplicateGLSW(_ReplicateFilterBase):
    """Generalized least-squares weighting from replicate differences.

    G = V diag( 1 / sqrt(lambda / (alpha * lambda_max) + 1) ) V'.  transform(X) = X G.
    Smaller alpha = stronger down-weighting of replicate directions.
    Validated best on wheat: SG 1st derivative, then GLSW with alpha ~1e-4 (edge of
    tested grid; try 1e-6..1e-1 and select by CV).
    """
    def __init__(self, alpha=1e-3, exclude_outliers=True):
        self.alpha = alpha; self.exclude_outliers = exclude_outliers
    def _build(self):
        lam, V = self.eigenvalues_, self.eigenvectors_
        w = 1.0 / np.sqrt(lam / (self.alpha * lam[0]) + 1.0)
        self.filter_ = (V * w) @ V.T
    def transform(self, X):
        check_is_fitted(self, "filter_"); return np.asarray(X, float) @ self.filter_


class ReplicateEPO(_ReplicateFilterBase):
    """External parameter orthogonalization: remove the top-k replicate directions.
    transform(X) = X (I - V_k V_k')."""
    def __init__(self, n_components=5, exclude_outliers=True):
        self.n_components = n_components; self.exclude_outliers = exclude_outliers
    def _build(self):
        Vk = self.eigenvectors_[:, :self.n_components]; self.filter_ = np.eye(len(Vk)) - Vk @ Vk.T
    def transform(self, X):
        check_is_fitted(self, "filter_"); return np.asarray(X, float) @ self.filter_


# --------------------------------------------------------------------------- 3. repeatability
def prediction_repeatability(predict, X, noise_model, method="analytic", n_draws=100,
                             eps=None, random_state=None):
    """Per-sample sd of predictions under realistic re-measurement, from ONE scan each.

    predict : callable mapping raw spectra (m, p) -> predictions (m,)  (full pipeline:
              preprocessing + model, e.g. fitted sklearn Pipeline .predict)
    noise_model : fitted ReplicateNoiseModel (on raw spectra)
    method : 'analytic' -> sd_i = sqrt(g_i' S g_i), g_i = numerical gradient of predict at x_i
                           (delta method; p+1 predictions per sample; exact for linear pipelines)
             'mc'       -> sd over n_draws of predict(x_i + d) - predict(x_i)  (any model)
    Returns (n,) array. Note: S is the 2nd moment of replicate DIFFERENCES, so this is the
    spread of 'another measurement relative to this one' (variance of a difference of two
    measurements). Divide by sqrt(2) for single-measurement repeatability sd.
    Validated: matched real repack spread within ~10% and gave 92-99% coverage of real
    repack predictions on 2 wheat instruments + 3 soil properties; MC and analytic agree.
    """
    X = np.atleast_2d(np.asarray(X, float)); rng = np.random.default_rng(random_state)
    base = np.asarray(predict(X)).ravel(); out = np.empty(len(X))
    if method == "mc":
        for i, x in enumerate(X):
            q = np.asarray(predict(x[None] + noise_model.sample(n_draws, rng))).ravel() - base[i]
            out[i] = np.sqrt(np.mean(q ** 2))
    elif method == "analytic":
        S = noise_model.covariance_; p = X.shape[1]
        for i, x in enumerate(X):
            h = eps if eps is not None else 1e-4 * (np.abs(x).mean() + 1e-12)
            grad = (np.asarray(predict(x[None] + h * np.eye(p))).ravel() - base[i]) / h
            out[i] = np.sqrt(max(grad @ S @ grad, 0.0))
    else:
        raise ValueError("method must be 'analytic' or 'mc'")
    return out


def scatter_share(predict, X, groups, y):
    """Share of prediction-error variance caused by replicate scatter.

    X, groups : replicate scans of TEST samples (not used in training), y : per-scan reference.
    share = mean within-sample prediction variance / mean squared prediction error.
    Rule of thumb from validation: share ~0.45 -> replicate GLSW/augmentation cut error
    ~20-80% (wheat); share <= 0.03 -> no gain (soil). Decide BEFORE investing in these steps.
    """
    pred = np.asarray(predict(np.asarray(X, float))).ravel(); groups = np.asarray(groups)
    within = [np.var(pred[groups == g], ddof=1) for g in np.unique(groups) if np.sum(groups == g) > 1]
    return float(np.mean(within) / np.mean((pred - np.asarray(y)) ** 2))
