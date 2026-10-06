"""Model-level diagnostics from a replicate library.

* :func:`prediction_repeatability` - per-sample sd of a model's prediction under
  realistic re-measurement, from one scan per sample.
* :func:`scatter_share` - share of prediction-error variance caused by replicate
  scatter; tells you in advance whether replicate augmentation or filtering is
  worth trying.

References
----------
Faber, N. M., Kowalski, B. R. (1997). Propagation of measurement errors for the
validation of predictions obtained by principal component regression and partial
least squares. J. Chemometrics 11, 181-238.

Wentzell, P. D., Andrews, D. T., Hamilton, D. C., Faber, K., Kowalski, B. R. (1997).
Maximum likelihood principal component analysis. J. Chemometrics 11, 339-366.
"""
from __future__ import annotations

import numpy as np

__all__ = ["prediction_repeatability", "scatter_share"]


def _as_noise_model(obj):
    """Accept a fitted ReplicateNoiseModel, ReplicateAugmenter or ReplicateNoise."""
    for attr in ("noise_model_", "noise_"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            return inner
    if hasattr(obj, "covariance_") and hasattr(obj, "sample"):
        return obj
    raise TypeError("noise_model must be a fitted ReplicateNoiseModel (or ReplicateNoise / ReplicateAugmenter)")


def prediction_repeatability(predict, X, noise_model, method="analytic", n_draws=100,
                             eps=None, random_state=None):
    """Per-sample sd of predictions under realistic re-measurement, from one scan each.

    Parameters
    ----------
    predict : callable
        Maps raw spectra (m, p) to predictions (m,), i.e. the FULL pipeline
        (preprocessing + model), for example a fitted ``Pipeline.predict``.
    X : array (n, p)
        One raw scan per sample.
    noise_model : fitted ReplicateNoiseModel (or ReplicateNoise / ReplicateAugmenter)
        Library built on raw spectra.
    method : {'analytic', 'mc'}, default='analytic'
        'analytic': ``sd_i = sqrt(g_i' S g_i)`` with ``g_i`` the forward-difference
        gradient of ``predict`` at ``x_i`` (delta method; p + 1 predictions per
        sample; exact for linear pipelines).
        'mc': root-mean-square of ``predict(x_i + d) - predict(x_i)`` over
        ``n_draws`` real differences (any model).
    n_draws : int, default=100
        Monte Carlo draws per sample.
    eps : float, optional
        Finite-difference step for 'analytic'; defaults to 1e-4 x mean |x_i|.
    random_state : int, Generator or None

    Returns
    -------
    sd : array (n,)
        Spread of "another measurement relative to this one", i.e. the sd of a
        difference of two measurements. Divide by sqrt(2) for single-measurement
        repeatability.

    Notes
    -----
    Validated on two wheat instruments and three soil properties: simulated spread
    matched real repack spread within ~10 %, with 92-99 % coverage of real repack
    predictions, and analytic and MC estimates agreed (ratio 0.97-1.01).
    Per-sample ranking is only partly captured (rank correlation with real spread
    roughly 0-0.65 depending on preprocessing).
    """
    noise_model = _as_noise_model(noise_model)
    X = np.atleast_2d(np.asarray(X, float))
    rng = np.random.default_rng(random_state)
    base = np.asarray(predict(X)).ravel()
    out = np.empty(len(X))
    if method == "mc":
        for i, x in enumerate(X):
            q = np.asarray(predict(x[None] + noise_model.sample(n_draws, rng))).ravel() - base[i]
            out[i] = np.sqrt(np.mean(q ** 2))
    elif method == "analytic":
        S = noise_model.covariance_
        p = X.shape[1]
        for i, x in enumerate(X):
            h = eps if eps is not None else 1e-4 * (np.abs(x).mean() + 1e-12)
            grad = (np.asarray(predict(x[None] + h * np.eye(p))).ravel() - base[i]) / h
            out[i] = np.sqrt(max(grad @ S @ grad, 0.0))
    else:
        raise ValueError("method must be 'analytic' or 'mc'")
    return out


def scatter_share(predict, X, groups, y):
    """Share of prediction-error variance caused by replicate scatter.

    ``share = mean within-sample prediction variance / mean squared prediction error``

    Parameters
    ----------
    predict : callable
        Full fitted pipeline's predict.
    X : array (n_scans, p)
        Replicate scans of TEST samples (not used in training or in the library).
    groups : array (n_scans,)
        Sample ID per scan; samples with a single scan are ignored in the numerator.
    y : array (n_scans,)
        Reference value per scan.

    Returns
    -------
    share : float

    Notes
    -----
    Rule of thumb from validation: share ~0.45 (wheat) went with 20-80 % error
    reduction from replicate GLSW or augmentation; share <= 0.03 (soil) with none.
    Compute it before investing in those steps.
    """
    pred = np.asarray(predict(np.asarray(X, float))).ravel()
    groups = np.asarray(groups)
    within = [np.var(pred[groups == g], ddof=1) for g in np.unique(groups) if np.sum(groups == g) > 1]
    return float(np.mean(within) / np.mean((pred - np.asarray(y)) ** 2))
