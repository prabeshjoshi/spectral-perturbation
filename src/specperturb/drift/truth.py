"""Ground truth for a PLS model: how much of the drift the model can see, how
much it ignores, and the true prediction bias it causes.

For a model z -> PLS (z = preprocessed spectrum), write the drift in the
model's input space as

    dz = dz_in + dz_res,  dz_in = inverse_transform(t(z + dz)) - inverse_transform(t(z))

dz_in lies in the span of the PLS loadings (what T^2 sees); dz_res is the
rest (what Q / DModX sees). Because P'W* = I for PLS, dz_res changes the scores
by exactly zero, so it CANNOT change the prediction: all prediction bias comes
from dz_in. A Q alarm with no in-model drift is therefore harmless to this model
(though it may signal that the model is being used outside its domain).

Exact for any preprocessing (the split is done after it); bias is computed by
running the full model, so it is exact too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline

__all__ = ["ModelTruth", "model_truth", "harm_onset"]


@dataclass
class ModelTruth:
    bias: np.ndarray             # predict(X_drift) - predict(X_clean), incl. noise
    bias_systematic: np.ndarray  # same, using the noise-free drifted stream
    delta_scores: np.ndarray     # change in PLS scores (n, A)
    delta_in: np.ndarray         # in-model part of the drift, model input space (n, p)
    delta_res: np.ndarray        # residual part (n, p)

    @property
    def norm_in(self):
        return np.linalg.norm(self.delta_in, axis=1)

    @property
    def norm_res(self):
        return np.linalg.norm(self.delta_res, axis=1)

    @property
    def residual_share(self):
        """Share of drift energy outside the model, ||dz_res||^2 / ||dz||^2 (nan where no drift)."""
        tot = self.norm_in ** 2 + self.norm_res ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, self.norm_res ** 2 / tot, np.nan)


def _split(model):
    if isinstance(model, Pipeline):
        pre, pls = model[:-1], model[-1]
    else:
        pre, pls = None, model
    if not (hasattr(pls, "transform") and hasattr(pls, "inverse_transform") and hasattr(pls, "x_rotations_")):
        raise TypeError("model must be a fitted PLS model or a Pipeline ending in one")
    return pre, pls


def model_truth(model, X_clean, X_drift, X_systematic=None):
    """Ground truth for a fitted PLS model (or Pipeline ending in PLS).

    Parameters
    ----------
    model : fitted PLSRegression / Pipeline(..., PLSRegression)
    X_clean, X_drift : raw spectra without / with drift (row-paired)
    X_systematic : drifted spectra without stochastic mechanisms; defaults to X_drift
    """
    pre, pls = _split(model)
    Xc = np.asarray(X_clean, float)
    Xd = np.asarray(X_drift, float)
    Xs = Xd if X_systematic is None else np.asarray(X_systematic, float)

    zc = Xc if pre is None else pre.transform(Xc)
    zd = Xd if pre is None else pre.transform(Xd)
    tc, td = pls.transform(zc), pls.transform(zd)
    dz = zd - zc
    dz_in = pls.inverse_transform(td) - pls.inverse_transform(tc)

    y0 = np.asarray(model.predict(Xc)).reshape(len(Xc), -1)[:, 0]
    bias = np.asarray(model.predict(Xd)).reshape(len(Xd), -1)[:, 0] - y0
    bias_sys = bias if Xs is Xd else np.asarray(model.predict(Xs)).reshape(len(Xs), -1)[:, 0] - y0
    return ModelTruth(bias=bias, bias_systematic=bias_sys, delta_scores=td - tc,
                      delta_in=dz_in, delta_res=dz - dz_in)


def harm_onset(bias, threshold, persistence=1):
    """Index of the first spectrum from which |bias| > threshold for
    ``persistence`` consecutive spectra, or None.

    Choose ``threshold`` from the method's fitness for purpose (e.g. the
    acceptable bias in the method specification, or k x RMSEP)."""
    exceed = np.abs(np.asarray(bias)) > threshold
    if persistence <= 1:
        idx = np.flatnonzero(exceed)
        return int(idx[0]) if len(idx) else None
    run = np.convolve(exceed.astype(int), np.ones(persistence, int), mode="valid")
    idx = np.flatnonzero(run == persistence)
    return int(idx[0]) if len(idx) else None
