"""Empirical drift SURROGATES. These are not physical models: they move spectra
along directions learned from real data, scaled by theta. Every result carries
``surrogate=True`` and a ``source`` label, so reports can keep them apart from
the physical mechanisms.

* :class:`PairedInstrumentDrift` - the difference between two instruments (or
  one instrument at two times) measured on the same samples. Uses chemotools'
  direct standardization (DS) or piecewise DS (PDS) to map any spectrum from the
  reference state toward the drifted state; theta = 1 means "fully moved".
* :class:`ReplicateDrift` - a fixed direction from the replicate library: the
  k-th principal direction of replicate differences (scaled to one sd), or one
  real replicate difference.
"""
from __future__ import annotations

import numpy as np

from ..replicate import ReplicateNoiseModel
from .mechanisms import Mechanism

__all__ = ["PairedInstrumentDrift", "ReplicateDrift"]


class PairedInstrumentDrift(Mechanism):
    """Surrogate drift from paired spectra of the same samples on a reference
    instrument (A, where the model was built) and a drifted / second
    instrument (B).

        X -> X + theta * (T(X) - X),   T(X_A) ~= X_B

    Parameters
    ----------
    X_reference, X_drifted : arrays (n_pairs, n_channels), row-paired.
    method : 'ds' (chemotools DirectStandardization), 'pds'
        (chemotools PiecewiseDirectStandardization) or 'mean' (add the mean
        difference X_B - X_A, sample-independent).
    pds_kwargs : passed to PiecewiseDirectStandardization.

    Notes
    -----
    DS with fewer pairs than channels is rank-deficient: spectra far from the
    transfer set are mapped poorly. PDS or 'mean' are safer with few pairs.
    """

    parameter = "fraction of the reference->drifted instrument difference"
    stage = "optics"
    surrogate = True

    def __init__(self, X_reference, X_drifted, method="ds", pds_kwargs=None, name=None):
        super().__init__(name)
        A = np.asarray(X_reference, float)
        B = np.asarray(X_drifted, float)
        if A.shape != B.shape:
            raise ValueError("X_reference and X_drifted must be row-paired with equal shape")
        self.method = method
        self.source = f"paired-instrument ({method})"
        if method == "mean":
            self.mean_diff_ = (B - A).mean(0)
            self.mapper_ = None
        elif method in ("ds", "pds"):
            from chemotools.adaptation import DirectStandardization, PiecewiseDirectStandardization
            cls = DirectStandardization if method == "ds" else PiecewiseDirectStandardization
            kw = {} if method == "ds" else dict(pds_kwargs or {})
            # chemotools maps the fitted X ("target") onto X_source
            self.mapper_ = cls(**kw).fit(A, X_source=B)
        else:
            raise ValueError("method must be 'ds', 'pds' or 'mean'")

    def apply(self, X, theta, x=None, rng=None):
        X = np.array(X, dtype=float)
        theta = np.broadcast_to(np.asarray(theta, float), (X.shape[0],))
        active = theta != 0
        if not active.any():
            return X
        if self.mapper_ is None:
            step = np.broadcast_to(self.mean_diff_, X[active].shape)
        else:
            step = self.mapper_.transform(X[active]) - X[active]
        X[active] = X[active] + theta[active, None] * step
        return X


class ReplicateDrift(Mechanism):
    """Surrogate drift along a fixed replicate direction.

        X -> X + theta * d

    mode='direction': d = sqrt(lambda_k) v_k, the k-th eigenpair of the
        replicate second moment, so theta = 1 is one sd of replicate differences
        along that direction (sign of v_k is arbitrary).
    mode='sample': d = one real replicate difference, drawn with ``seed``.

    Parameters
    ----------
    replicates, groups : replicate scans and sample IDs (or pass noise_model).
    noise_model : a fitted ReplicateNoiseModel, instead of replicates/groups.
    """

    parameter = "multiples of the replicate direction"
    surrogate = True

    def __init__(self, replicates=None, groups=None, noise_model=None, mode="direction",
                 component=0, seed=None, name=None):
        super().__init__(name)
        if noise_model is None:
            if replicates is None or groups is None:
                raise ValueError("Pass replicates and groups, or a fitted noise_model")
            noise_model = ReplicateNoiseModel().fit(replicates, groups=groups)
        self.mode = mode
        if mode == "direction":
            lam, V = np.linalg.eigh(noise_model.covariance_)
            lam, V = lam[::-1].clip(0), V[:, ::-1]
            self.direction_ = np.sqrt(lam[component]) * V[:, component]
        elif mode == "sample":
            self.direction_ = noise_model.sample(1, seed)[0]
        else:
            raise ValueError("mode must be 'direction' or 'sample'")
        self.source = f"replicate-learned ({mode})"

    def apply(self, X, theta, x=None, rng=None):
        X = np.array(X, dtype=float)
        theta = np.broadcast_to(np.asarray(theta, float), (X.shape[0],))
        return X + theta[:, None] * self.direction_[None, :]
