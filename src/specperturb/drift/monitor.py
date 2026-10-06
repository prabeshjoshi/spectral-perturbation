"""Monitoring statistics and change detectors, mostly from existing libraries,
behind one interface so they can be calibrated to the same false-alarm rate.

Statistics (chemotools): Hotelling T^2, Q residuals, DModX of a fitted PCA/PLS
model or Pipeline, via ``predict_residuals``.

Detectors
---------
=====================  ==========================  =======================
Detector               Implementation              Calibrated parameter
=====================  ==========================  =======================
Shewhart               here (threshold on stat)    threshold
EWMA                   here (scipy lfilter)        threshold on EWMA
CUSUM                  menelaus.CUSUM              threshold (h)
PageHinkley            river.drift.PageHinkley     threshold
ADWIN                  river.drift.ADWIN           delta
KSWIN (slow)           river.drift.KSWIN           alpha
Martingale             MAPIE OnlineMartingaleTest  threshold on martingale
=====================  ==========================  =======================

Detectors work on a univariate statistic stream (T^2, Q, ...). CUSUM,
Page-Hinkley, ADWIN and KSWIN receive it standardized with the in-control
reference mean and sd. Native thresholds of different libraries imply different
false-alarm rates, so compare detectors only after :func:`calibrate`.

Two notes for interpreting results:
* Q and DModX are monotone transforms of each other per spectrum, so after
  calibration they give identical alarm times.
* Martingale(pvalues='stream') tests exchangeability of the stream with its own
  past (MAPIE's native use); pvalues='reference' compares each value with the
  in-control reference set, like the control charts.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
from scipy.signal import lfilter

__all__ = ["Statistic", "hotelling_t2", "q_residuals", "dmodx",
           "Detector", "Shewhart", "EWMA", "CUSUM", "PageHinkley", "ADWIN", "KSWIN",
           "Martingale", "calibrate", "bootstrap_sequences"]


def _require(module, extra="monitor"):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"{module} is needed here: pip install 'specperturb[{extra}]'") from e


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
class Statistic:
    """Per-spectrum monitoring statistic from a chemotools outlier detector.

    Parameters
    ----------
    kind : 'HotellingT2' | 'QResiduals' | 'DModX' | 'Leverage'
    model : fitted PCA / PLS model or Pipeline ending in one
    log : apply log to the statistic (makes T^2 / Q less skewed for EWMA/CUSUM)
    kwargs : passed to the chemotools class
    """

    def __init__(self, kind, model, log=False, name=None, **kwargs):
        self.kind, self.model, self.log, self.kwargs = kind, model, log, kwargs
        self.name = name or {"HotellingT2": "T2", "QResiduals": "Q"}.get(kind, kind)

    def fit(self, X_cal):
        outliers = _require("chemotools.outliers")
        self.detector_ = getattr(outliers, self.kind)(self.model, **self.kwargs).fit(X_cal)
        return self

    def __call__(self, X):
        s = np.asarray(self.detector_.predict_residuals(np.asarray(X, float)), float).ravel()
        return np.log(np.clip(s, 1e-300, None)) if self.log else s


def hotelling_t2(model, **kw):
    return Statistic("HotellingT2", model, **kw)


def q_residuals(model, **kw):
    return Statistic("QResiduals", model, **kw)


def dmodx(model, **kw):
    return Statistic("DModX", model, **kw)


# --------------------------------------------------------------------------- #
#  Detectors
# --------------------------------------------------------------------------- #
class Detector:
    """Base: ``setup(s_ref)`` learns the in-control reference; ``first_alarm(s,
    knob)`` returns the index of the first alarm or None.

    ``knob`` is the parameter calibrated to a target in-control ARL;
    ``knob_increases_arl`` says which way it acts; ``log_knob`` bisects in log space.
    """

    knob_increases_arl = True
    log_knob = False
    knob_bounds = None
    path_based = False

    def setup(self, s_ref):
        s_ref = np.asarray(s_ref, float)
        self.mu_, self.sd_ = float(np.mean(s_ref)), float(np.std(s_ref, ddof=1))
        self.ref_sorted_ = np.sort(s_ref)
        return self

    def _z(self, s):
        return (np.asarray(s, float) - self.mu_) / (self.sd_ if self.sd_ > 0 else 1.0)

    def first_alarm(self, s, knob):
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def __repr__(self):
        kv = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.endswith("_"))
        return f"{type(self).__name__}({kv})"


class _PathDetector(Detector):
    """Alarm when a monitoring path exceeds the knob (a threshold)."""

    path_based = True

    def path(self, s):
        raise NotImplementedError

    def first_alarm(self, s, knob):
        idx = np.flatnonzero(self.path(s) > knob)
        return int(idx[0]) if len(idx) else None


class Shewhart(_PathDetector):
    """Alarm when the statistic itself exceeds the threshold."""

    def path(self, s):
        return np.asarray(s, float)


class EWMA(_PathDetector):
    """Upper EWMA chart: z_t = lam s_t + (1 - lam) z_{t-1}, z_0 = in-control mean."""

    def __init__(self, lam=0.2):
        self.lam = lam

    def path(self, s):
        s = np.asarray(s, float)
        z, _ = lfilter([self.lam], [1.0, -(1.0 - self.lam)], s, zi=[(1.0 - self.lam) * self.mu_])
        return z


class CUSUM(Detector):
    """One-sided upper tabular CUSUM (menelaus) on the standardized statistic.
    k : reference value (allowance) in sd units; knob = decision threshold h."""

    knob_bounds = (0.05, 200.0)

    def __init__(self, k=0.5):
        self.k = k

    def first_alarm(self, s, knob):
        from menelaus.change_detection import CUSUM as _C
        det = _C(target=0.0, sd_hat=1.0, burn_in=1, delta=self.k, threshold=knob, direction="positive")
        for i, v in enumerate(self._z(s)):
            det.update(np.array([[v]]))
            if det.drift_state == "drift":
                return i
        return None


class PageHinkley(Detector):
    """river's Page-Hinkley test (upward changes) on the standardized statistic."""

    knob_bounds = (0.01, 500.0)

    def __init__(self, delta=0.1, alpha=1.0, min_instances=1):
        self.delta, self.alpha, self.min_instances = delta, alpha, min_instances

    def first_alarm(self, s, knob):
        from river.drift import PageHinkley as _PH
        det = _PH(min_instances=self.min_instances, delta=self.delta, threshold=knob,
                  alpha=self.alpha, mode="up")
        for i, v in enumerate(self._z(s)):
            det.update(float(v))
            if det.drift_detected:
                return i
        return None


class ADWIN(Detector):
    """river's ADWIN (adaptive windowing; two-sided change in mean). Knob = delta."""

    knob_increases_arl = False
    log_knob = True
    knob_bounds = (1e-30, 0.5)

    def __init__(self, clock=1, grace_period=10):
        self.clock, self.grace_period = clock, grace_period

    def first_alarm(self, s, knob):
        from river.drift import ADWIN as _A
        det = _A(delta=knob, clock=self.clock, grace_period=self.grace_period)
        for i, v in enumerate(self._z(s)):
            det.update(float(v))
            if det.drift_detected:
                return i
        return None


class KSWIN(Detector):
    """river's KSWIN (Kolmogorov-Smirnov windowing). Knob = alpha. ~300 us per
    update, so calibration is slow; reduce n_sim or arl0 when using it."""

    knob_increases_arl = False
    log_knob = True
    knob_bounds = (1e-12, 0.5)

    def __init__(self, window_size=100, stat_size=30, seed=0):
        self.window_size, self.stat_size, self.seed = window_size, stat_size, seed

    def first_alarm(self, s, knob):
        from river.drift import KSWIN as _K
        det = _K(alpha=knob, window_size=self.window_size, stat_size=self.stat_size, seed=self.seed)
        for i, v in enumerate(self._z(s)):
            det.update(float(v))
            if det.drift_detected:
                return i
        return None


class Martingale(_PathDetector):
    """Conformal test martingale (MAPIE OnlineMartingaleTest) on p-values of the
    statistic; alarm when the martingale exceeds the knob.

    method  : 'jumper' (simple jumper, MAPIE default) or 'plugin' (KDE plug-in;
              slower, O(t) per step)
    pvalues : 'reference' -> p = (1 + #{ref >= s}) / (n_ref + 1)
              'stream'    -> MAPIE's own p-value against the stream's past
    Ville's inequality gives P(ever > 1/alpha) <= alpha under the null, so the
    native threshold is 1/alpha; use calibrate() to compare at matched ARL.
    """

    log_knob = True
    knob_bounds = (1.0, 1e12)

    def __init__(self, method="jumper", pvalues="reference", jump_size=0.01, random_state=0):
        if method not in ("jumper", "plugin") or pvalues not in ("reference", "stream"):
            raise ValueError("method in {'jumper','plugin'}, pvalues in {'reference','stream'}")
        self.method, self.pvalues = method, pvalues
        self.jump_size, self.random_state = jump_size, random_state

    def _pvals_reference(self, s):
        n = len(self.ref_sorted_)
        n_ge = n - np.searchsorted(self.ref_sorted_, np.asarray(s, float), side="left")
        return (1.0 + n_ge) / (n + 1.0)

    def path(self, s):
        from mapie.exchangeability_testing import OnlineMartingaleTest
        s = np.asarray(s, float)
        test = OnlineMartingaleTest(warn=False, jump_size=self.jump_size, random_state=self.random_state)
        update = test.update_simple_jumper_martingale if self.method == "jumper" else test.update_plugin_martingale
        p_ref = self._pvals_reference(s) if self.pvalues == "reference" else None
        out = np.empty(len(s))
        for i, v in enumerate(s):
            if p_ref is not None:
                p = float(p_ref[i])
            else:
                p = float(test.compute_p_value(current_conformity_score=float(v),
                                               conformity_score_history=np.asarray(test.conformity_score_history, float)))
            out[i] = update(p)
            test.conformity_score_history.append(float(v))   # same order as MAPIE's update()
            test.pvalue_history.append(p)
        return out


# --------------------------------------------------------------------------- #
#  Calibration to a target in-control ARL
# --------------------------------------------------------------------------- #
def bootstrap_sequences(s, n_sim, length, rng, block=1):
    """In-control sequences resampled from in-control statistic values ``s``.
    block > 1 uses a circular moving-block bootstrap to keep autocorrelation."""
    s = np.asarray(s, float)
    rng = np.random.default_rng(rng)
    if block <= 1:
        return s[rng.integers(0, len(s), (n_sim, length))]
    n_blocks = -(-length // block)
    starts = rng.integers(0, len(s), (n_sim, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(s)
    return s[idx.reshape(n_sim, -1)[:, :length]]


def _run_lengths_from_paths(runmax, h, horizon):
    hit = runmax > h
    return np.where(hit.any(1), hit.argmax(1) + 1, horizon + 1)


def calibrate(detector, s_incontrol, arl0=200, n_sim=200, horizon=None, block=1, rng=None,
              n_iter=30, return_details=False, criterion="arl", window=None, fap=0.05):
    """Set the detector's knob from in-control behaviour.

    criterion='arl' : mean in-control run length ~= ``arl0`` (classical charts).
        Run lengths are censored at ``horizon`` (default 5 x arl0).
    criterion='fap' : P(false alarm within ``window`` spectra) ~= ``fap``.
        Use this for conformal martingales (whose in-control run-length
        distribution is defective: many runs never alarm, so their ARL depends on
        the horizon) and whenever detectors of different kinds are compared.

    Run lengths come from bootstrap in-control sequences of ``s_incontrol``:
    statistic values of in-control spectra NOT used to fit the model or the
    detector's reference. Returns the knob (and the achieved metric if
    ``return_details``). Warns when the target is out of the detector's reach.
    """
    if criterion == "arl":
        horizon = horizon or int(5 * arl0)
        target = arl0
        metric = lambda rl: float(np.mean(np.minimum(rl, horizon)))  # noqa: E731
        too_frequent = lambda v: v < target  # noqa: E731
    elif criterion == "fap":
        if window is None:
            raise ValueError("criterion='fap' needs window")
        horizon = int(window)
        target = fap
        metric = lambda rl: float(np.mean(rl <= window))  # noqa: E731
        too_frequent = lambda v: v > target  # noqa: E731
    else:
        raise ValueError("criterion must be 'arl' or 'fap'")

    seqs = bootstrap_sequences(s_incontrol, n_sim, horizon, rng, block)
    if detector.path_based:
        paths = np.vstack([detector.path(q) for q in seqs])
        runmax = np.maximum.accumulate(paths, axis=1)
        lo, hi = float(np.min(paths)) - 1e-9, float(np.max(runmax)) + 1e-9
        log = detector.log_knob and lo > 0
        f = lambda h: metric(_run_lengths_from_paths(runmax, h, horizon))  # noqa: E731
        increasing = True
    else:
        lo, hi = detector.knob_bounds
        log = detector.log_knob
        increasing = detector.knob_increases_arl

        def f(h):
            rl = np.array([horizon + 1 if (a := detector.first_alarm(q, h)) is None else a + 1 for q in seqs])
            return metric(rl)

    a, b = (np.log(lo), np.log(hi)) if log else (lo, hi)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        h = np.exp(mid) if log else mid
        if too_frequent(f(h)) == increasing:
            a = mid
        else:
            b = mid
    knob = float(np.exp(0.5 * (a + b)) if log else 0.5 * (a + b))
    achieved = f(knob)
    tol = 0.25 * target if criterion == "arl" else max(0.02, 0.5 * target)
    if abs(achieved - target) > tol:
        warnings.warn(f"{detector!r}: achieved {criterion} {achieved:.3g} vs target {target:.3g} "
                      "(target outside the detector's range, or run lengths too discrete)")
    return (knob, achieved) if return_details else knob
