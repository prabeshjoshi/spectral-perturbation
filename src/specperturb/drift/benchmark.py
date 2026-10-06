"""Benchmark harness: monitors x scenarios, at matched in-control ARL.

Workflow
--------
1. ``fit(X_cal, X_incontrol)``: fit each statistic on the calibration set;
   split the in-control set (spectra NOT used to build the model) into a
   reference half (detector mean / sd / p-values) and a calibration half
   (bootstrap run lengths). Calibrate every (statistic, detector) to the same
   in-control ARL0.
2. ``run(scenarios, X_pool, n_steps, n_runs)``: each run draws an in-control
   stream from ``X_pool`` (also unused in model building), applies a scenario,
   computes ground truth and every monitor's first alarm.
3. ``result.summary()``: false alarms before onset, detection delay, and alarm
   vs harm: was the alarm raised before the prediction bias became harmful?

Outcome per run and monitor
---------------------------
false_alarm : alarm before drift onset (or in a no-drift scenario)
detected    : alarm at/after onset;   delay = alarm - onset (spectra)
missed      : no alarm although drift started
harm        : |systematic bias| > harm_threshold (for ``persistence`` spectra)
lead        : harm_onset - alarm (> 0: warned before harm, < 0: alarm came late)
nuisance    : detected but the drift never became harmful
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .monitor import Detector, EWMA, Martingale, PageHinkley, Shewhart, CUSUM, calibrate, hotelling_t2, q_residuals
from .scenario import DriftScenario
from .truth import harm_onset, model_truth

__all__ = ["MonitoringBenchmark", "BenchmarkResult", "default_detectors"]


def default_detectors():
    return {"Shewhart": Shewhart(), "EWMA": EWMA(0.2), "CUSUM": CUSUM(0.5),
            "PageHinkley": PageHinkley(), "Martingale": Martingale()}


@dataclass
class BenchmarkResult:
    rows: list = field(default_factory=list)
    arl0: float = None

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def summary(self):
        """One row per (scenario, statistic, detector)."""
        groups = {}
        for r in self.rows:
            groups.setdefault((r["scenario"], r["statistic"], r["detector"]), []).append(r)
        out = []
        for (sc, st, de), rs in groups.items():
            fa = [r["false_alarm"] for r in rs]
            det = [r for r in rs if r["detected"]]
            harmed = [r for r in rs if r["harm_onset"] is not None]
            warned = [r for r in harmed if r["detected"] and r["alarm"] <= r["harm_onset"]]
            harmless = [r for r in rs if r["onset"] is not None and r["harm_onset"] is None
                        and not r["false_alarm"]]
            med = lambda v: float(np.median(v)) if len(v) else np.nan  # noqa: E731
            out.append(dict(
                scenario=sc, statistic=st, detector=de, n_runs=len(rs),
                false_alarm_rate=float(np.mean(fa)),
                detection_rate=len(det) / max(1, sum(r["onset"] is not None and not r["false_alarm"] for r in rs)),
                median_delay=med([r["delay"] for r in det]),
                harm_rate=len(harmed) / len(rs),
                warned_before_harm=len(warned) / len(harmed) if harmed else np.nan,
                median_lead=med([r["lead"] for r in harmed if r["lead"] is not None]),
                nuisance_alarm_rate=(np.mean([r["detected"] for r in harmless]) if harmless else np.nan),
            ))
        return out


class MonitoringBenchmark:
    """Compare drift monitors at matched false-alarm rate, against ground truth.

    Parameters
    ----------
    model : fitted PLS model or Pipeline ending in PLS (raw spectra in)
    statistics : {name: Statistic}; default T2 and Q of ``model``
    detectors : {name: Detector}; default :func:`default_detectors`
    arl0 : target in-control average run length (spectra)
    harm_threshold : |bias| above which a prediction is harmful, in y units
        (from the method's fitness for purpose, e.g. allowed bias or k x RMSEP)
    persistence : consecutive spectra above the threshold that define harm onset
    n_sim, block : bootstrap settings for calibration (block > 1 keeps
        autocorrelation of in-control statistics)
    in_control : optional Perturbation applied to every in-control spectrum
        (e.g. ReplicateNoise), both for calibration and for streams
    x : spectral axis passed to perturbations and scenarios
    criterion : 'arl' (match in-control ARL = ``arl0``) or 'fap' (match
        P(false alarm within ``window`` spectra) = ``fap``; set ``window`` to the
        drift onset index to match the pre-onset false-alarm rate). Use 'fap'
        whenever conformal martingales are among the detectors.
    """

    def __init__(self, model, harm_threshold, statistics=None, detectors=None, arl0=200,
                 persistence=1, n_sim=200, block=1, in_control=None, x=None, random_state=None,
                 criterion="arl", window=None, fap=0.05):
        self.model = model
        self.harm_threshold = harm_threshold
        self.statistics = statistics
        self.detectors = detectors
        self.arl0 = arl0
        self.persistence = persistence
        self.n_sim, self.block = n_sim, block
        self.in_control, self.x = in_control, x
        self.random_state = random_state
        self.criterion, self.window, self.fap = criterion, window, fap

    def _ic(self, X, rng):
        return X if self.in_control is None else self.in_control(X, x=self.x, rng=rng)

    def fit(self, X_cal, X_incontrol):
        rng = np.random.default_rng(self.random_state)
        self.rng_ = rng
        stats = self.statistics or {"T2": hotelling_t2(self.model), "Q": q_residuals(self.model)}
        dets = self.detectors or default_detectors()
        self.statistics_ = {k: s.fit(X_cal) for k, s in stats.items()}

        X_ic = self._ic(np.asarray(X_incontrol, float), rng)
        perm = rng.permutation(len(X_ic))
        ref, cal = X_ic[perm[: len(perm) // 2]], X_ic[perm[len(perm) // 2:]]
        self.monitors_ = {}
        self.achieved_ = {}
        for sname, stat in self.statistics_.items():
            s_ref, s_cal = stat(ref), stat(cal)
            for dname, det in dets.items():
                d = det.clone().setup(s_ref)
                knob, achieved = calibrate(d, s_cal, self.arl0, self.n_sim, block=self.block,
                                           rng=rng, return_details=True, criterion=self.criterion,
                                           window=self.window, fap=self.fap)
                self.monitors_[(sname, dname)] = (d, knob)
                self.achieved_[(sname, dname)] = achieved
        return self

    def verify_calibration(self, X_pool, n_runs=200):
        """Out-of-sample check of the calibration on fresh in-control streams.

        Returns {(statistic, detector): realized metric}: mean run length
        (censored at 5 x arl0) for criterion='arl', or the fraction of streams
        with a false alarm within ``window`` for criterion='fap'.
        """
        rng = np.random.default_rng(self.rng_.integers(2 ** 63))
        horizon = int(5 * self.arl0) if self.criterion == "arl" else int(self.window)
        X_pool = np.asarray(X_pool, float)
        out = {k: [] for k in self.monitors_}
        for _ in range(n_runs):
            X = self._ic(X_pool[rng.integers(0, len(X_pool), horizon)], rng)
            svals = {k: s(X) for k, s in self.statistics_.items()}
            for (sname, dname), (det, knob) in self.monitors_.items():
                a = det.first_alarm(svals[sname], knob)
                out[(sname, dname)].append(a)          # None = no alarm within horizon
        if self.criterion == "arl":
            return {k: float(np.mean([horizon if a is None else a + 1 for a in v])) for k, v in out.items()}
        return {k: float(np.mean([a is not None for a in v])) for k, v in out.items()}

    def run(self, scenarios, X_pool, n_steps, n_runs, t=None):
        """Simulate ``n_runs`` streams of ``n_steps`` spectra per scenario.

        scenarios : {name: DriftScenario or None}; None is a no-drift control
        X_pool    : in-control spectra the streams are drawn from (with replacement)
        """
        rng = np.random.default_rng(self.rng_.integers(2 ** 63))
        X_pool = np.asarray(X_pool, float)
        t = np.arange(n_steps, dtype=float) if t is None else np.asarray(t, float)
        result = BenchmarkResult(arl0=self.arl0)
        for sc_name, scenario in scenarios.items():
            for run in range(n_runs):
                X_clean = self._ic(X_pool[rng.integers(0, len(X_pool), n_steps)], rng)
                if scenario is None:
                    X, X_sys, onset = X_clean, X_clean, None
                else:
                    if not isinstance(scenario, DriftScenario):
                        raise TypeError(f"{sc_name}: expected DriftScenario or None")
                    res = scenario.simulate(X_clean, t=t, x=self.x, rng=rng)
                    X, X_sys, onset = res.X, res.X_systematic, res.onset
                truth = model_truth(self.model, X_clean, X, X_sys)
                h_on = harm_onset(truth.bias_systematic, self.harm_threshold, self.persistence)
                svals = {k: s(X) for k, s in self.statistics_.items()}
                for (sname, dname), (det, knob) in self.monitors_.items():
                    alarm = det.first_alarm(svals[sname], knob)
                    false_alarm = alarm is not None and (onset is None or alarm < onset)
                    detected = alarm is not None and not false_alarm
                    result.rows.append(dict(
                        scenario=sc_name, run=run, statistic=sname, detector=dname,
                        onset=onset, harm_onset=h_on, alarm=alarm, n_steps=n_steps,
                        false_alarm=false_alarm, detected=detected,
                        delay=(alarm - onset) if detected else None,
                        lead=(h_on - alarm) if (detected and h_on is not None) else None,
                        residual_share_at_alarm=(float(truth.residual_share[alarm])
                                                 if detected else None),
                        bias_at_alarm=(float(truth.bias_systematic[alarm]) if detected else None),
                    ))
        return result
