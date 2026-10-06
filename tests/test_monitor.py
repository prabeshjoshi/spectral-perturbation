"""Monitoring layer: library wiring, ARL calibration, benchmark harness."""
import warnings

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

pytest.importorskip("chemotools")
pytest.importorskip("menelaus")
pytest.importorskip("river")
pytest.importorskip("mapie")

from specperturb.drift import mechanisms as M  # noqa: E402
from specperturb.drift import profiles as T  # noqa: E402
from specperturb.drift import DriftScenario  # noqa: E402
from specperturb.drift.benchmark import MonitoringBenchmark  # noqa: E402
from specperturb.drift.monitor import (ADWIN, CUSUM, EWMA, Martingale, PageHinkley,  # noqa: E402
                                       Shewhart, bootstrap_sequences, calibrate, dmodx,
                                       hotelling_t2, q_residuals)

from test_drift import X_AX, X_CAL, Y_CAL, spectra  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
S_REF = RNG.chisquare(3, 1000)
S_CAL = RNG.chisquare(3, 1000)


def realized_arl(det, knob, n=300, horizon=250, seed=99):
    seqs = bootstrap_sequences(S_CAL, n, horizon, seed)
    return np.mean([horizon if (a := det.first_alarm(q, knob)) is None else a + 1 for q in seqs])


@pytest.mark.parametrize("det", [EWMA(0.2), PageHinkley(), CUSUM(0.5)],
                         ids=lambda d: type(d).__name__)
def test_calibration_hits_target_arl(det):
    det = det.setup(S_REF)
    knob, achieved = calibrate(det, S_CAL, arl0=50, n_sim=150, rng=0, return_details=True)
    assert achieved == pytest.approx(50, rel=0.15)
    assert realized_arl(det, knob) == pytest.approx(50, rel=0.25)  # fresh sequences


def test_adwin_calibrates_in_log_space_and_warns_when_out_of_range():
    det = ADWIN().setup(S_REF)
    knob, achieved = calibrate(det, S_CAL, arl0=2000, n_sim=40, rng=0, return_details=True)
    assert 0 < knob < 0.5 and achieved == pytest.approx(2000, rel=0.25)
    with pytest.warns(UserWarning, match="outside the detector's range"):
        calibrate(det, S_CAL, arl0=60, n_sim=40, rng=0)   # ADWIN cannot alarm this often


@pytest.mark.parametrize("det", [Martingale(), EWMA(0.2), CUSUM(0.5)], ids=lambda d: type(d).__name__)
def test_fap_calibration(det):
    det = det.setup(S_REF)
    knob, achieved = calibrate(det, S_CAL, n_sim=300, rng=0, return_details=True,
                               criterion="fap", window=100, fap=0.1)
    assert achieved == pytest.approx(0.1, abs=0.03)
    seqs = bootstrap_sequences(S_CAL, 400, 100, 42)
    realized = np.mean([det.first_alarm(q, knob) is not None for q in seqs])
    assert realized == pytest.approx(0.1, abs=0.05)


def test_martingale_arl_is_horizon_dependent():
    # why martingales need criterion="fap": many null runs never alarm
    m = Martingale().setup(S_REF)
    knob = calibrate(m, S_CAL, arl0=50, n_sim=150, rng=0)
    short, long = realized_arl(m, knob, horizon=250), realized_arl(m, knob, horizon=1000)
    assert long > 1.5 * short


def test_cusum_beats_shewhart_on_small_shift_at_matched_arl():
    sh, cu = Shewhart().setup(S_REF), CUSUM(0.25).setup(S_REF)
    h_sh = calibrate(sh, S_CAL, arl0=100, n_sim=200, rng=1)
    h_cu = calibrate(cu, S_CAL, arl0=100, n_sim=200, rng=1)
    shifted = bootstrap_sequences(S_CAL, 200, 400, 5) + 0.6 * S_CAL.std()
    d_sh = np.mean([a if (a := sh.first_alarm(q, h_sh)) is not None else 400 for q in shifted])
    d_cu = np.mean([a if (a := cu.first_alarm(q, h_cu)) is not None else 400 for q in shifted])
    assert d_cu < 0.7 * d_sh


def test_martingale_null_respects_ville():
    m = Martingale().setup(S_REF)
    seqs = bootstrap_sequences(S_CAL, 200, 300, 7)
    exceed = np.mean([m.path(q).max() > 20 for q in seqs])   # alpha = 0.05
    assert exceed <= 0.08


def test_block_bootstrap_keeps_autocorrelation():
    from specperturb.drift.profiles import AR1
    s = AR1(0.9, 1.0)(np.arange(5000), rng=0)
    lag1 = lambda q: np.corrcoef(q[:, :-1].ravel(), q[:, 1:].ravel())[0, 1]  # noqa: E731
    assert lag1(bootstrap_sequences(s, 50, 200, 0, block=1)) < 0.1
    assert lag1(bootstrap_sequences(s, 50, 200, 0, block=25)) > 0.7


def _model_and_data(n_ic=400):
    model = PLSRegression(3, scale=False).fit(X_CAL, Y_CAL)
    X_ic, _ = spectra(n_ic, 11)
    X_pool, _ = spectra(max(400, n_ic), 12)
    return model, X_ic, X_pool


def test_chemotools_statistics_and_q_dmodx_equivalence():
    model, X_ic, X_pool = _model_and_data()
    stats = {k: f(model).fit(X_CAL) for k, f in [("T2", hotelling_t2), ("Q", q_residuals), ("DModX", dmodx)]}
    sq, sd = stats["Q"](X_pool), stats["DModX"](X_pool)
    assert np.all(np.diff(sd[np.argsort(sq)]) >= -1e-12)   # DModX monotone in Q
    # identical calibrated alarms on the same drifted stream
    drift = DriftScenario([(M.Slope(), T.Ramp(30, 2e-4))]).simulate(X_pool[:150], x=X_AX).X
    alarms = []
    for k in ("Q", "DModX"):
        det = Shewhart().setup(stats[k](X_ic[:200]))
        h = calibrate(det, stats[k](X_ic[200:]), arl0=40, n_sim=100, rng=3)
        alarms.append(det.first_alarm(stats[k](drift), h))
    assert alarms[0] == alarms[1]


def test_benchmark_end_to_end():
    model, X_ic, X_pool = _model_and_data()
    film = np.exp(-0.5 * ((X_AX - 1700) / 30) ** 2)
    scen = {
        "none": None,
        "lamp aging": DriftScenario([(M.Gain(), T.Saturating(60, -0.3, tau=40))]),
        "fouling": DriftScenario([(M.Fouling(film), T.Ramp(60, 0.002))]),
    }
    bm = MonitoringBenchmark(model, harm_threshold=0.05, arl0=40, n_sim=60,
                             detectors={"EWMA": EWMA(0.2), "CUSUM": CUSUM(0.5), "Martingale": Martingale()},
                             x=X_AX, random_state=0).fit(X_CAL, X_ic)
    assert all(v == pytest.approx(40, rel=0.3) for v in bm.achieved_.values())
    res = bm.run(scen, X_pool, n_steps=150, n_runs=3)
    assert len(res.rows) == 3 * 3 * 2 * 3
    for r in res.rows:
        if r["scenario"] == "none":
            assert r["onset"] is None and not r["detected"]
        if r["detected"]:
            assert r["delay"] >= 0 and r["alarm"] >= r["onset"]
        if r["lead"] is not None:
            assert r["lead"] == r["harm_onset"] - r["alarm"]
    summ = res.summary()
    assert {"false_alarm_rate", "median_delay", "warned_before_harm", "median_lead"} <= set(summ[0])
    real = bm.verify_calibration(X_pool, n_runs=20)
    assert set(real) == set(bm.monitors_)


def test_benchmark_fap_mode():
    # calibration accuracy is limited by the in-control set size: with a few
    # hundred spectra the T2 tail is under-sampled and realized FAP is ~3x target
    model, X_ic, X_pool = _model_and_data(n_ic=4000)
    scen = {"step": DriftScenario([(M.Offset(), T.Step(60, 0.05))])}
    bm = MonitoringBenchmark(model, harm_threshold=0.05, criterion="fap", window=60, fap=0.1, n_sim=150,
                             detectors={"Martingale": Martingale(), "CUSUM": CUSUM(0.5)},
                             x=X_AX, random_state=0).fit(X_CAL, X_ic)
    assert all(v == pytest.approx(0.1, abs=0.04) for v in bm.achieved_.values())
    real = bm.verify_calibration(X_pool, n_runs=300)
    assert all(v == pytest.approx(0.1, abs=0.06) for v in real.values())
    res = bm.run(scen, X_pool, n_steps=150, n_runs=4)
    assert len(res.summary()) == 4
