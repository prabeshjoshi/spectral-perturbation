import warnings

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from specperturb.drift import mechanisms as M
from specperturb.drift import profiles as T
from specperturb.drift import (DriftScenario, PairedInstrumentDrift, ReplicateDrift,
                               harm_onset, model_truth)

X_AX = np.linspace(1100, 2500, 150)
BANDS = np.vstack([np.exp(-0.5 * ((X_AX - c) / w) ** 2) for c, w in [(1450, 40), (1940, 50), (2100, 60)]])


def spectra(n, seed):
    rng = np.random.default_rng(seed)
    C = rng.uniform(0.2, 1.0, (n, 3))
    return C @ BANDS + 0.1 + rng.normal(0, 2e-3, (n, len(X_AX))), C[:, 0]


X_CAL, Y_CAL = spectra(80, 0)
X_STREAM, _ = spectra(200, 1)


def all_mechanisms():
    film = np.exp(-0.5 * ((X_AX - 1700) / 30) ** 2)
    return [M.Offset(), M.Slope(), M.Curvature(), M.Gain(), M.Gain(domain="intensity"),
            M.WavelengthShift(), M.WavelengthStretch(), M.Bandwidth(), M.NoiseIncrease(),
            M.StrayLightDrift(), M.TemperatureDrift(shift_per_degree=-0.5, broadening_per_degree=1.0,
                                                    region=(1350, 1550)),
            M.Fouling(film), M.Interferent(film), M.ScatterChange()]


# --------------------------------------------------------------------------- #
#  Mechanisms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mech", all_mechanisms(), ids=lambda m: m.name + getattr(m, "domain", ""))
def test_theta_zero_is_identity(mech):
    np.testing.assert_array_equal(mech.apply(X_STREAM[:5], 0.0, x=X_AX, rng=0), X_STREAM[:5])


@pytest.mark.parametrize("mech", all_mechanisms(), ids=lambda m: m.name + getattr(m, "domain", ""))
def test_mechanism_changes_spectra_and_documents_parameter(mech):
    theta = 2.0 if isinstance(mech, (M.Bandwidth, M.WavelengthShift, M.TemperatureDrift)) else 0.01
    out = mech.apply(X_STREAM[:5], theta, x=X_AX, rng=0)
    assert out.shape == (5, len(X_AX)) and not np.allclose(out, X_STREAM[:5])
    assert mech.parameter and mech.stage in M.STAGES


def test_per_row_theta():
    th = np.array([0.0, 0.1, 0.2])
    out = M.Offset().apply(X_STREAM[:3], th)
    np.testing.assert_allclose(out - X_STREAM[:3], th[:, None] * np.ones((1, len(X_AX))))


def test_baseline_shapes():
    z = np.zeros((1, 101))
    ax = np.linspace(0, 1, 101)
    s = M.Slope().apply(z, 0.2, x=ax)[0]
    assert s[50] == pytest.approx(0) and s[-1] == pytest.approx(0.2) and s[0] == pytest.approx(-0.2)
    c = M.Curvature().apply(z, 0.3, x=ax)[0]
    assert c[50] == pytest.approx(0) and c[0] == pytest.approx(0.3) and c[-1] == pytest.approx(0.3)


def test_gain_domains():
    out = M.Gain().apply(X_STREAM[:2], -0.1)
    np.testing.assert_allclose(out - X_STREAM[:2], -np.log10(0.9))
    out = M.Gain(domain="intensity").apply(X_STREAM[:2], -0.1)
    np.testing.assert_allclose(out, 0.9 * X_STREAM[:2])


def test_stray_light_compresses_high_absorbance():
    A = np.array([[0.1, 1.0, 2.0, 3.0]])
    s = 1e-3
    out = M.StrayLightDrift().apply(A, s)
    np.testing.assert_allclose(out, -np.log10((10 ** -A + s) / (1 + s)))
    drop = A - out
    assert np.all(np.diff(drop[0]) > 0)  # larger loss at higher absorbance


def test_shift_moves_peak():
    out = M.WavelengthShift().apply(X_STREAM[:1], 20.0, x=X_AX)
    peak = lambda r: X_AX[np.argmax(r[(X_AX > 1350) & (X_AX < 1550)]) + np.argmax((X_AX > 1350))]  # noqa: E731
    assert peak(out[0]) > peak(X_STREAM[0])


def test_bandwidth_lowers_peaks_keeps_area():
    out = M.Bandwidth().apply(BANDS[:1], 8.0, x=X_AX)
    assert out.max() < BANDS[0].max()
    assert out.sum() == pytest.approx(BANDS[0].sum(), rel=1e-3)
    with pytest.raises(ValueError):
        M.Bandwidth().apply(BANDS[:1], -1.0)


def test_temperature_only_in_region():
    mech = M.TemperatureDrift(shift_per_degree=-1.0, broadening_per_degree=2.0, region=(1350, 1550))
    out = mech.apply(X_STREAM[:3], 5.0, x=X_AX)
    outside = (X_AX < 1350) | (X_AX > 1550)
    np.testing.assert_array_equal(out[:, outside], X_STREAM[:3, outside])
    assert not np.allclose(out[:, ~outside], X_STREAM[:3, ~outside])
    with pytest.raises(ValueError, match="shift_per_degree"):
        import specperturb as sp
        sp.Temperature()


def test_additive_spectra_and_scatter():
    f = np.linspace(0, 1, len(X_AX))
    np.testing.assert_allclose(M.Fouling(f).apply(X_STREAM[:2], 0.3) - X_STREAM[:2], np.tile(0.3 * f, (2, 1)))
    np.testing.assert_allclose(M.ScatterChange().apply(X_STREAM[:2], 0.1), 1.1 * X_STREAM[:2])


def test_noise_is_stochastic_with_right_sd():
    m = M.NoiseIncrease()
    assert m.stochastic
    d = m.apply(np.zeros((200, 150)), 0.01, rng=0)
    assert d.std() == pytest.approx(0.01, rel=0.05)


# --------------------------------------------------------------------------- #
#  Profiles
# --------------------------------------------------------------------------- #
TT = np.arange(0, 100, 1.0)


def test_basic_profiles():
    np.testing.assert_array_equal(T.Step(10, 2.0)(TT), np.where(TT >= 10, 2.0, 0.0))
    r = T.Ramp(10, 0.5, cap=5)(TT)
    assert r[10] == 0 and r[12] == 1.0 and r.max() == 5
    s = T.Saturating(10, 1.0, tau=5)(TT)
    assert s[9] == 0 and 0.6 < s[15] < 0.65 and s[-1] == pytest.approx(1.0, abs=1e-6)
    p = T.Periodic(1.0, 20, onset=30)(TT)
    assert np.all(p[:30] == 0) and np.abs(p[30:]).max() > 0.9


def test_onsets_and_composition():
    assert T.Step(10, 1).onset == 10
    assert (T.Step(10, 1) + T.Ramp(30, 1)).onset == 10
    assert (T.Step(10, 1) * T.Ramp(30, 1)).onset == 30
    assert (T.Step(10, 1) + T.AR1(0.5, 0.1)).onset == 10
    assert T.AR1(0.5, 0.1).onset is None and T.Periodic(1, 10).onset is None
    comp = 2 * T.Step(10, 1.0) - 0.5
    np.testing.assert_allclose(comp(TT), np.where(TT >= 10, 1.5, -0.5))


def test_random_profiles():
    rw = T.RandomWalk(40, 0.1)
    a, b = rw(TT, rng=3), rw(TT, rng=3)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:40] == 0) and np.any(a[41:] != 0)
    it = T.Intermittent(20, rate=0.1, duration=3, size=2.0)(TT, rng=1)
    assert set(np.unique(it)) <= {0.0, 2.0} and np.all(it[:20] == 0) and it.max() == 2.0
    ar = T.AR1(0.8, 1.0)(np.arange(20000), rng=0)
    assert ar.std() == pytest.approx(1.0, rel=0.05)
    assert np.corrcoef(ar[:-1], ar[1:])[0, 1] == pytest.approx(0.8, abs=0.03)


# --------------------------------------------------------------------------- #
#  Scenarios
# --------------------------------------------------------------------------- #
def test_scenario_physical_order_and_names():
    sc = DriftScenario([(M.Offset(), 0.1), (M.StrayLightDrift(), 0.01), (M.Offset(), 0.2)])
    assert [type(m).__name__ for m, _ in sc.components] == ["StrayLightDrift", "Offset", "Offset"]
    assert sc.names == ["StrayLightDrift", "Offset", "Offset_2"]
    given = DriftScenario([(M.Offset(), 0.1), (M.StrayLightDrift(), 0.01)], order="given")
    a = sc.simulate(X_STREAM[:3]).X
    b = DriftScenario([(M.Offset(), 0.1), (M.Offset(), 0.2), (M.StrayLightDrift(), 0.01)], order="given").simulate(X_STREAM[:3]).X
    assert not np.allclose(a, b)  # order matters for non-linear mechanisms
    assert given.names == ["Offset", "StrayLightDrift"]


def test_scenario_result_ground_truth():
    sc = DriftScenario([(M.Offset(), T.Step(50, 0.05)), (M.NoiseIncrease(), T.Step(50, 0.01))])
    r = sc.simulate(X_STREAM, x=X_AX, rng=0)
    assert r.onset == 50 and set(r.theta) == {"Offset", "NoiseIncrease"}
    np.testing.assert_array_equal(r.X[:50], X_STREAM[:50])
    np.testing.assert_allclose(r.X_systematic[50:] - X_STREAM[50:], 0.05)   # noise excluded
    assert (r.X - r.X_systematic)[50:].std() == pytest.approx(0.01, rel=0.1)
    assert np.all(r.size[:50] == 0) and np.all(r.size[50:] > 0)
    a = sc.simulate(X_STREAM, x=X_AX, rng=0).X
    np.testing.assert_array_equal(a, r.X)


# --------------------------------------------------------------------------- #
#  Empirical surrogates
# --------------------------------------------------------------------------- #
def test_paired_instrument_surrogates():
    pytest.importorskip("chemotools")
    A, _ = spectra(30, 5)
    B = 1.05 * A + 0.02 + 0.01 * np.linspace(-1, 1, len(X_AX))
    mean = PairedInstrumentDrift(A, B, method="mean")
    np.testing.assert_allclose(mean.apply(A, 1.0), A + (B - A).mean(0))
    ds = PairedInstrumentDrift(A, B, method="ds")
    np.testing.assert_allclose(ds.apply(A, 1.0), B, atol=1e-6)
    np.testing.assert_array_equal(ds.apply(A, 0.0), A)
    np.testing.assert_allclose(ds.apply(A, 0.5), A + 0.5 * (B - A), atol=1e-6)
    assert ds.surrogate and "paired-instrument" in ds.source
    pds = PairedInstrumentDrift(A, B, method="pds", pds_kwargs=dict(window_length=11, n_components=2))
    assert pds.apply(A, 1.0).shape == A.shape
    r = DriftScenario([(ds, T.Step(5, 1.0))]).simulate(A[:10])
    assert r.surrogate == {"PairedInstrumentDrift": True}
    assert r.source["PairedInstrumentDrift"].startswith("paired-instrument")


def test_replicate_drift():
    rng = np.random.default_rng(0)
    base, _ = spectra(20, 7)
    R = np.repeat(base, 3, axis=0) + rng.normal(0, 0.01, (60, 1)) * np.linspace(0, 1, len(X_AX))
    g = np.repeat(np.arange(20), 3)
    d = ReplicateDrift(R, g)
    from specperturb.replicate import ReplicateNoiseModel
    S = ReplicateNoiseModel().fit(R, groups=g).covariance_
    assert np.linalg.norm(d.direction_) ** 2 == pytest.approx(np.linalg.eigvalsh(S).max(), rel=1e-8)
    np.testing.assert_allclose(d.apply(base[:2], 2.0) - base[:2], np.tile(2.0 * d.direction_, (2, 1)))
    s1 = ReplicateDrift(R, g, mode="sample", seed=3).direction_
    s2 = ReplicateDrift(R, g, mode="sample", seed=3).direction_
    np.testing.assert_array_equal(s1, s2)
    assert d.surrogate and d.source.startswith("replicate-learned")


# --------------------------------------------------------------------------- #
#  Ground truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("with_pre", [False, True])
def test_truth_split_and_residual_is_harmless(scale, with_pre):
    pls = PLSRegression(3, scale=scale)
    model = make_pipeline(StandardScaler(), pls) if with_pre else pls
    model.fit(X_CAL, Y_CAL)
    sc = DriftScenario([(M.Slope(), 0.02), (M.WavelengthShift(), 3.0), (M.StrayLightDrift(), 0.01)])
    r = sc.simulate(X_STREAM[:20], x=X_AX)
    tr = model_truth(model, r.X_clean, r.X)
    pre = (lambda X: model[:-1].transform(X)) if with_pre else (lambda X: X)  # noqa: E731
    final = model[-1] if with_pre else model
    dz = pre(r.X) - pre(r.X_clean)
    np.testing.assert_allclose(tr.delta_in + tr.delta_res, dz, atol=1e-10)
    zc = pre(r.X_clean)
    # residual part changes neither scores nor predictions
    np.testing.assert_allclose(final.transform(zc + tr.delta_res), final.transform(zc), atol=1e-8)
    np.testing.assert_allclose(final.predict(zc + tr.delta_res), final.predict(zc), atol=1e-8)
    np.testing.assert_allclose(tr.bias, model.predict(r.X).ravel() - model.predict(r.X_clean).ravel())
    assert np.all((tr.residual_share >= 0) & (tr.residual_share <= 1))


def test_drift_orthogonal_to_model_causes_no_bias():
    pls = PLSRegression(3, scale=False).fit(X_CAL, Y_CAL)
    P = pls.x_loadings_
    rng = np.random.default_rng(0)
    d = rng.normal(size=len(X_AX))
    W = pls.x_rotations_
    d -= P @ np.linalg.solve(W.T @ P, W.T @ d)   # remove the in-model part (oblique projection)
    Xd = X_STREAM[:10] + 0.05 * d
    tr = model_truth(pls, X_STREAM[:10], Xd)
    np.testing.assert_allclose(tr.bias, 0, atol=1e-10)
    np.testing.assert_allclose(tr.residual_share, 1.0, atol=1e-10)


def test_harm_onset():
    b = np.array([0, 0.1, 0.3, 0.1, 0.3, 0.3, 0.3, 0.1])
    assert harm_onset(b, 0.2) == 2
    assert harm_onset(b, 0.2, persistence=3) == 4
    assert harm_onset(b, 1.0) is None
    assert harm_onset(-b, 0.2) == 2


def test_systematic_bias_excludes_noise():
    pls = PLSRegression(3, scale=False).fit(X_CAL, Y_CAL)
    r = DriftScenario([(M.NoiseIncrease(), 0.01)]).simulate(X_STREAM[:20], rng=0)
    tr = model_truth(pls, r.X_clean, r.X, r.X_systematic)
    np.testing.assert_allclose(tr.bias_systematic, 0, atol=1e-12)
    assert np.abs(tr.bias).max() > 0
