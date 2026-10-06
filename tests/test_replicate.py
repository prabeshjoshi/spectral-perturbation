import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import specperturb as sp
from specperturb.replicate import (ReplicateAugmenter, ReplicateEPO, ReplicateGLSW,
                                   ReplicateNoise, ReplicateNoiseModel,
                                   prediction_repeatability, replicate_differences,
                                   scatter_share)

from conftest import NUISANCE, make_scans

SG1 = FunctionTransformer(savgol_filter, kw_args=dict(window_length=11, polyorder=2, deriv=1, axis=1))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.ravel(a) - np.ravel(b)) ** 2)))


# --------------------------------------------------------------------------- #
#  Numerical equivalence with the original module
# --------------------------------------------------------------------------- #
class TestMatchesOriginal:
    def test_differences_and_covariance(self, ref):
        X, g, _ = make_scans(10, 4, np.random.default_rng(5), outlier=True)
        np.testing.assert_array_equal(replicate_differences(X, g), ref.replicate_differences(X, g))
        np.testing.assert_array_equal(replicate_differences(X, g, exclude_outliers=False),
                                      ref.replicate_differences(X, g, exclude_outliers=False))
        new = ReplicateNoiseModel().fit(X, groups=g)
        old = ref.ReplicateNoiseModel().fit(X, g)
        np.testing.assert_array_equal(new.covariance_, old.covariance_)
        np.testing.assert_array_equal(new.sample(7, 3), old.sample(7, 3))

    def test_augmenter(self, ref, library, calib):
        R, g, _ = library
        X, _, y = calib
        new = ReplicateAugmenter(scale=0.7, random_state=11).fit(R, groups=g)
        old = ref.ReplicateAugmenter(scale=0.7, random_state=11).fit(R, g)
        np.testing.assert_array_equal(new.transform(X), old.transform(X))
        for a, b in zip(new.augment(X, y, 4), old.augment(X, y, 4)):
            np.testing.assert_array_equal(a, b)

    def test_replicate_noise_perturbation_matches_augmenter(self, ref, library, calib):
        R, g, _ = library
        X, _, y = calib
        old = ref.ReplicateAugmenter(random_state=11).fit(R, g).augment(X, y, 5)
        new = sp.augment_dataset(ReplicateNoise(R, g), X, y, n_copies=5, rng=11)
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("cls,kw", [("ReplicateGLSW", dict(alpha=1e-3)),
                                        ("ReplicateEPO", dict(n_components=3))])
    def test_filters(self, ref, library, calib, cls, kw):
        R, g, _ = library
        X, _, _ = calib
        old = getattr(ref, cls)(**kw).fit(R, groups=g)
        via_fit = getattr(sp.replicate, cls)(**kw).fit(R, groups=g)
        via_ctor = getattr(sp.replicate, cls)(replicates=R, replicate_groups=g, **kw).fit(X)
        for new in (via_fit, via_ctor):
            np.testing.assert_array_equal(new.filter_, old.filter_)
            np.testing.assert_array_equal(new.transform(X), old.transform(X))

    def test_filter_with_preprocessor_matches_manual_pipeline(self, ref, library, calib):
        R, g, _ = library
        X, _, _ = calib
        sg = SG1.fit(X)
        old = ref.ReplicateGLSW(alpha=1e-4).fit(sg.transform(R), groups=g)
        new = ReplicateGLSW(alpha=1e-4, replicates=R, replicate_groups=g, preprocessor=SG1).fit(X)
        np.testing.assert_allclose(new.transform(X), old.transform(sg.transform(X)), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("method", ["analytic", "mc"])
    def test_repeatability(self, ref, library, calib, method):
        R, g, _ = library
        X, _, y = calib
        model = make_pipeline(SG1, PLSRegression(3)).fit(X, y)
        nm_new = ReplicateNoiseModel().fit(R, groups=g)
        nm_old = ref.ReplicateNoiseModel().fit(R, g)
        a = prediction_repeatability(model.predict, X[:5], nm_new, method=method, n_draws=50, random_state=0)
        b = ref.prediction_repeatability(model.predict, X[:5], nm_old, method=method, n_draws=50, random_state=0)
        np.testing.assert_array_equal(a, b)

    def test_scatter_share(self, ref, calib):
        Xc, _, yc = calib
        Xt, gt, yt = make_scans(15, 3, np.random.default_rng(9))
        model = PLSRegression(3).fit(Xc, yc)
        assert scatter_share(model.predict, Xt, gt, yt) == ref.scatter_share(model.predict, Xt, gt, yt)


# --------------------------------------------------------------------------- #
#  Requested behaviour tests
# --------------------------------------------------------------------------- #
def test_shapes(library, calib):
    R, g, _ = library
    X, _, y = calib
    nm = ReplicateNoiseModel().fit(R, groups=g)
    assert nm.differences_.shape == (30 * 4 * 3, R.shape[1])
    assert nm.covariance_.shape == (R.shape[1],) * 2
    aug = ReplicateAugmenter(random_state=0).fit(R, groups=g)
    Xa, ya, ga = aug.augment(X, y, n_copies=3)
    assert Xa.shape == (len(X) * 4, X.shape[1]) and ya.shape == ga.shape == (len(X) * 4,)
    for f in (ReplicateGLSW(), ReplicateEPO(3)):
        assert f.fit(R, groups=g).transform(X).shape == X.shape


def test_reproducible_with_random_state(library, calib):
    R, g, _ = library
    X, _, _ = calib
    a = ReplicateAugmenter(random_state=4).fit(R, groups=g).transform(X)
    b = ReplicateAugmenter(random_state=4).fit(R, groups=g).transform(X)
    c = ReplicateAugmenter(random_state=5).fit(R, groups=g).transform(X)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    np.testing.assert_array_equal(ReplicateNoise(R, g)(X, rng=4), ReplicateNoise(R, g)(X, rng=4))


def test_glsw_large_alpha_is_identity(library):
    R, g, _ = library
    G = ReplicateGLSW(alpha=1e12).fit(R, groups=g).filter_
    np.testing.assert_allclose(G, np.eye(len(G)), atol=1e-8)


def test_glsw_shrinks_monotonically_with_alpha(library):
    R, g, _ = library
    u = NUISANCE[0]
    norms = [np.linalg.norm(u @ ReplicateGLSW(alpha=a).fit(R, groups=g).filter_)
             for a in (1e-5, 1e-3, 1e-1, 10)]
    assert np.all(np.diff(norms) > 0)


@pytest.mark.parametrize("k", [1, 3, 6])
def test_epo_removes_exactly_top_k(library, calib, k):
    R, g, _ = library
    X, _, _ = calib
    epo = ReplicateEPO(n_components=k).fit(R, groups=g)
    V = epo.eigenvectors_
    Xf = epo.transform(X)
    np.testing.assert_allclose(Xf @ V[:, :k], 0, atol=1e-10)           # removed
    np.testing.assert_allclose(Xf @ V[:, k:], X @ V[:, k:], atol=1e-10)  # rest untouched
    np.testing.assert_allclose(epo.filter_ @ epo.filter_, epo.filter_, atol=1e-10)  # projector


def test_analytic_equals_exact_and_mc_for_linear_model(library, calib):
    R, g, _ = library
    X, _, _ = calib
    b = np.random.default_rng(0).normal(size=R.shape[1])
    predict = lambda Z: Z @ b + 3.0  # noqa: E731
    nm = ReplicateNoiseModel().fit(R, groups=g)
    exact = np.sqrt(np.mean((nm.differences_ @ b) ** 2))
    analytic = prediction_repeatability(predict, X[:4], nm, method="analytic")
    mc = prediction_repeatability(predict, X[:4], nm, method="mc", n_draws=20000, random_state=0)
    np.testing.assert_allclose(analytic, exact, rtol=1e-6)
    np.testing.assert_allclose(mc, exact, rtol=0.03)


def test_repeatability_accepts_perturbation_object(library, calib):
    R, g, _ = library
    X, _, y = calib
    model = PLSRegression(3).fit(X, y)
    a = prediction_repeatability(model.predict, X[:3], ReplicateNoise(R, g))
    b = prediction_repeatability(model.predict, X[:3], ReplicateNoiseModel().fit(R, groups=g))
    np.testing.assert_array_equal(a, b)


class TestRecoversKnownNuisance:
    """Nuisance is 3 known directions + small diagonal noise."""

    def test_epo_subspace_matches_true_nuisance(self, library):
        R, g, _ = library
        Vk = ReplicateEPO(n_components=3).fit(R, groups=g).eigenvectors_[:, :3]
        Q, _ = np.linalg.qr(NUISANCE.T)
        cosines = np.linalg.svd(Q.T @ Vk, compute_uv=False)  # principal angles
        assert cosines.min() > 0.999

    def test_glsw_attenuates_nuisance_not_rest(self, library):
        R, g, _ = library
        # nuisance shapes overlap, so the weakest direction in their span has
        # ~6 % of the top variance; alpha=1e-3 still sits far above the diagonal noise
        G = ReplicateGLSW(alpha=1e-3).fit(R, groups=g).filter_
        Q, _ = np.linalg.qr(NUISANCE.T)
        in_span = np.linalg.norm(Q.T @ G @ Q, 2)
        rng = np.random.default_rng(0)
        v = rng.normal(size=(G.shape[0], 5))
        v -= Q @ (Q.T @ v)
        v /= np.linalg.norm(v, axis=0)
        out_span = np.linalg.norm(G @ v, axis=0)
        assert in_span < 0.2
        assert out_span.min() > 0.9

    def test_filters_improve_prediction(self, library, calib, test_set):
        R, g, _ = library
        Xc, _, yc = calib
        Xt, _, yt = test_set
        base = rmse(PLSRegression(2).fit(Xc, yc).predict(Xt), yt)
        for f in (ReplicateGLSW(alpha=1e-3, replicates=R, replicate_groups=g),
                  ReplicateEPO(3, replicates=R, replicate_groups=g)):
            m = make_pipeline(f, PLSRegression(2)).fit(Xc, yc)
            assert rmse(m.predict(Xt), yt) < 0.5 * base


# --------------------------------------------------------------------------- #
#  scikit-learn behaviour
# --------------------------------------------------------------------------- #
def test_filters_clone_and_grid_search(library, calib):
    R, g, _ = library
    Xc, _, yc = calib
    pipe = make_pipeline(ReplicateGLSW(replicates=R, replicate_groups=g, preprocessor=SG1),
                         PLSRegression(2))
    c = clone(pipe)
    assert c.get_params()["replicateglsw__alpha"] == 1e-3
    gs = GridSearchCV(pipe, {"replicateglsw__alpha": [1e-4, 1e-2]}, cv=KFold(4)).fit(Xc, yc)
    assert gs.best_params_["replicateglsw__alpha"] in (1e-4, 1e-2)
    assert not hasattr(pipe.steps[0][1], "filter_")  # original left unfitted


def test_filter_errors(library):
    R, g, _ = library
    with pytest.raises(ValueError, match="No replicate library"):
        ReplicateGLSW().fit(R)
    with pytest.raises(ValueError, match="not both"):
        ReplicateGLSW(replicates=R, replicate_groups=g).fit(R, groups=g)
    with pytest.raises(ValueError, match="replicate_groups"):
        ReplicateEPO(replicates=R).fit(R)
    with pytest.raises(ValueError, match="groups"):
        ReplicateNoiseModel().fit(R, g)  # positional groups lands in y: must fail loudly
    with pytest.raises(ValueError, match="channels"):
        ReplicateGLSW().fit(R, groups=g).transform(R[:, :10])


def test_replicate_noise_in_compose(library, calib):
    R, g, _ = library
    X, _, _ = calib
    pipe = sp.Compose([ReplicateNoise(R, g), sp.GaussianNoise(std=1e-4)])
    out = pipe(X, rng=0)
    assert out.shape == X.shape and not np.allclose(out, X)
    np.testing.assert_array_equal(sp.Compose(pipe.transforms, magnitude=0)(X, rng=0), X)


def test_replicate_noise_without_library_raises(calib):
    X, _, _ = calib
    with pytest.raises(RuntimeError, match="no library"):
        ReplicateNoise()(X, rng=0)
