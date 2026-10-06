import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate
from sklearn.pipeline import make_pipeline

import specperturb as sp
from specperturb.replicate import ReplicateNoise

from conftest import make_scans


class Recorder(RegressorMixin, BaseEstimator):
    """Ridge that remembers what it was trained on."""

    def fit(self, X, y):
        self.X_seen_, self.y_seen_ = X, y
        self.model_ = Ridge(1e-3).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# --------------------------------------------------------------------------- #
#  PerturbationTransformer
# --------------------------------------------------------------------------- #
def test_transformer_basic(calib):
    X, _, _ = calib
    t = sp.PerturbationTransformer(sp.GaussianNoise(std=0.01), random_state=0)
    a = t.fit_transform(X)
    b = clone(t).fit(X).transform(X)
    assert a.shape == X.shape and not np.allclose(a, X)
    np.testing.assert_array_equal(a, b)  # reseeded at fit -> reproducible
    assert not np.allclose(t.transform(X), a)  # stream continues across calls


def test_transformer_uses_axis(calib):
    X, _, _ = calib
    from conftest import AXIS
    t = sp.PerturbationTransformer(sp.AxisShift(shift=(5, 5)), x=AXIS, random_state=0).fit(X)
    np.testing.assert_allclose(t.transform(X), sp.AxisShift(shift=(5, 5))(X, x=AXIS, rng=0))


def test_transformer_nested_params():
    t = sp.PerturbationTransformer(sp.Compose([sp.GaussianNoise()]))
    assert "perturbation__magnitude" in t.get_params()
    t.set_params(perturbation__magnitude=0.5)
    assert t.perturbation.magnitude == 0.5


@pytest.mark.parametrize("name", sp.available())
def test_every_perturbation_clones(name):
    from test_contract import make
    p = make(name)
    c = clone(p)
    assert type(c) is type(p) and c is not p
    assert c.get_params(deep=False).keys() == p.get_params(deep=False).keys()


def test_combinators_clone():
    for comb in (sp.Compose([sp.GaussianNoise(), sp.MSCScatter()], magnitude=0.5),
                 sp.OneOf([sp.GaussianNoise(), sp.MSCScatter()], weights=[1, 3]),
                 sp.SomeOf([sp.GaussianNoise(), sp.MSCScatter()], n=1),
                 sp.RandomApply(sp.GaussianNoise(), p=0.3)):
        c = clone(comb)
        assert repr(c) == repr(comb)


def test_set_params_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid parameter"):
        sp.GaussianNoise().set_params(sigma=1)


def test_transformer_rejects_non_perturbation(calib):
    X, _, _ = calib
    with pytest.raises(TypeError):
        sp.PerturbationTransformer(perturbation=lambda X: X).fit(X)


# --------------------------------------------------------------------------- #
#  AugmentedEstimator
# --------------------------------------------------------------------------- #
def test_augments_in_fit_only(calib, test_set):
    X, _, y = calib
    Xt, _, _ = test_set
    m = sp.AugmentedEstimator(Recorder(), sp.GaussianNoise(std=0.01), n_copies=4, random_state=0).fit(X, y)
    assert m.estimator_.X_seen_.shape == (len(X) * 5, X.shape[1])
    np.testing.assert_array_equal(m.estimator_.X_seen_[:len(X)], X)  # originals kept, first
    np.testing.assert_array_equal(m.predict(Xt), m.estimator_.predict(Xt))  # no test perturbation


def test_augmented_reproducible_and_clonable(calib):
    X, _, y = calib
    m = sp.AugmentedEstimator(Recorder(), sp.MSCScatter(), n_copies=3, random_state=7)
    a = clone(m).fit(X, y).estimator_.X_seen_
    b = clone(m).fit(X, y).estimator_.X_seen_
    np.testing.assert_array_equal(a, b)
    assert not hasattr(m, "estimator_")


def test_augmented_grid_search_nested_params(calib):
    X, _, y = calib
    m = sp.AugmentedEstimator(make_pipeline(PLSRegression(2)), sp.GaussianNoise(std=0.005),
                              n_copies=2, random_state=0)
    grid = {"n_copies": [1, 3],
            "augmenter__magnitude": [0.0, 1.0],
            "estimator__plsregression__n_components": [1, 2]}
    gs = GridSearchCV(m, grid, cv=KFold(4)).fit(X, y)
    assert set(gs.best_params_) == set(grid)


def test_replicate_library_fitted_per_training_fold(library):
    # calibration set that itself contains replicates: library must come from train rows only
    X, g, y = library
    m = sp.AugmentedEstimator(Recorder(), ReplicateNoise(), n_copies=2, random_state=0)
    res = cross_validate(m, X, y, groups=g, cv=GroupKFold(3), params={"groups": g},
                         return_estimator=True)
    for est, test_idx in zip(res["estimator"], GroupKFold(3).split(X, y, g)):
        D_fold = est.augmenter_.noise_model_.differences_
        test_rows = X[test_idx[1]]
        # every library difference is built from training rows: none equals a test-row pair
        n_test_pairs = sum(len(np.where(g[test_idx[1]] == s)[0]) * (len(np.where(g[test_idx[1]] == s)[0]) - 1)
                           for s in np.unique(g[test_idx[1]]))
        assert len(D_fold) == len(sp.replicate.replicate_differences(X, g)) - n_test_pairs
        assert test_rows.shape[0] > 0


def test_augmented_rejects_bad_augmenter(calib):
    X, _, y = calib
    with pytest.raises(TypeError):
        sp.AugmentedEstimator(Recorder(), augmenter="noise").fit(X, y)


def test_passes_sklearn_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    est = sp.PerturbationTransformer(sp.Compose([sp.MSCScatter(), sp.GaussianNoise(std=0.01)]), random_state=0)
    res = check_estimator(est, on_fail=None)
    assert not [r["check_name"] for r in res if r["status"] == "failed"]
