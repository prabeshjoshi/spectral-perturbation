"""Contract tests run against EVERY registered perturbation. A new class you
add with @register is picked up automatically; if it needs constructor
arguments, add them to REQUIRED_ARGS."""
import numpy as np
import pytest

import specperturb as sp
from specperturb.base import _REGISTRY

M = 300
WN = np.linspace(4000, 10000, M)  # cm-1
X0 = (0.5 * np.exp(-0.5 * ((WN - 5200) / 150) ** 2)
      + 0.3 * np.exp(-0.5 * ((WN - 8300) / 250) ** 2) + 0.1)
X = np.tile(X0, (5, 1))

REQUIRED_ARGS = {
    "AddInterferent": dict(interferents=np.exp(-0.5 * ((WN - 7000) / 80) ** 2)),
    "PowerLawScatter": dict(axis_unit="cm-1"),
    "Temperature": dict(shift_per_degree=-3.0, broadening_per_degree=2.0),
    "ReplicateNoise": dict(
        replicates=np.vstack([X0 + np.random.default_rng(k).normal(0, 0.01, M) for k in range(6)]),
        groups=np.repeat([0, 1, 2], 2),
    ),
}
NON_DETERMINISTIC_IDENTITY = set()  # classes where magnitude=0 is not an exact identity

NAMES = sp.available()


def make(name, **kw):
    return sp.get(name, **{**REQUIRED_ARGS.get(name, {}), **kw})


@pytest.mark.parametrize("name", NAMES)
def test_shape_and_finite(name):
    out = make(name)(X, x=WN, rng=0)
    assert out.shape == X.shape
    if not (name == "DeadPixels"):
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", NAMES)
def test_1d_input_roundtrip(name):
    out = make(name)(X0, x=WN, rng=0)
    assert out.shape == X0.shape


@pytest.mark.parametrize("name", NAMES)
def test_reproducible(name):
    a = make(name)(X, x=WN, rng=42)
    b = make(name)(X, x=WN, rng=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_does_not_mutate_input(name):
    Xc = X.copy()
    make(name)(Xc, x=WN, rng=0)
    np.testing.assert_array_equal(Xc, X)


@pytest.mark.parametrize("name", NAMES)
def test_magnitude_zero_is_identity(name):
    if name in NON_DETERMINISTIC_IDENTITY:
        pytest.skip()
    out = make(name, magnitude=0.0)(X, x=WN, rng=0)
    np.testing.assert_allclose(out, X, atol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_p_zero_is_identity(name):
    out = make(name, p=0.0)(X, x=WN, rng=0)
    np.testing.assert_array_equal(out, X)


@pytest.mark.parametrize("name", NAMES)
def test_actually_changes_something(name):
    out = make(name)(X, x=WN, rng=1)
    assert not np.allclose(out, X), f"{name} had no effect at magnitude=1"


def test_registry_complete():
    for name in sp.perturbations.__all__:
        assert name in _REGISTRY, f"{name} exported but not registered"
