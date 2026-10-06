"""Behavioural checks that each effect does what its docstring says."""
import numpy as np

import specperturb as sp

M = 400
WN = np.linspace(4000, 10000, M)
X0 = 0.5 * np.exp(-0.5 * ((WN - 6500) / 200) ** 2) + 0.2
X = np.tile(X0, (8, 1))


def _fit_msc(ref, y):
    b, a = np.polyfit(ref, y, 1)
    return b, a


def test_msc_forward_is_exactly_affine():
    out = sp.MSCScatter(gain=(0.8, 1.2), offset=(-0.1, 0.1))(X, x=WN, rng=0)
    for row in out:
        b, a = _fit_msc(X0, row)
        np.testing.assert_allclose(b * X0 + a, row, atol=1e-10)


def test_emsc_residual_is_polynomial():
    order = 2
    out = sp.EMSCScatter(order=order)(X, x=WN, rng=0)
    w = 2 * (WN - WN.min()) / np.ptp(WN) - 1
    D = np.column_stack([X0, np.vander(w, order + 1, increasing=True)])
    for row in out:
        coef, *_ = np.linalg.lstsq(D, row, rcond=None)
        np.testing.assert_allclose(D @ coef, row, atol=1e-10)


def test_exponential_baseline_bounded_by_amplitude():
    out = sp.ExponentialBaseline(amplitude=0.3, rate=(1, 8))(X, x=WN, rng=0)
    bg = out - X
    assert bg.min() >= -1e-12
    np.testing.assert_allclose(bg.max(axis=1), 0.3, rtol=1e-9)


def test_axis_shift_has_no_edge_spikes():
    out = sp.AxisShift(shift=(50, 50))(X, x=WN, rng=0)
    assert out.max() <= X.max() + 1e-12
    assert out.min() >= X.min() - 1e-12


def test_axis_shift_moves_peak():
    out = sp.AxisShift(shift=(30, 30))(X0, x=WN, rng=0)
    assert WN[np.argmax(out)] > WN[np.argmax(X0)]


def test_poisson_noise_scales_with_counts():
    Xi = np.tile(X0 * 1000, (200, 1))
    lo = sp.PoissonNoise(counts=1e3)(Xi, x=WN, rng=0) - Xi
    hi = sp.PoissonNoise(counts=1e5)(Xi, x=WN, rng=0) - Xi
    assert lo.std() > 5 * hi.std()


def test_gaussian_noise_std():
    out = sp.GaussianNoise(std=0.01)(np.zeros((500, M)), rng=0)
    assert abs(out.std() - 0.01) < 5e-4


def test_compose_magnitude_zero_is_identity():
    pipe = sp.Compose([sp.MSCScatter(), sp.PolynomialBaseline(), sp.GaussianNoise()], magnitude=0.0)
    np.testing.assert_allclose(pipe(X, x=WN, rng=0), X, atol=1e-12)


def test_compose_reproducible():
    pipe = sp.Compose([sp.SplineBaseline(), sp.AddPeaks(), sp.GaussianNoise()])
    np.testing.assert_array_equal(pipe(X, x=WN, rng=3), pipe(X, x=WN, rng=3))


def test_oneof_and_someof_run():
    a = sp.OneOf([sp.ConstantOffset(), sp.GaussianNoise()])(X, x=WN, rng=0)
    b = sp.SomeOf([sp.ConstantOffset(), sp.GaussianNoise(), sp.PathLength()], n=(1, 2))(X, x=WN, rng=0)
    assert a.shape == b.shape == X.shape


def test_custom_perturbation_registers():
    from specperturb.base import Perturbation, register, sample, _REGISTRY

    @register
    class _Tmp(Perturbation):
        def __init__(self, v=(0.1, 0.1), **kw):
            super().__init__(**kw)
            self.v = v

        def _apply(self, X, x, rng):
            return X + sample(rng, self.v, (X.shape[0], 1), self.magnitude)

    try:
        out = sp.get("_Tmp")(X, x=WN, rng=0)
        np.testing.assert_allclose(out, X + 0.1)
    finally:
        _REGISTRY.pop("_Tmp")
