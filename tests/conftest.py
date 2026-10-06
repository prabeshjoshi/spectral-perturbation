"""Shared synthetic replicate data with a KNOWN nuisance model.

Each scan = chemistry (2 analytes, Gaussian bands) + nuisance, where nuisance is
low-rank (offset, slope, broad hump: 3 directions with large scores) plus small
independent noise on every channel. Scans of one sample share the chemistry and
differ only in nuisance.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

P = 80
AXIS = np.linspace(1000, 2500, P)  # nm
_u = (AXIS - AXIS.min()) / np.ptp(AXIS)
ANALYTES = np.vstack([
    np.exp(-0.5 * ((AXIS - 1450) / 40) ** 2),
    np.exp(-0.5 * ((AXIS - 1940) / 60) ** 2),
])
NUISANCE = np.vstack([np.ones(P), _u - 0.5, np.exp(-0.5 * ((_u - 0.6) / 0.25) ** 2)])
NUISANCE /= np.linalg.norm(NUISANCE, axis=1, keepdims=True)
NUISANCE_SCALE = 0.3
DIAG_SD = 1e-3


def make_scans(n_samples, n_scans, rng, outlier=False):
    conc = rng.uniform(0.2, 1.0, (n_samples, 2))
    rows, groups, y = [], [], []
    for i in range(n_samples):
        for _ in range(n_scans):
            nuis = rng.normal(0, NUISANCE_SCALE, 3) @ NUISANCE
            rows.append(conc[i] @ ANALYTES + nuis + rng.normal(0, DIAG_SD, P))
            groups.append(i)
            y.append(conc[i, 0])
    X = np.asarray(rows)
    if outlier:
        X[1] += 50.0  # one gross bad scan
    return X, np.asarray(groups), np.asarray(y)


@pytest.fixture(scope="session")
def ref():
    """The original replicate_perturbation.py, unmodified, as a numerical reference."""
    path = Path(__file__).with_name("_reference_replicate.py")
    spec = importlib.util.spec_from_file_location("_reference_replicate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def library():
    return make_scans(30, 4, np.random.default_rng(0))


@pytest.fixture(scope="session")
def calib():
    return make_scans(20, 1, np.random.default_rng(1))


@pytest.fixture(scope="session")
def test_set():
    return make_scans(60, 1, np.random.default_rng(2))
