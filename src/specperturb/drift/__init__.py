"""Time-resolved drift simulation with ground truth, for evaluating monitors.

Four parts:

1. :mod:`~specperturb.drift.mechanisms` - one physical parameter each
   (offset, slope, curvature, gain, wavelength shift / stretch, bandwidth,
   noise, stray light, temperature, fouling, interferent, scatter).
2. :mod:`~specperturb.drift.profiles` - theta(t): Step, Ramp, Saturating,
   Periodic, RandomWalk, Intermittent, AR1; composable with + and *.
3. :mod:`~specperturb.drift.empirical` - surrogates learned from data:
   paired-instrument differences (chemotools DS/PDS) and replicate directions.
   Flagged ``surrogate=True`` in every result.
4. :mod:`~specperturb.drift.truth` - onset, drift size, in-model vs residual
   split for a PLS model, true prediction bias and harm onset.

Plus :mod:`~specperturb.drift.monitor` (T^2 / Q / DModX from chemotools;
detectors from menelaus, river, MAPIE) and
:mod:`~specperturb.drift.benchmark` (matched-ARL comparison, alarm vs harm).
Monitoring needs ``pip install 'specperturb[monitor]'``.
"""
from . import mechanisms, profiles
from .mechanisms import *  # noqa: F401,F403
from .profiles import *  # noqa: F401,F403
from .empirical import PairedInstrumentDrift, ReplicateDrift
from .scenario import DriftResult, DriftScenario
from .truth import ModelTruth, harm_onset, model_truth

__all__ = (mechanisms.__all__ + profiles.__all__
           + ["PairedInstrumentDrift", "ReplicateDrift", "DriftScenario", "DriftResult",
              "ModelTruth", "model_truth", "harm_onset"])
