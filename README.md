# specperturb

Composable, reproducible perturbations for NIR, Raman and UV-Vis spectra.

Use it to augment training sets, or to stress-test a calibration by injecting
controlled, physically motivated distortions (scatter, baseline drift, noise,
wavelength miscalibration, interferents) at a chosen severity.

```python
import numpy as np
import specperturb as sp

aug = sp.Compose([
    sp.EMSCScatter(order=2),
    sp.RandomApply(sp.SineBaseline(), p=0.3),
    sp.AxisShift(shift=(-1.5, 1.5)),       # in axis units
    sp.GaussianNoise(std=(0.001, 0.003)),
])

X_aug = aug(X, x=wavelengths, rng=0)       # X: (n_samples, n_channels)
```

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[dev]"     # plus pytest and matplotlib
```

Depends on numpy, scipy and scikit-learn (>= 1.6).

## What's in it

| Module | Perturbations |
|---|---|
| `noise` | `GaussianNoise`, `HeteroscedasticNoise`, `PoissonNoise`, `MultiplicativeNoise`, `ColoredNoise`, `CosmicSpikes`, `Quantization`, `DeadPixels` |
| `baseline` | `ConstantOffset`, `LinearBaseline`, `PolynomialBaseline`, `ExponentialBaseline`, `SineBaseline`, `SplineBaseline`, `GaussianHump` |
| `scatter` | `MSCScatter`, `EMSCScatter`, `PathLength`, `PowerLawScatter` |
| `peaks` | `AddPeaks`, `IntensityScale`, `Broadening` |
| `axis` | `AxisShift`, `AxisStretch`, `AxisWarp` |
| `masking` | `AddInterferent`, `BandMask` |
| `instrument` | `StrayLight`, `LampIntensity`, `Temperature` |
| `replicate` | `ReplicateNoise` (real repeat-scan differences), plus `ReplicateGLSW`, `ReplicateEPO` filters and repeatability diagnostics; see below |

Combinators: `Compose`, `OneOf`, `SomeOf`, `RandomApply`.

`sp.available()` lists everything registered; `sp.get("PolynomialBaseline", order=4)`
builds one by name (handy for config files).

## scikit-learn

A Pipeline runs every step at predict time and cannot add rows, so a perturbation
used as a plain Pipeline step would also perturb your test spectra. Use the
wrapper that matches the job. (`SavGol` below stands for any Savitzky-Golay
transformer, e.g. chemotools' `SavitzkyGolay` or a `FunctionTransformer` around
`scipy.signal.savgol_filter`.)

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# 1. Training-time augmentation: copies are added inside fit only, after the CV
#    split, so they never leak across folds. predict() is untouched.
model = sp.AugmentedEstimator(
    make_pipeline(SavGol(deriv=1), PLSRegression(8)),   # preprocessing + model
    augmenter=sp.Compose([sp.EMSCScatter(), sp.GaussianNoise(std=2e-4)]),
    n_copies=10, x=wavelengths, random_state=0,
)
GridSearchCV(model, {"n_copies": [5, 10],
                     "augmenter__magnitude": [0.5, 1.0],
                     "estimator__plsregression__n_components": [6, 8, 10]})

# 2. Perturb at transform time, e.g. to stress-test a fitted model
shift = sp.PerturbationTransformer(sp.AxisShift(shift=(-1, 1)), x=wavelengths, random_state=0)
y_shifted = fitted_model.predict(shift.fit_transform(X_test))
```

Every perturbation supports `get_params(deep=True)`, `set_params` and
`sklearn.base.clone`. `PerturbationTransformer` passes scikit-learn's
`check_estimator` (it is tagged non-deterministic, so sklearn skips the three
checks that require identical output on repeated calls).

## Replicate-learned perturbation

Scan ~60-100 samples 3-6 times each (repack / re-present / rescan; no reference
values needed). Differences between scans of one sample contain no chemistry,
so they are a direct sample of measurement nuisance. One library, four uses:

```python
from specperturb.replicate import (ReplicateNoise, ReplicateGLSW, ReplicateEPO,
                                   ReplicateNoiseModel, prediction_repeatability,
                                   scatter_share)

# 0. Diagnostic first: share of prediction error caused by replicate scatter
#    (replicate scans of TEST samples). ~0.45 -> large gains seen; <= 0.03 -> none.
share = scatter_share(baseline_model.predict, X_test_reps, test_ids, y_test_reps)

# 1. Augmentation (raw spectra in, preprocessing inside the estimator)
model = sp.AugmentedEstimator(make_pipeline(SavGol(deriv=1), PLSRegression(8)),
                              augmenter=ReplicateNoise(X_rep, rep_ids), n_copies=15)

# 2. Filters that remove only replicate directions. The library is held by the
#    step, and `preprocessor` makes sure it is filtered in the same representation
#    as the step's input (preprocessor is fitted on calibration spectra).
model = make_pipeline(
    ReplicateGLSW(alpha=1e-4, replicates=X_rep, replicate_groups=rep_ids,
                  preprocessor=SavGol(deriv=1)),
    PLSRegression(8),
)

# 3. Per-sample repeatability of any fitted pipeline, from one scan per sample
noise = ReplicateNoiseModel().fit(X_rep, groups=rep_ids)
sd = prediction_repeatability(model.predict, X_test, noise, method="analytic")
sd_single = sd / 2 ** 0.5   # sd of one measurement rather than of a difference
```

Rules that matter for correctness:

1. Augment raw spectra, then preprocess (`AugmentedEstimator` does this).
2. Fit GLSW / EPO on replicates in the same representation as their input (`preprocessor`).
3. Cross-validate group-aware after augmentation (automatic inside `AugmentedEstimator`;
   with `ReplicateAugmenter.augment`, use the returned group IDs).
4. Keep replicate-library samples disjoint from test samples when validating. If
   the calibration set itself holds the replicates, build the library per training
   fold: `cross_validate(AugmentedEstimator(..., augmenter=ReplicateNoise()), X, y,
   groups=ids, cv=GroupKFold(5), params={"groups": ids})`.
5. `prediction_repeatability` returns the sd of a difference of two measurements;
   divide by sqrt(2) for single-measurement repeatability.
6. MSC / EMSC references come from calibration spectra only.

The operations relate to known methods: WinISI repeatability-file augmentation,
EPO (Roger et al. 2003), GLSW (Martens et al. 2003), PLS error propagation (Faber &
Kowalski 1997) and measurement-error covariance (Wentzell et al. 1997). Generated
spectra are useful and calibrated, not indistinguishable from real ones.

## Drift simulation and monitoring benchmark

`specperturb.drift` simulates drift over time with exact ground truth, so any
monitor can be scored on *when it alarms relative to when predictions become
harmful*. Monitoring parts reuse existing libraries:
`pip install 'specperturb[monitor]'` adds chemotools, menelaus, river and MAPIE
(Python >= 3.11; the rest of the package runs on 3.9+).

```python
from specperturb.drift import DriftScenario, model_truth, harm_onset
from specperturb.drift import mechanisms as M, profiles as T
from specperturb.drift.benchmark import MonitoringBenchmark

lamp_aging = DriftScenario([
    (M.Gain(), T.Saturating(onset=150, size=-0.25, tau=200)),     # -25 % intensity
    (M.NoiseIncrease(), T.Ramp(onset=150, rate=5e-6)),
])
res = lamp_aging.simulate(X_stream, x=wavelengths, rng=0)        # X, X_clean, theta, onset
truth = model_truth(pls_pipeline, res.X_clean, res.X, res.X_systematic)
harm = harm_onset(truth.bias_systematic, threshold=0.03)          # in y units

bench = MonitoringBenchmark(pls_pipeline, harm_threshold=0.03,
                            criterion="fap", window=150, fap=0.05).fit(X_cal, X_incontrol)
result = bench.run({"lamp aging": lamp_aging, "no drift": None}, X_pool, n_steps=600, n_runs=50)
result.summary()   # false alarms, delay, warned-before-harm, lead, nuisance alarms
```

**1. Mechanisms** (one physical parameter θ each, θ = 0 is the identity):
`Offset`, `Slope`, `Curvature`, `Gain` (absorbance with a stale reference, or
intensity), `WavelengthShift`, `WavelengthStretch`, `Bandwidth`,
`NoiseIncrease`, `StrayLightDrift`, `TemperatureDrift` (shift + broadening,
optionally limited to a region such as the OH bands; coefficients required),
`Fouling` (film spectrum), `Interferent` (pure spectrum), `ScatterChange`.
Scenarios apply them in physical order (sample → interface → optics → stray
light → source → detector), which matters for the non-linear ones.

**2. Profiles** θ(t): `Step`, `Ramp`, `Saturating`, `Periodic`, `RandomWalk`,
`Intermittent`, `AR1`; compose with `+` and `*`. Each knows its onset;
`AR1` and an always-on `Periodic` count as in-control variation.

**3. Empirical surrogates**, flagged `surrogate=True` with a `source` label:
`PairedInstrumentDrift` (chemotools DS / PDS, or mean difference, between
paired spectra on two instruments or two dates) and `ReplicateDrift` (a
replicate-library direction).

**4. Ground truth** for a PLS model or a Pipeline ending in PLS: true bias, the
split of the drift into its in-model part (what T² sees) and residual part (what
Q / DModX sees), and harm onset. The residual part cannot change a PLS
prediction (P'W* = I), so all bias comes from in-model drift.

**Harness**: statistics from chemotools (`HotellingT2`, `QResiduals`, `DModX`);
detectors Shewhart and EWMA (here), CUSUM (menelaus), Page-Hinkley / ADWIN /
KSWIN (river) and conformal test martingales (MAPIE). Every detector is
calibrated to the same in-control behaviour before comparison:

* `criterion="arl"` matches the in-control average run length;
  `criterion="fap"` matches the probability of a false alarm before onset.
  Use `"fap"` when martingales are included: under no drift many martingale runs
  never alarm, so their ARL depends on the simulation horizon.
* Calibrate on in-control spectra that were not used to build the model:
  statistics on training spectra are optimistic.
* Use thousands of in-control spectra. With a few hundred, the tail of T² is
  under-sampled and the realized false-alarm rate can be several times the
  target; `verify_calibration()` measures it on separate data. Shewhart limits
  are the most sensitive.
* In-control variation (temperature cycles, repacking) must be in the
  calibration data, or pass it as `in_control=` so it is added both to the
  calibration data and to the streams.
* Q and DModX are monotone in each other, so after calibration they raise
  identical alarms.
* ADWIN is conservative: on standardized stationary input its in-control ARL is
  about 1000 even at its loosest setting, so it cannot be matched to short targets.

`examples/drift_benchmark.py` runs four scenarios and draws
`examples/drift_timeline.png`.

## Conventions

Every perturbation is called as `p(X, x=axis, rng=seed_or_generator)`.

- **`X`** is `(n_samples, n_channels)` or a single 1-D spectrum. Input is never modified.
- **`x`** is the physical axis (nm or cm-1). Defaults to channel index. Shapes
  (polynomials, exponentials, fringes) are defined on the axis normalised to
  [-1, 1] or [0, 1], so coefficients mean the same thing whatever the units.
  Shift and warp amplitudes are in axis units.
- **Ranges.** Parameters are `(low, high)` ranges sampled per spectrum, or a
  scalar for a fixed value.
- **`magnitude`** in [0, 1] scales severity. `magnitude=0` is always the exact
  identity; for gains, it shrinks the range toward 1. `Compose(..., magnitude=m)`
  scales a whole pipeline, which is what you want for robustness sweeps.
- **`p`** is the per-spectrum probability of applying an effect.
  `RandomApply(t, p)` instead applies or skips the whole batch.
- **Reproducibility.** One seed passed to `Compose` reproduces the whole chain.

### Domain matters

Additive effects (baselines, offsets, peaks) are natural in absorbance.
Multiplicative effects (`MSCScatter`, `EMSCScatter`, `PathLength`,
`PowerLawScatter`) model scatter as a gain, which is how MSC/EMSC assume it
behaves in log(1/R). `PoissonNoise` needs non-negative intensity data (counts,
R, T), not absorbance.

## Adding a perturbation

Subclass `Perturbation`, implement `_apply`, decorate with `@register`:

```python
from specperturb.base import Perturbation, register, sample

@register
class WaterVapourLines(Perturbation):
    def __init__(self, strength=(0.0, 0.02), **kw):
        super().__init__(**kw)
        self.strength = strength

    def _apply(self, X, x, rng):
        # X: float copy, (n, m); x: 1-D axis; rng: numpy Generator
        s = sample(rng, self.strength, (X.shape[0], 1), self.magnitude)
        return X + s * my_line_shape(x)
```

If the perturbation learns from data, add a `fit(X, groups=None)` method (see
`ReplicateNoise`); `AugmentedEstimator` calls it on each training fold.

Put it in the matching module under `src/specperturb/perturbations/` (or a new
module imported in that package's `__init__.py`) and add its name to the
module's `__all__`. The contract tests in `tests/test_contract.py` then check it
automatically for shape, reproducibility, no input mutation, `magnitude=0`
identity, and `p=0` identity.

Use `sample(...)` for severity parameters (scaled by magnitude),
`sample_gain(...)` for multiplicative factors around 1, and `sample(..., magnitude=1)`
style calls (omit magnitude) for shape parameters like frequencies or exponents.

## Tests

```bash
pytest
```

## License

MIT
