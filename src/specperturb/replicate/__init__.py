"""Replicate-learned perturbation: one library of repeat scans, four uses.

1. Augmentation        :class:`ReplicateAugmenter`, :class:`ReplicateNoise`
2. Preprocessing       :class:`ReplicateGLSW`, :class:`ReplicateEPO`
3. Repeatability       :func:`prediction_repeatability`
4. Diagnostic          :func:`scatter_share`

Scan ~60-100 samples 3-6 times each on one instrument (repack / re-present /
rescan; no reference values needed). The differences between scans of the same
sample are a direct sample of measurement nuisance.

Rules that matter for correctness
---------------------------------
1. Augment raw spectra, then preprocess the augmented set (and test spectra the
   same way).
2. Fit GLSW / EPO on replicates in the same representation as their input (use the
   ``preprocessor`` argument when they follow a preprocessing step).
3. Cross-validate group-aware after augmentation; copies of a sample stay in one
   fold (automatic with :class:`specperturb.AugmentedEstimator`).
4. Replicate-library samples must be disjoint from test samples when validating.
5. Repeatability from ``covariance_`` is the spread of a difference of two
   measurements; divide by sqrt(2) for single-measurement repeatability.
6. MSC / EMSC references elsewhere in a pipeline come from calibration spectra only.

Related methods: WinISI repeatability-file augmentation; EPO (Roger et al. 2003);
GLSW (Martens et al. 2003); PLS error propagation (Faber & Kowalski 1997);
measurement-error covariance (Wentzell et al. 1997). What this module adds is the
replicate-library framing, its validation and the scatter-share diagnostic, not a
new algorithm.
"""
from .core import ReplicateNoiseModel, replicate_differences
from .augment import ReplicateAugmenter, ReplicateNoise
from .filters import ReplicateEPO, ReplicateGLSW
from .diagnostics import prediction_repeatability, scatter_share

__all__ = ["replicate_differences", "ReplicateNoiseModel", "ReplicateAugmenter",
           "ReplicateNoise", "ReplicateGLSW", "ReplicateEPO",
           "prediction_repeatability", "scatter_share"]
