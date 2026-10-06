"""specperturb: composable, reproducible perturbations for NIR, Raman and UV-Vis spectra.

    import specperturb as sp
    aug = sp.Compose([
        sp.MSCScatter(),
        sp.PolynomialBaseline(order=2),
        sp.GaussianNoise(std=0.002),
    ])
    X_aug = aug(X, x=wavelengths, rng=0)

scikit-learn integration: :class:`AugmentedEstimator` (training-time augmentation)
and :class:`PerturbationTransformer` (perturb at transform time).
Replicate-learned perturbation and filters: :mod:`specperturb.replicate`.
"""
from .base import Perturbation, available, get, register
from .compose import Compose, OneOf, RandomApply, SomeOf
from .perturbations import *  # noqa: F401,F403
from .perturbations import __all__ as _perturbations_all
from . import replicate, drift
from .replicate import ReplicateNoise
from .estimators import AugmentedEstimator, PerturbationTransformer, augment_dataset

__version__ = "0.3.0"

__all__ = (["Perturbation", "register", "available", "get",
            "Compose", "OneOf", "SomeOf", "RandomApply",
            "AugmentedEstimator", "PerturbationTransformer", "augment_dataset",
            "replicate", "drift", "ReplicateNoise"]
           + list(_perturbations_all))
