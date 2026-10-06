"""scikit-learn integration for every perturbation.

Why two wrappers
----------------
A scikit-learn Pipeline runs every step's ``transform`` at predict time as well as
at fit time, and it cannot change the number of rows. So a perturbation used as an
ordinary Pipeline step would also perturb your test spectra, and it could not add
augmented copies. Each wrapper solves one of the two jobs:

* :class:`AugmentedEstimator` - training-time augmentation. Adds perturbed copies
  inside ``fit`` only; ``predict`` passes spectra through untouched. Because the
  copies are made after the CV split, they can never leak across folds.
* :class:`PerturbationTransformer` - perturb every spectrum it sees. Use it to
  stress-test a fitted model (``model.predict(PerturbationTransformer(...)
  .fit_transform(X_test))``) or wherever perturbing at predict time is the point.

Both are proper estimators: ``get_params`` / ``set_params``, ``clone``, and grid
search over nested parameters (``augmenter__magnitude``, ``estimator__...``).
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted

from .base import Perturbation

__all__ = ["PerturbationTransformer", "AugmentedEstimator", "augment_dataset"]


def augment_dataset(perturbation, X, y, n_copies=10, x=None, groups=None, rng=None):
    """Originals plus ``n_copies`` perturbed copies, with labels and group IDs.

    Returns
    -------
    X_aug, y_aug, groups_aug
        Originals first, then each block of copies. ``groups_aug`` repeats
        ``groups`` (or the row index) so a group-aware splitter keeps every copy
        of a sample in the same fold.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, float)
    y = np.asarray(y)
    g = np.arange(len(X)) if groups is None else np.asarray(groups)
    blocks = [X] + [perturbation(X, x=x, rng=rng) for _ in range(n_copies)]
    reps = n_copies + 1
    y_aug = np.tile(y, (reps,) + (1,) * (y.ndim - 1)) if y.ndim > 1 else np.tile(y, reps)
    return np.vstack(blocks), y_aug, np.tile(g, reps)


class PerturbationTransformer(TransformerMixin, BaseEstimator):
    """Use any specperturb perturbation as a scikit-learn transformer.

    Parameters
    ----------
    perturbation : Perturbation
        Any perturbation or combinator (``Compose`` etc.).
    x : array (n_channels,), optional
        Physical axis (nm or cm-1). Defaults to channel index.
    random_state : int, Generator or None
        Seeds a generator at ``fit``; successive ``transform`` calls continue it.

    Notes
    -----
    ``transform`` perturbs every spectrum, including at predict time if this sits
    in a Pipeline. For training-only augmentation use :class:`AugmentedEstimator`.

    Examples
    --------
    >>> stress = PerturbationTransformer(sp.AxisShift(shift=(-1, 1)), x=wl, random_state=0)
    >>> rmsep_shifted = rmse(y_test, model.predict(stress.fit_transform(X_test)))
    """

    def __init__(self, perturbation=None, x=None, random_state=None):
        self.perturbation = perturbation
        self.x = x
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        if not isinstance(self.perturbation, Perturbation):
            raise TypeError("perturbation must be a specperturb Perturbation")
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X):
        check_is_fitted(self, "_rng")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} "
                             f"is expecting {self.n_features_in_} features as input.")
        return self.perturbation(X, x=self.x, rng=self._rng)

    def __sklearn_tags__(self):  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"non_deterministic": True}


class AugmentedEstimator(MetaEstimatorMixin, BaseEstimator):
    """Fit ``estimator`` on the data plus perturbed copies; predict without perturbing.

    Parameters
    ----------
    estimator : estimator
        Typically a Pipeline of preprocessing + model. Augmentation happens
        before it, on raw spectra (rule: augment raw, then preprocess).
    augmenter : Perturbation
        Any specperturb perturbation or combinator, including
        :class:`specperturb.replicate.ReplicateNoise`.
    n_copies : int, default=10
        Perturbed copies per training spectrum (originals are kept).
    x : array (n_channels,), optional
        Physical axis passed to the perturbation.
    random_state : int, Generator or None

    Attributes
    ----------
    estimator_ : fitted clone of ``estimator``
    augmenter_ : the perturbation used (fitted copy when ``groups`` was given)

    Notes
    -----
    Copies are generated inside ``fit``, after any CV split, so they never cross
    folds and no group-aware bookkeeping is needed for the copies.

    If ``fit`` receives ``groups`` and the augmenter can learn from data (has a
    ``fit`` method, like ``ReplicateNoise``), a copy of the augmenter is fitted on
    the training rows. Use this when your calibration set contains replicates and
    the library must come from the training fold only::

        cross_validate(model, X, y, groups=ids, cv=GroupKFold(5), params={"groups": ids})

    Examples
    --------
    >>> model = AugmentedEstimator(
    ...     make_pipeline(SavGol(deriv=1), PLSRegression(8)),
    ...     augmenter=ReplicateNoise(X_rep, rep_ids),
    ...     n_copies=15, random_state=0)
    >>> GridSearchCV(model, {"n_copies": [5, 15], "estimator__plsregression__n_components": [6, 8, 10]})
    """

    def __init__(self, estimator=None, augmenter=None, n_copies=10, x=None, random_state=None):
        self.estimator = estimator
        self.augmenter = augmenter
        self.n_copies = n_copies
        self.x = x
        self.random_state = random_state

    def fit(self, X, y, groups=None, **fit_params):
        X = check_array(X, dtype=float)
        y = np.asarray(y)
        if not isinstance(self.augmenter, Perturbation):
            raise TypeError("augmenter must be a specperturb Perturbation")
        aug = self.augmenter
        if groups is not None and hasattr(aug, "fit"):
            aug = copy.deepcopy(aug).fit(X, groups=groups)
        self.augmenter_ = aug
        Xa, ya, _ = augment_dataset(aug, X, y, n_copies=self.n_copies, x=self.x,
                                    rng=np.random.default_rng(self.random_state))
        self.estimator_ = clone(self.estimator).fit(Xa, ya, **fit_params)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def score(self, X, y, **kw):
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(X, y, **kw)
