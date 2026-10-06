"""All perturbations, grouped by physical origin.

To add a new family, create a module here, decorate classes with
``@register``, and import it below. Nothing else needs to change.
"""
from .noise import *      # noqa: F401,F403
from .baseline import *   # noqa: F401,F403
from .scatter import *    # noqa: F401,F403
from .peaks import *      # noqa: F401,F403
from .axis import *       # noqa: F401,F403
from .masking import *    # noqa: F401,F403
from .instrument import *  # noqa: F401,F403

from . import noise, baseline, scatter, peaks, axis, masking, instrument

__all__ = (noise.__all__ + baseline.__all__ + scatter.__all__
           + [n for n in peaks.__all__ if n[0].isupper()]
           + axis.__all__ + masking.__all__
           + [n for n in instrument.__all__ if n[0].isupper()])
