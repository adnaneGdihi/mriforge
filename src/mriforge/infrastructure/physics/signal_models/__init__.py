"""Signal models — nonlinear forward physics mapping parameters to a signal.

Two things live here:

* Sequence-specific acquisition models (``TSESignalModel``, ``flow_encoding``)
  that generate a measured signal from an image or a velocity field.
* Tracer-kinetic / relaxometry forward maps (``perfusion_kinetics``,
  ``diffusion_models``) that turn physical parameters into a measured curve.

Importing this package fires every ``@register_signal_model`` decorator, which is
what makes :func:`~.registry.list_signal_models` order-independent. See the
module docstring in :mod:`.registry` for why a signal model is deliberately NOT
an ``OperatorRegistry`` operator — the short version is that an operator's
contract promises ``adjoint()``, and these maps are nonlinear and have none.

Note ``flow_encoding`` is NOT registered here: phase contrast has a genuine
linear operator (``phase_contrast``, whose ``pc_adjoint`` is a true inverse
within ``|v| < venc``), so it belongs in the ``OperatorRegistry``. The split is
about whether an adjoint exists, not about which regime is fashionable.

The submodule imports below are load-bearing side effects, not re-exports: drop
one and its physics silently vanishes from the registry, and the maturity ledger
reports that regime as declaring a forward model it cannot resolve.
"""

from . import diffusion_models as diffusion_models  # side effect: register
from . import perfusion_kinetics as perfusion_kinetics  # side effect: register
from . import spectroscopy as spectroscopy  # side effect: register
from .registry import (
    SignalModelRegistry,
    SignalModelSpec,
    get_signal_model,
    list_signal_models,
    register_signal_model,
)
from .tse_b0_model import (
    TSESignalModel,
    shepp_logan_2d,
    simulate_halbach_b0,
)

__all__ = [
    "SignalModelRegistry",
    "SignalModelSpec",
    "TSESignalModel",
    "get_signal_model",
    "list_signal_models",
    "register_signal_model",
    "shepp_logan_2d",
    "simulate_halbach_b0",
]
