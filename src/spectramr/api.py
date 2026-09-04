"""Public scripting API for spectraMR.

The single documented import root for **script-driven** use of the framework —
the fourth execution mode alongside config-driven ``train``, campaigns, and the
container/cluster backends. A user who has ``pip install``-ed the framework can:

* register custom components defined *outside* the source tree::

      from spectramr.api import register_model, register_loss

      @register_model("my_unet", "reconstruction")
      class MyUNet(nn.Module): ...

* build framework components from a config (in-memory or YAML), bypassing the
  CLI::

      from spectramr.api import settings_from_dict, make_model, make_optimizer

      cfg = settings_from_dict({"model": {"model_type": "my_unet"}, "data": {},
                                "optimization": {}, "logging": {}})
      model, _ = make_model(config=cfg, device="cpu")
      opt, _ = make_optimizer(config=cfg, model=model)

* (WS-A) run a training loop directly with hand-built objects via ``fit`` /
  ``Trainer`` — reusing the *same* loop engine the config-driven pipeline uses.

This module only re-exports objects whose canonical home is elsewhere, so
``spectramr.api.X`` and ``spectramr.X`` resolve to the identical object.

Resolution is LAZY (PEP 562), sharing ``spectramr.__init__``'s ``_LAZY_EXPORTS``
table so the two surfaces cannot drift apart.

Why this facade cannot be eager
-------------------------------
It used to eagerly ``from spectramr.pipelines.fit import Trainer, fit``, which is
fatal for the out-of-tree plugin path this module documents at the top. Plugin
discovery runs at module-import time (``core/metrics/__init__``,
``models/losses/__init__``), so a plugin doing the documented
``from spectramr.api import register_model`` imported this module while
``spectramr`` was still initialising, and the eager chain
``api -> pipelines.fit -> pipelines.train -> bootstrap -> ...`` hit a
half-built module::

    ImportError: cannot import name 'MetricsTracker' from partially initialized
    module 'spectramr.infrastructure.services.metrics_tracker'

The plugin body was abandoned mid-way, nothing registered, and nothing was
raised. Resolving per name means ``register_model`` now costs exactly
``spectramr.models.registry`` — no cycle, and no reason for a plugin to prefer
one import spelling over another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spectramr import _LAZY_EXPORTS as _LAZY_EXPORTS  # SSOT: one table, two surfaces
from spectramr import __getattr__ as _root_getattr

if TYPE_CHECKING:
    # Static-only imports so type-checkers see the lazily-exported names.
    from spectramr.config.settings import (
        TrainingSettings as TrainingSettings,
    )
    from spectramr.config.settings import (
        settings_from_dict as settings_from_dict,
    )
    from spectramr.core.metrics.registry import register_metric as register_metric
    from spectramr.models.losses.registry import register_loss as register_loss
    from spectramr.models.registry import register_model as register_model
    from spectramr.pipelines.fit import (
        Trainer as Trainer,
    )
    from spectramr.pipelines.fit import (
        fit as fit,
    )
    from spectramr.pipelines.make import (
        make_dataloader as make_dataloader,
    )
    from spectramr.pipelines.make import (
        make_dataset as make_dataset,
    )
    from spectramr.pipelines.make import (
        make_model as make_model,
    )
    from spectramr.pipelines.make import (
        make_optimizer as make_optimizer,
    )

__all__ = [
    "Trainer",
    "TrainingSettings",
    "fit",
    "make_dataloader",
    "make_dataset",
    "make_model",
    "make_optimizer",
    "register_loss",
    "register_metric",
    "register_model",
    "settings_from_dict",
]


def __getattr__(name: str) -> Any:
    """Resolve a public export lazily (PEP 562).

    Delegates to ``spectramr.__getattr__`` so both surfaces resolve through one
    table to one object -- the identity this module's docstring promises. The
    ``AttributeError`` that raises names ``spectramr``; raise against this module
    instead so the message points at the import the caller actually wrote.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'spectramr.api' has no attribute {name!r}")
    value = _root_getattr(name)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
