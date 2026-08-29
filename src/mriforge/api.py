"""Public scripting API for MRIForge.

The single documented import root for **script-driven** use of the framework —
the fourth execution mode alongside config-driven ``train``, campaigns, and the
container/cluster backends. A user who has ``pip install``-ed the framework can:

* register custom components defined *outside* the source tree::

      from mriforge.api import register_model, register_loss

      @register_model("my_unet")
      class MyUNet(nn.Module): ...

* build framework components from a config (in-memory or YAML), bypassing the
  CLI::

      from mriforge.api import settings_from_dict, make_model, make_optimizer

      cfg = settings_from_dict({"model": {"model_type": "my_unet"}, "data": {},
                                "optimization": {}, "logging": {}})
      model, _ = make_model(config=cfg, device="cpu")
      opt, _ = make_optimizer(config=cfg, model=model)

* (WS-A) run a training loop directly with hand-built objects via ``fit`` /
  ``Trainer`` — reusing the *same* loop engine the config-driven pipeline uses.

This module is a thin, eager facade: it only re-exports objects whose canonical
home is elsewhere, so ``mriforge.api.X`` and the lazy ``mriforge.X`` resolve to
the identical object. It is imported on demand (never at package init), so its
eager imports do not slow down ``import mriforge``.
"""

from __future__ import annotations

from mriforge.config.settings import TrainingSettings, settings_from_dict
from mriforge.core.metrics.registry import register_metric
from mriforge.models.losses.registry import register_loss
from mriforge.models.registry import register_model
from mriforge.pipelines.fit import Trainer, fit
from mriforge.pipelines.make import (
    make_dataloader,
    make_dataset,
    make_model,
    make_optimizer,
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
