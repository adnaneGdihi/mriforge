"""Regression: ``optimization.optimizer.discriminator_learning_rate`` (TTUR) was dead.

2026-06-11 infrastructure audit. The schema field
``optimization.optimizer.discriminator_learning_rate`` was read by nobody, so the
discriminator optimizer was always built at the generator's learning rate even
when a two-time-scale update rule (TTUR) was configured. The builder now honours
it. Also: the configured ``eps`` reaches adaptive optimizers (previously the
leaf builder fell back to torch's default).
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders import OptimizationBuilder


class _Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)


def _gan_config():
    return TrainingSettings.from_yaml("experiments/inprogress/dummy/dummy_gan.yaml")


#: Phase 8 moved these onto ``optimization.optimizer``. Routed explicitly
#: because ``model_copy(update=...)`` does NOT validate: an update aimed at the
#: old level lands as an attribute nothing reads, so the test would assert the
#: unchanged default against itself and pass for the wrong reason.
_ON_THE_OPTIMIZER = {
    "type",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "betas",
    "eps",
    "momentum",
    "nesterov",
    "amsgrad",
    "kwargs",
    "param_groups",
    "lookahead",
    "generator_learning_rate",
    "discriminator_learning_rate",
}
_LEGACY_NAMES = {
    "optimizer": "type",
    "optimizer_type": "type",
    "optimizer_kwargs": "kwargs",
}


def _with_optimization(config, **updates):
    """Return a copy with ``optimization`` sub-fields overridden (frozen-safe)."""
    canonical = {_LEGACY_NAMES.get(k, k): v for k, v in updates.items()}
    on_optimizer = {k: v for k, v in canonical.items() if k in _ON_THE_OPTIMIZER}
    rest = {k: v for k, v in canonical.items() if k not in _ON_THE_OPTIMIZER}
    if on_optimizer:
        rest["optimizer"] = config.optimization.optimizer.model_copy(
            update=on_optimizer
        )
    new_opt = config.optimization.model_copy(update=rest)
    return config.model_copy(update={"optimization": new_opt})


def test_discriminator_learning_rate_is_applied_to_opt_d():
    base = _gan_config()
    gen_lr = base.optimization.optimizer.learning_rate
    disc_lr = gen_lr * 0.25  # distinct TTUR value
    config = _with_optimization(base, discriminator_learning_rate=disc_lr)

    models = {"generator": _Dummy(), "discriminator": _Dummy()}
    builder = OptimizationBuilder(config, models=models)
    builder.build_optimizers()

    opt_d = builder._optimizers["opt_d"]
    opt_g = builder._optimizers["opt_g"]
    assert opt_d.param_groups[0]["lr"] == disc_lr
    # Generator unaffected by the D-only override.
    assert opt_g.param_groups[0]["lr"] == gen_lr
    # And D must NOT silently inherit G's LR.
    assert opt_d.param_groups[0]["lr"] != gen_lr


def test_discriminator_lr_defaults_to_generator_lr_when_unset():
    base = _gan_config()
    config = _with_optimization(base, discriminator_learning_rate=None)
    models = {"generator": _Dummy(), "discriminator": _Dummy()}
    builder = OptimizationBuilder(config, models=models)
    builder.build_optimizers()
    opt_d = builder._optimizers["opt_d"]
    assert opt_d.param_groups[0]["lr"] == base.optimization.optimizer.learning_rate


def test_configured_eps_reaches_adaptive_optimizer():
    base = _gan_config()
    # Force an Adam-family optimizer with a non-default eps.
    config = _with_optimization(base, optimizer="adamw", eps=1e-4)
    models = {"generator": _Dummy(), "discriminator": _Dummy()}
    builder = OptimizationBuilder(config, models=models)
    builder.build_optimizers()
    opt_g = builder._optimizers["opt_g"]
    assert isinstance(opt_g, torch.optim.AdamW)
    assert opt_g.param_groups[0]["eps"] == 1e-4
