"""``opt_d`` follows the discriminator, not the paradigm name.

``ModelBuilder.build_discriminator`` is keyed on CONFIGURATION -- it builds
whenever ``model.discriminator_component`` is present, whatever the
``training_mode``. ``OptimizationBuilder.build_optimizers`` was keyed on the
paradigm NAME, via a hardcoded allowlist. A ``training_mode: diffusion`` arm
that configured a critic therefore got the model and no optimizer:
``env.opt_d`` is ``optimizers.get("opt_d")``, so it answered ``None`` and the
run died in ``StepExecutor`` naming ``build_optimizers`` rather than the
allowlist that omitted the paradigm (#1670).

``fit()`` never had the gate, so the same config worked programmatically and
failed under ``spectramr train`` -- two owners for one question (NN17).

One case per SHAPE the rule takes (NN15), not just the easy one:
  * non-GAN paradigm WITH a discriminator  -> opt_d must now be built
  * GAN paradigm WITHOUT a discriminator   -> must still raise
  * non-GAN paradigm WITHOUT one           -> must stay untouched (additive)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from spectramr.infrastructure.training.builders.optimization_builder import (  # noqa: E402
    OptimizationBuilder,
)


class _Optimizer:
    """Minimal stand-in for ``optimization.optimizer``."""

    def __init__(self) -> None:
        self.type = "adamw"
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.beta1, self.beta2 = 0.9, 0.999
        self.betas = None
        self.momentum = 0.9
        self.eps = 1e-8
        self.nesterov = None
        self.amsgrad = None
        self.kwargs: dict = {}
        self.param_groups = None
        self.lookahead = None
        self.generator_learning_rate = None
        self.discriminator_learning_rate = None
        self.model_fields_set: set[str] = set()


class _Opt:
    def __init__(self) -> None:
        self.optimizer = _Optimizer()
        self.scheduler = None
        self.mixed_precision = False


class _Training:
    """``build_optimizers`` reads the paradigm off ``config.training``.

    Not off the top level: the top-level branch is only consulted when
    ``config.training.training_mode`` is falsy, and it never is here.
    """

    gan = None
    max_iterations = 100

    def __init__(self, training_mode: str) -> None:
        self.training_mode = training_mode


class _Cfg:
    def __init__(self, training_mode: str) -> None:
        self.optimization = _Opt()
        self.training = _Training(training_mode)


def _models(with_disc: bool) -> dict[str, nn.Module]:
    m = {"generator": nn.Linear(2, 2)}
    if with_disc:
        m["discriminator"] = nn.Linear(2, 2)
    return m


def _build(training_mode: str, with_disc: bool):
    b = OptimizationBuilder(_Cfg(training_mode), models=_models(with_disc))
    return b.build_optimizers().build()[0]


def test_non_gan_paradigm_with_a_discriminator_gets_opt_d():
    """The shape that was broken: a critic present, an optimizer absent."""
    opts = _build("diffusion", with_disc=True)
    assert "opt_d" in opts, (
        "a configured discriminator got no optimizer -- env.opt_d would answer "
        "None and the run would die in StepExecutor (#1670)"
    )
    assert "opt_g" in opts


def test_non_gan_paradigm_without_a_discriminator_is_unchanged():
    """Additive: an arm with no critic must not gain an optimizer."""
    opts = _build("diffusion", with_disc=False)
    assert "opt_d" not in opts
    assert "opt_g" in opts


def test_gan_with_a_discriminator_still_gets_both():
    """The GAN path is untouched by the generalisation."""
    opts = _build("gan", with_disc=True)
    assert {"opt_g", "opt_d"} <= set(opts)


def test_gan_without_a_discriminator_still_raises():
    """The REQUIREMENT is unchanged -- a GAN without a critic is not a GAN."""
    with pytest.raises(Exception, match="[Dd]iscriminator"):
        _build("gan", with_disc=False)
