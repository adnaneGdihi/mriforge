"""The N-critic-updates-then-one-generator-update cadence has one owner.

This assembly was written out twice -- ``AdversarialMixin.train_step_adversarial``
and ``GANTrainingStrategy.train_step`` -- through different accessors
(``self.state.opt_d`` vs ``self.env.opt_d``). Only the GAN copy runs, so a
divergence between them would surface as a wrong TRAINING SCHEDULE rather than an
error: "N discriminator updates per generator update" is the definition of the
paradigm, and nothing downstream re-checks it (non-negotiable 17).

``losses.gan.disc_updates`` is the knob that feeds it (``_resolve_disc_updates``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from spectramr.infrastructure.training.strategies.mixins.adversarial import (  # noqa: E402
    assemble_adversarial_step_configs,
)


def _assemble(num_d_updates: int):
    made: list[int] = []

    def d_factory():
        made.append(len(made))
        return lambda: torch.zeros(())

    return (
        assemble_adversarial_step_configs(
            num_d_updates=num_d_updates,
            d_closure_factory=d_factory,
            g_closure=lambda: torch.zeros(()),
            discriminator=nn.Identity(),
            generator=nn.Identity(),
            opt_d="OPT_D",
            opt_g="OPT_G",
        ),
        made,
    )


@pytest.mark.parametrize("n", [1, 2, 5])
def test_exactly_n_discriminator_steps_then_one_generator_step(n):
    configs, _ = _assemble(n)
    names = [c["name"] for c in configs]
    assert names == ["discriminator"] * n + ["generator"], names


@pytest.mark.parametrize("n", [1, 3])
def test_each_discriminator_step_gets_its_own_closure(n):
    """The factory is called per update, not once and reused.

    Each critic update needs a closure over its own fresh forward pass; reusing
    one would train D repeatedly against a single stale generator output.
    """
    configs, made = _assemble(n)
    assert len(made) == n
    d_closures = [c["closure"] for c in configs if c["name"] == "discriminator"]
    assert len({id(c) for c in d_closures}) == n


def test_optimizers_are_routed_to_the_right_side():
    """The G step must never be handed opt_d, or D would train on G's objective."""
    configs, _ = _assemble(2)
    for cfg in configs:
        expected = "OPT_D" if cfg["name"] == "discriminator" else "OPT_G"
        assert cfg["optimizer"] == expected, cfg


def test_generator_step_is_last():
    """D updates precede G so the generator faces an already-updated critic."""
    configs, _ = _assemble(3)
    assert configs[-1]["name"] == "generator"
    assert all(c["name"] == "discriminator" for c in configs[:-1])
