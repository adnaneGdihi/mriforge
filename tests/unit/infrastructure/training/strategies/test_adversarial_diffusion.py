"""A discriminator on a diffusion arm must TRAIN, or not be accepted at all.

`DiffusionTrainingStrategy` advertised adversarial support -- its class docstring
says "**Adversarial Loss**: Optional GAN-style discriminator loss", and
`_compute_losses_impl` passes `discriminator=` to the loss computer -- while
providing no way to update the critic: it did not inherit `AdversarialMixin`,
defined no discriminator step, and had no `train_step` of its own. A critic
attached to a diffusion arm stayed at its initialisation and fed the generator a
meaningless signal (pitfall #16).

There were TWO gates, both keyed on the paradigm NAME, and fixing either alone
changes nothing observable:

* the strategy consulted a discriminator it never updated;
* ``fit()`` wired a discriminator only when ``paradigm == "gan"``, so
  ``fit(paradigm="diffusion", discriminator=d)`` accepted the argument and
  silently dropped it -- no ``opt_d``, no model in the env.

Only both together move a weight, which is why these tests assert on weights
rather than on configuration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

pytestmark = [pytest.mark.slow]


class _Pairs(Dataset):
    def __init__(self, n: int = 4, size: int = 64) -> None:
        self.x = torch.randn(n, 1, size, size)
        self.y = torch.randn(n, 1, size, size)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"input": self.x[i], "target": self.y[i]}


class _TimestepConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


def _adversarial_config():
    from spectramr.pipelines.fit import _PARADIGM_DEFAULTS

    return {
        "losses": {
            **_PARADIGM_DEFAULTS["gan"]["losses"],
            # The adversarial term is warm-up gated for 1000 iterations by
            # default; a short run has to step outside that to observe it.
            "reconstruction": {"warmup_iterations": 0},
        },
        "training": {
            "strategy_class": "diffusion",
            **_PARADIGM_DEFAULTS["diffusion"]["training"],
        },
    }


def test_discriminator_on_a_diffusion_arm_actually_trains():
    """THE regression: the critic's weights must move.

    Asserting that a discriminator is present, or that a loss was computed,
    would both have passed against the facade. Only the weights distinguish
    "trained" from "consulted".
    """
    from spectramr.pipelines.fit import fit

    gen, disc = _TimestepConv(), nn.Conv2d(1, 1, 3, padding=1)
    disc_before = disc.weight.detach().clone()
    gen_before = gen.conv.weight.detach().clone()

    result = fit(
        gen,
        DataLoader(_Pairs(), batch_size=2),
        paradigm="diffusion",
        discriminator=disc,
        opt_d=torch.optim.Adam(disc.parameters(), lr=1e-3),
        device="cpu",
        max_iterations=3,
        config=_adversarial_config(),
    )

    assert result.get("success") is True, result.get("error")
    assert not torch.equal(disc_before, disc.weight.detach()), (
        "the discriminator was built and consulted but never updated — the facade"
    )
    assert not torch.equal(gen_before, gen.conv.weight.detach())


def test_a_diffusion_arm_without_a_discriminator_is_unchanged():
    """ADDITIVE: no discriminator means the base generator-only step, as before.

    This is what makes the feature opt-in rather than a change of paradigm, and
    it is the property most at risk from a `train_step` override.
    """
    from spectramr.pipelines.fit import fit

    result = fit(
        _TimestepConv(),
        DataLoader(_Pairs(), batch_size=2),
        paradigm="diffusion",
        device="cpu",
        max_iterations=2,
    )
    assert result.get("success") is True, result.get("error")


def test_step_configs_are_n_critic_updates_then_one_generator():
    """The cadence comes from ``losses.gan.disc_updates``, shared with GAN."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    assert hasattr(DiffusionTrainingStrategy, "train_step")
    import inspect

    source = inspect.getsource(DiffusionTrainingStrategy.train_step)
    assert "assemble_adversarial_step_configs" in source, (
        "the diffusion adversarial step must use the shared cadence composer, "
        "not a third copy of the N-then-one assembly"
    )
    assert "_resolve_disc_updates" in source


def test_fit_wires_a_discriminator_for_any_paradigm_not_just_gan():
    """The second gate: ``fit`` used to drop the argument unless paradigm=='gan'."""
    import inspect

    from spectramr.pipelines import fit as fit_mod

    source = inspect.getsource(fit_mod.fit)
    assert 'if disc is not None:\n        models["discriminator"] = disc' in source, (
        "discriminator wiring is gated on the paradigm name again; "
        "fit(paradigm='diffusion', discriminator=d) would silently drop it"
    )
