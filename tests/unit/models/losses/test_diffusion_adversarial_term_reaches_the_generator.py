"""The adversarial term reaches the diffusion generator, or the run refuses.

``UnifiedDiffusionLossComputer.compute`` had an adversarial branch guarded on
``lambda_adv > 0 and discriminator and self.adversarial_loss_fn`` whose body was
a bare ``pass`` -- and ``adversarial_loss_fn`` was set to ``None`` in
``__init__`` and assigned nowhere, so the branch could never be entered anyway.
An arm could declare a non-zero adversarial weight, attach a discriminator,
satisfy every visible precondition, and receive NO adversarial gradient. The
comment said "usually handled in strategy"; ``DiffusionTrainingStrategy`` had no
adversarial term either (#1669).

The failure mode is why these assert on ``components``, not on configuration:
the arm trains either way, the logs read like an adversarial run, and only the
NUMBER is wrong. ``test_discriminator_on_a_diffusion_arm_actually_trains``
passed throughout the period the term was dead, because it asserts the CRITIC's
weights move -- which they do, on the critic's own step, whether or not the
generator ever hears the answer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class _Critic(nn.Module):
    def forward(self, x):  # noqa: D102
        return x.mean(dim=(1, 2, 3), keepdim=True)


def _computer(weight: float):
    from spectramr.models.losses.computers import UnifiedDiffusionLossComputer

    c = UnifiedDiffusionLossComputer.__new__(UnifiedDiffusionLossComputer)
    c.config = None
    c.device = torch.device("cpu")
    c.diffusion_loss_fn = None
    c.reconstruction_loss_fn = None
    c.adversarial_loss_fn = None
    c.discriminator = None
    c._recon_fallback_name = None
    # ``_weight_table`` is a cached property backed by ``_loss_weight_table``;
    # seed the backing attribute directly so no config parse is needed here.
    from spectramr.models.losses.weights import LossWeightSpec, LossWeightTable

    c._loss_weight_table = LossWeightTable(
        {
            "adversarial": LossWeightSpec(
                name="adversarial",
                weight=weight,
                enabled=weight > 0,
                source="test",
                warmup_gated=False,
            )
        },
        warmup_iterations=0,
        warmup_losses=frozenset(),
    )
    return c


def test_declared_weight_with_a_critic_and_no_loss_object_raises():
    """The dead-branch shape: preconditions met, nothing computed.

    Before, this configuration silently produced a total loss with no
    adversarial component. Dropping a declared loss term silently is the
    substitution non-negotiable 3 forbids, so it must raise instead.
    """
    c = _computer(weight=1.0)
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    with pytest.raises(ValueError, match="adversarial"):
        c.compute(pred=pred, target=target, epoch=0, iteration=0, discriminator=_Critic())


def test_zero_weight_is_not_an_error():
    """A plain diffusion arm is untouched -- no critic, no term, no raise."""
    c = _computer(weight=0.0)
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    out = c.compute(pred=pred, target=target, epoch=0, iteration=0, discriminator=_Critic())
    assert "adversarial" not in out.components


def test_the_term_is_computed_and_carries_gradient_when_wired():
    """With a loss object present the term appears AND reaches ``pred``.

    Asserting the gradient, not just the key: a component recorded into the
    dict but detached from ``pred`` would still leave the generator untrained,
    which is the same defect wearing a passing test.
    """
    c = _computer(weight=2.0)
    c.adversarial_loss_fn = lambda fake_pred, is_real: fake_pred.mean()
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    out = c.compute(pred=pred, target=target, epoch=0, iteration=0, discriminator=_Critic())

    assert "adversarial" in out.components, out.components
    grad = torch.autograd.grad(out.components["adversarial"], pred, allow_unused=True)[0]
    assert grad is not None and torch.any(grad != 0), (
        "the adversarial component does not reach the generator's prediction"
    )


def test_the_critic_is_frozen_for_the_generator_step():
    """D's parameters must not accumulate gradient from the GENERATOR's objective.

    ``GANTrainingStrategy._train_generator_step`` freezes the critic for the
    duration of the G step; the diffusion path did not. Once the generator-side
    adversarial term is actually computed, ``discriminator(pred)`` runs inside
    the G step and D's parameters take gradient from G's objective. Under
    gradient accumulation those survive to D's own ``optimizer.step()`` -- the
    "G trains D" leak. With accumulation off, ``opt_d.zero_grad`` at the next D
    step clears them, which is exactly what makes this visible only in the
    configuration nobody smoke-tests.

    Freezing the parameters must NOT cut the graph: the adversarial term still
    has to reach ``pred`` through D. Both halves are asserted, because a fix
    that froze too much would silently stop training the generator.
    """
    critic = nn.Linear(16, 1)
    critic.requires_grad_(False)  # what the G step does

    c = _computer(weight=1.0)
    c.adversarial_loss_fn = lambda fake_pred, is_real: fake_pred.mean()
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)

    class _Flat(nn.Module):
        def forward(self, x):
            return critic(x.reshape(x.shape[0], -1))

    out = c.compute(pred=pred, target=target, epoch=0, iteration=0, discriminator=_Flat())
    out.components["adversarial"].backward()

    assert all(p.grad is None for p in critic.parameters()), (
        "the critic accumulated gradient from the generator's objective"
    )
    assert pred.grad is not None and torch.any(pred.grad != 0), (
        "freezing the critic also cut the generator's adversarial gradient"
    )
