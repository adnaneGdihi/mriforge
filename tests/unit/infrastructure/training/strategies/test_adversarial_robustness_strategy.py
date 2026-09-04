"""Tests for AdversarialRobustnessStrategy loss-dict key normalisation.

Regression for the A-8.3 review BLOCKER (2026-06-16): the merge read ``loss_total``
from the parent's dict, but ReconstructionTrainingStrategy emits ``g_total_loss``
(+ a ``loss`` alias), so ``loss_total`` was always None — the merged dict ended up
with NO grad-carrying total, which (a) would raise "No g_total_loss" downstream and
(b) made the CertRob Lipschitz penalty inert (#16). The fix reads the canonical
``g_total_loss`` and writes it back.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from spectramr.infrastructure.training.strategies.adversarial_robustness_strategy import (
    AdversarialRobustnessStrategy,
)


def _make_strategy(gen: nn.Module) -> AdversarialRobustnessStrategy:
    strat = object.__new__(AdversarialRobustnessStrategy)
    strat.env = types.SimpleNamespace(generator=gen)
    strat.clean_weight = 0.1
    strat.adv_weight = 0.9
    strat.epsilon = 0.01
    strat.attack_steps = 2
    strat.attack_norm = "linf"
    return strat


def test_merge_emits_grad_carrying_g_total_loss(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.adversarial_robustness_strategy as mod

    # The grandparent (ReconstructionTrainingStrategy) emits g_total_loss, NOT loss_total.
    def _fake_super(self, input_batch=None, target_batch=None, epoch=0, **k):
        return {"g_total_loss": (input_batch.sum() * 0.0 + 1.0).requires_grad_(True), "l1": torch.tensor(0.5)}

    monkeypatch.setattr(mod.ReconstructionTrainingStrategy, "_compute_losses_impl", _fake_super)
    # Skip the actual PGD attack — return the input unchanged.
    monkeypatch.setattr(mod.AdversarialRobustnessStrategy, "pgd_attack", lambda self, x, loss_fn: x)

    gen = nn.Conv2d(1, 1, 1)
    strat = _make_strategy(gen)
    batch = {"input": torch.rand(2, 1, 8, 8, requires_grad=True), "target": torch.rand(2, 1, 8, 8)}

    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0, batch=batch)

    # The canonical key must exist, be a tensor, and carry gradient (for backward).
    assert isinstance(out.get("g_total_loss"), torch.Tensor)
    assert out["g_total_loss"].requires_grad
    # No stale loss_total key (the old broken contract).
    assert "loss_total" not in out
    # Components are forwarded with clean_/adv_ prefixes (detached), not the totals.
    assert "clean_l1" in out and "adv_l1" in out
    assert "clean_g_total_loss" not in out  # the total is merged, not duplicated as a component
