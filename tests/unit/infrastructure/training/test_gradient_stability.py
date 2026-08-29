"""Regression tests for gradient_stability raise-on-unknown contract (NN#3).

OPT-002: ``EnhancedLRScheduler.create`` must raise ``ValueError`` on an
unknown scheduler strategy rather than logging a warning and returning
``None`` (a silent fall-through to a fixed-LR run).

OPT-003: ``GradientStabilityManager._perform_gradient_clipping`` must raise
``ValueError`` on an unknown ``clip_method`` rather than silently applying
no clipping for the whole run.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.gradient_stability import (
    EnhancedLRScheduler,
    GradientStabilityManager,
    Lookahead,
    adaptive_gradient_clip_,
)


class TestEnhancedLRScheduler:
    def test_unknown_strategy_raises(self) -> None:
        """OPT-002: unknown scheduler strategy must raise, never return None."""
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = EnhancedLRScheduler(strategy="__nope__", total_epochs=10)
        with pytest.raises(ValueError, match="Unknown scheduler strategy"):
            sched.create(optimizer, steps_per_epoch=100)

    def test_known_strategy_still_builds(self) -> None:
        """A valid strategy must still return a scheduler (no regression)."""
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = EnhancedLRScheduler(strategy="cosine_annealing", total_epochs=10)
        result = sched.create(optimizer, steps_per_epoch=100)
        assert result is not None


class TestGradientStabilityManager:
    def _model_with_grad(self) -> nn.Module:
        model = nn.Linear(2, 1)
        loss = model(torch.randn(4, 2)).pow(2).mean()
        loss.backward()
        return model

    def test_unknown_clip_method_raises(self) -> None:
        """OPT-003: unknown clip_method must raise, never silently no-op."""
        model = self._model_with_grad()
        mgr = GradientStabilityManager(
            clip_gradients=True,
            clip_method="__bogus__",
            monitor_gradients=False,
        )
        with pytest.raises(ValueError, match="Unknown clip_method"):
            mgr._perform_gradient_clipping(model)

    def test_known_clip_method_does_not_raise(self) -> None:
        """A valid clip_method must still clip without error (no regression)."""
        model = self._model_with_grad()
        mgr = GradientStabilityManager(
            clip_gradients=True,
            clip_method="norm",
            clip_value=1.0,
            monitor_gradients=False,
        )
        # Should not raise.
        mgr._perform_gradient_clipping(model)


class TestAdaptiveGradientClipEps:
    """Bug 1: the AGC eps must floor the *parameter* norm, not the grad norm.

    For a zero-initialised parameter the parameter norm is 0, so the clipping
    ceiling ``max_norm = param_norm * clip_factor`` is 0.  If eps floors the
    parameter norm (correct AGC, Brock et al. 2021) the ceiling becomes
    ``eps * clip_factor`` -> a tiny but NON-ZERO scale, so the gradient is
    shrunk but survives.  If eps is (incorrectly) applied to the grad-norm
    divisor instead, the gradient is multiplied by ``0 / max(grad_norm, eps)``
    and is annihilated, permanently freezing the zero-init layer.
    """

    def test_zero_init_param_grad_not_annihilated(self) -> None:
        p = nn.Parameter(torch.zeros(4, 4))
        p.grad = torch.ones(4, 4) * 10.0

        adaptive_gradient_clip_([p], clip_factor=0.01, eps=1e-3)

        # Correct AGC keeps a non-zero (eps-floored) gradient; the buggy
        # grad-norm-eps placement zeroes it out entirely.
        assert float(p.grad.norm()) > 0.0, (
            "Zero-init param grad was annihilated: AGC eps must floor the "
            "parameter norm (max(param_norm, eps)), not the grad norm."
        )

    def test_eps_floor_matches_reference_ceiling(self) -> None:
        """The clip ceiling on a zero-norm param must equal ``eps*clip_factor``."""
        clip_factor, eps = 0.01, 1e-3
        p = nn.Parameter(torch.zeros(3, 3))
        # grad_norm of 5 along each output unit (row).
        g = torch.zeros(3, 3)
        g[:, 0] = 5.0
        p.grad = g.clone()

        adaptive_gradient_clip_([p], clip_factor=clip_factor, eps=eps)

        # Each row's grad norm should be clipped to eps * clip_factor.
        row_norms = p.grad.norm(dim=1)
        expected = eps * clip_factor
        assert torch.allclose(
            row_norms, torch.full_like(row_norms, expected), atol=1e-9
        ), f"row norms {row_norms.tolist()} != eps*clip_factor={expected}"


class TestAmpOverflowSkip:
    """Bug 2: ``unscale_`` then ``optimizer.step()`` defeats the AMP inf/NaN skip.

    ``GradScaler.step`` checks for inf/NaN grads and SKIPS the step on
    overflow.  Calling ``unscale_`` then ``optimizer.step()`` directly applies
    the corrupted (inf/NaN) step.  After the fix the step must route through
    ``scaler.step`` so an overflow leaves the weights untouched.
    """

    def test_overflow_step_is_skipped(self) -> None:
        mgr = GradientStabilityManager(
            clip_gradients=False,
            monitor_gradients=False,
        )
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        before = model.weight.detach().clone()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        # Build a loss whose gradient is inf (simulating an AMP overflow).
        bad = torch.tensor([[float("inf"), 1.0]])
        loss = (model.weight * bad).sum()
        scaler.scale(loss).backward()

        mgr.step_optimization(model, opt, grad_scaler=scaler)

        after = model.weight.detach()
        assert not torch.isnan(after).any() and not torch.isinf(after).any(), (
            "AMP overflow corrupted the weights: the step must route through "
            "scaler.step() so an inf/NaN grad is skipped."
        )
        assert torch.equal(after, before), "Overflow step should be skipped (no-op)."

    def test_clean_step_still_applied(self) -> None:
        """A finite gradient must still produce a real optimizer step (no regression)."""
        mgr = GradientStabilityManager(
            clip_gradients=False,
            monitor_gradients=False,
        )
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        before = model.weight.detach().clone()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        good = torch.tensor([[3.0, 1.0]])
        loss = (model.weight * good).sum()
        scaler.scale(loss).backward()

        mgr.step_optimization(model, opt, grad_scaler=scaler)

        after = model.weight.detach()
        assert not torch.equal(after, before), "Clean step must update the weights."

    def test_diffusion_overflow_step_is_skipped(self) -> None:
        """step_diffusion_optimization must also honour the overflow skip."""
        mgr = GradientStabilityManager(
            clip_gradients=False,
            monitor_gradients=False,
        )
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        before = model.weight.detach().clone()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        bad = torch.tensor([[float("inf"), 1.0]])
        loss = (model.weight * bad).sum()
        scaler.scale(loss).backward()

        mgr.step_diffusion_optimization(model, opt, grad_scaler=scaler)

        after = model.weight.detach()
        assert torch.equal(after, before), (
            "Diffusion AMP overflow step should be skipped (no-op)."
        )


class TestLookaheadStateDictRoundTrip:
    """Bug 3: Lookahead state_dict must round-trip the per-group step counter.

    The counter gates the slow-weight sync (``if counter == 0: update_slow``).
    If state_dict drops it, a resumed optimizer resets the counter to 0 and
    syncs the slow weights one step too early, corrupting the cadence.
    """

    @staticmethod
    def _advance(la: Lookahead, model: nn.Module, n: int) -> None:
        for _ in range(n):
            la.zero_grad()
            model(torch.randn(4, 3)).pow(2).mean().backward()
            la.step()

    def test_counter_round_trips(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(3, 2)
        la = Lookahead(
            torch.optim.SGD(model.parameters(), lr=0.5), la_steps=5, la_alpha=0.5
        )
        self._advance(la, model, 3)
        counters_before = [g["counter"] for g in la.param_groups]
        assert counters_before == [3]  # sanity

        sd = copy.deepcopy(la.state_dict())

        torch.manual_seed(0)
        model2 = nn.Linear(3, 2)
        la2 = Lookahead(
            torch.optim.SGD(model2.parameters(), lr=0.5), la_steps=5, la_alpha=0.5
        )
        la2.load_state_dict(sd)

        counters_after = [g["counter"] for g in la2.param_groups]
        assert counters_after == counters_before, (
            f"Lookahead step counter dropped on resume: {counters_after} != "
            f"{counters_before}"
        )

    def test_slow_weights_round_trip(self) -> None:
        """Slow weights must also survive the round-trip (no regression)."""
        torch.manual_seed(0)
        model = nn.Linear(3, 2)
        la = Lookahead(
            torch.optim.SGD(model.parameters(), lr=0.5), la_steps=5, la_alpha=0.5
        )
        self._advance(la, model, 3)
        slow_before = {
            i: la.state[p]["slow_param"].clone()
            for g in la.param_groups
            for i, p in enumerate(g["params"])
        }

        sd = copy.deepcopy(la.state_dict())

        torch.manual_seed(0)
        model2 = nn.Linear(3, 2)
        la2 = Lookahead(
            torch.optim.SGD(model2.parameters(), lr=0.5), la_steps=5, la_alpha=0.5
        )
        la2.load_state_dict(sd)

        slow_after = {
            i: la2.state[p]["slow_param"]
            for g in la2.param_groups
            for i, p in enumerate(g["params"])
            if p in la2.state and "slow_param" in la2.state[p]
        }
        assert len(slow_after) == len(slow_before)
        for i, val in slow_before.items():
            assert torch.equal(slow_after[i], val)
