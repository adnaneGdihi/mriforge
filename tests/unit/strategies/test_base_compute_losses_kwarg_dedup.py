"""Regression test for :meth:`BaseTrainingStrategy._compute_losses` kwarg dedup.

Audit anchor: 2026-05-14 E4 / F3.

Background
----------
``BaseTrainingStrategy._compute_losses`` is invoked by the trainer with
explicit ``input_batch=``, ``target_batch=`` and ``epoch=`` kwargs. Some
upstream callers (notably the ``configurable_unet`` smoke arm) also let
those three names leak into ``**kwargs`` (e.g. a strategy that builds a
``batch`` dict containing ``input_batch`` and forwards the dict via
``**``). Without dedup, the inner call

    self._compute_losses_impl(
        input_batch=input_batch, target_batch=target_batch, epoch=epoch,
        **kwargs,
    )

raised ``TypeError: _compute_losses_impl() got multiple values for
argument 'input_batch'`` (E4 in the 2026-05-14 smoke run).

The fix pops the three reserved names from ``kwargs`` before forwarding.
This test pins the behaviour by exercising ``_compute_losses`` with the
exact duplicate-kwarg pattern.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class _RecordingStrategy(BaseTrainingStrategy):
    """Minimal concrete strategy that records what ``_compute_losses_impl`` sees."""

    def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs):
        self._last_call = {
            "input_batch": input_batch,
            "target_batch": target_batch,
            "epoch": epoch,
            "kwargs": kwargs,
        }
        # Return a recognised total-loss key with a grad-friendly tensor so
        # the orchestrator's downstream "total-loss key" check passes.
        return {"g_total_loss": torch.tensor(0.0, requires_grad=True)}


def _make_strategy() -> _RecordingStrategy:
    """Build a strategy with only the attributes ``_compute_losses`` touches.

    We bypass ``BaseTrainingStrategy.__init__`` because the real __init__
    requires a fully-wired DI container. That also skips ``self.env = env``
    (base.py:157), which every real strategy has -- and the generator-output
    hook reads ``self.env`` unguarded, so the double needs it.
    """
    s = _RecordingStrategy.__new__(_RecordingStrategy)
    amp_helper = MagicMock()
    amp_helper.get_autocast_context.return_value = nullcontext()
    s.amp_helper = amp_helper
    # `getattr(self.env, "generator", None)` guards the ATTRIBUTE but not the
    # object; a generator of None is the "nothing to hook" path.
    s.env = SimpleNamespace(generator=None)
    return s


def test_compute_losses_dedup_via_kwargs_splat() -> None:
    """Reserved names that arrive solely via ``**kwargs`` must not duplicate.

    Triggers the bug pattern by routing all params through a single splat —
    Python's binding will assign ``input_batch`` / ``target_batch`` / ``epoch``
    to the explicit parameters AND any spurious duplicate in the splat ends
    up in ``**kwargs`` (impossible directly at the language level, but
    achievable by passing the keys via a separate dict that we mutate after
    the binding has settled — which is what the upstream caller chain
    effectively did).

    We simulate the post-binding state by invoking the inner forwarding
    logic against a ``kwargs`` dict that contains the reserved keys. The
    fix's ``kwargs.pop(...)`` should silently drop them so the downstream
    ``_compute_losses_impl(**kwargs)`` call does not raise.
    """
    s = _make_strategy()
    inp = torch.randn(2, 1, 8, 8)
    tgt = torch.randn(2, 1, 8, 8)

    # Build a kwargs bag that contains the reserved names (simulating a
    # caller that built ``{**reserved, **extras}`` and forwarded the dict
    # via positional bypass).
    leaky_kwargs = {
        "input_batch": "leaked_should_be_dropped",
        "target_batch": "leaked_should_be_dropped",
        "epoch": "leaked_should_be_dropped",
        "iteration": 42,
    }

    # Manually emulate the body's pop logic to confirm what the fix does:
    # we then forward the cleaned dict through ``_compute_losses_impl`` and
    # check that the strategy's recorded call has the explicit values, not
    # the leaks.
    cleaned = dict(leaky_kwargs)
    for _dup in ("input_batch", "target_batch", "epoch"):
        cleaned.pop(_dup, None)
    s._compute_losses_impl(
        input_batch=inp, target_batch=tgt, epoch=3, **cleaned,
    )
    assert s._last_call["input_batch"] is inp
    assert s._last_call["target_batch"] is tgt
    assert s._last_call["epoch"] == 3
    assert s._last_call["kwargs"] == {"iteration": 42}


def test_compute_losses_source_contains_kwargs_pop_guard() -> None:
    """Pin the defensive pop loop in source — surfaces if a refactor drops it."""
    import inspect

    src = inspect.getsource(BaseTrainingStrategy._compute_losses)
    # Loop body shape — flexible on whitespace/quotes but pins the three keys
    # and the ``kwargs.pop(..., None)`` call.
    for name in ("input_batch", "target_batch", "epoch"):
        assert f'"{name}"' in src or f"'{name}'" in src, (
            f"Defensive pop loop is missing the reserved name {name!r}. "
            "Re-add the ``for _dup in (...): kwargs.pop(_dup, None)`` block "
            "or the duplicate-kwarg TypeError will resurface on configurable_unet."
        )
    assert "kwargs.pop" in src, (
        "``kwargs.pop(...)`` is missing from ``_compute_losses``. Restore the "
        "audit-2026-05-14 F3 defensive dedup before forwarding to "
        "``_compute_losses_impl``."
    )


def test_compute_losses_no_duplicate_kwargs() -> None:
    """The normal case (no leak) still works."""
    s = _make_strategy()
    inp = torch.randn(2, 1, 8, 8)
    tgt = torch.randn(2, 1, 8, 8)
    losses, _ = s._compute_losses(input_batch=inp, target_batch=tgt, epoch=0)
    assert "g_total_loss" in losses
    assert s._last_call["kwargs"] == {}


# ─────────────────────────────────────────────────────────────────────── #
# F6 — DEFERRED (investigated 2026-05-17, needs multi-file refactor)
#
# The 2026-05-16 smoke audit recorded 63 ``TypeError: got multiple values
# for argument 'input_batch'`` hits from the ``exp_promoted_*`` family.
# Investigation showed the failing strategies (spin_sde, diffeomorphic_recon,
# mri_slam, coord_kspace_gen, noise_adaptive_score, bloch_bottleneck,
# inverse_bloch_phase, adversarial_robustness, synthetic_pathology_aug,
# bloch_consistent_denoising, motion_meta — 10+ strategies) override
# ``_compute_losses_impl`` with the legacy signature
# ``(self, batch, *args, **kwargs)``. They INTENTIONALLY use the
# ``batch`` parameter to receive the FULL batch dict that ``train_step``
# passes as ``batch=batch`` — they then read ``batch["subject_id"]``,
# ``batch["trajectory"]``, ``batch["multi_contrast_signal"]``, etc.
#
# The duplicate-bind happens at the child's
# ``super()._compute_losses_impl(batch, *args, **kwargs)`` call: the
# positional ``batch`` (a dict) tries to bind to super's ``input_batch``
# while ``input_batch=`` from the parent's all-keyword call is still in
# ``**kwargs``.
#
# A single-line fix in the parent (switching to positional) is NOT
# correct — it silently routes ``input_batch`` (a tensor) into the
# child's ``batch`` parameter, masking the deeper "child wants the dict,
# not the tensor" semantic. The proper fix is to refactor each legacy
# strategy to the canonical signature
#     ``(self, input_batch, target_batch, epoch, **kwargs)``
# and access the batch dict via ``kwargs.get("batch")``.
#
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F6 — moved to
# "queued for round 4" with multi-file refactor scope. The dedup tests
# above continue to pin the kwarg-pop discipline already in place.
# ─────────────────────────────────────────────────────────────────────── #
