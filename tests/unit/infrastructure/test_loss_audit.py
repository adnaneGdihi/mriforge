"""The loss-audit probe must not report an unverified loss as HEALTHY.

Partner test for ``src/mriforge/infrastructure/loss_audit.py``.

The probe used to decide "is this loss OK?" from the *text* of the exception it raised::

    if "missing" in err_lower and "argument" in err_lower:
        return None        # -> HEALTHY

That is fail-open. A loss whose ``forward`` genuinely requires a discriminator, a sampling
mask or a coil map cannot be called with two random tensors, so it raised
``missing 1 required positional argument`` and the audit rubber-stamped it. The audit was
therefore incapable of validating the very migration that moves losses out of
``STRATEGY_MANAGED_LOSSES`` and into the probed set: every one of them would have gone
green on arrival.

The probe now decides from the **signature**, before running, and reports three distinct
states: HEALTHY (probed, passed), UNPROBED (no probe constructible — a counted warning,
never a pass), ERROR (probed, failed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.loss_audit import (
    UNPROBED,
    _test_loss_forward,
    _unsatisfiable_params,
)
from mriforge.models.losses.registry import create_loss


class _PairLoss(nn.Module):
    """A well-behaved (pred, target) loss — fully probeable."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()


class _OptionalContextLoss(nn.Module):
    """Extra context, but *defaulted* — the probe can still call it."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smaps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (pred - target).abs().mean()


class _RequiredContextLoss(nn.Module):
    """Extra context with NO default — the probe can never call it."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return ((pred - target) * mask).abs().mean()


class TestUnsatisfiableParams:
    def test_pair_loss_is_fully_probeable(self) -> None:
        assert _unsatisfiable_params(_PairLoss()) == []

    def test_defaulted_context_is_satisfiable(self) -> None:
        """An optional param is not a blocker — the probe simply takes the default.

        This distinction is load-bearing: 30 registered losses take optional context
        (``sense_adjoint_l1(pred, target, smaps=None)``). Treating them as unprobeable
        would flood the audit with false UNPROBEDs.
        """
        assert _unsatisfiable_params(_OptionalContextLoss()) == []

    def test_required_context_is_named(self) -> None:
        assert _unsatisfiable_params(_RequiredContextLoss()) == ["mask"]


class TestProbeStates:
    def test_probeable_loss_passes(self) -> None:
        assert _test_loss_forward(_PairLoss()) is None

    def test_required_context_loss_is_unprobed_not_healthy(self) -> None:
        result = _test_loss_forward(_RequiredContextLoss())
        assert result is not None, "an uncallable loss was reported HEALTHY (fail-open)"
        assert result.startswith(UNPROBED)
        assert "mask" in result


class TestAgainstTheRealRegistry:
    """The concrete losses this refactor will move out of STRATEGY_MANAGED_LOSSES."""

    def test_data_consistency_is_unprobed(self) -> None:
        """``data_consistency(pred_image, measured_kspace, mask)`` needs a mask.

        It is declared by a live arm and is the ONE loss that is unprobeable today while
        NOT skip-listed — i.e. the only place the old fail-open was actually load-bearing.
        Closing it now costs one arm; closing it after the skip-set shrinks would cost
        every context loss at once.
        """
        assert _unsatisfiable_params(create_loss("data_consistency")) == ["mask"]

    def test_r1_regularization_is_unprobed(self) -> None:
        """R1's FIRST slot is a discriminator, not a prediction.

        ``forward(discriminator: Any, real_images: Tensor, *args, **kwargs)``. A check that
        only inspected parameters *past* the first two would call this "probeable", then feed
        a random tensor into the discriminator slot, where it dies on ``discriminator(x)``
        -> "'Tensor' object is not callable". About 40 of the 204 registered losses do not
        carry (pred, target) in slots 0 and 1, so the probe must vet those slots too — not
        just the tail.

        Still tolerated (UNPROBED, not an error) — but no longer counted as a verified pass.
        """
        assert _unsatisfiable_params(create_loss("r1_regularization")) == [
            "discriminator"
        ]

    def test_plain_l1_is_probeable(self) -> None:
        assert _unsatisfiable_params(create_loss("l1")) == []
