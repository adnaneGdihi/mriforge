"""Regression: the loss-forward probe must not report a real ``'NoneType' object
is not callable`` failure as a HEALTHY loss.

``_test_loss_forward`` tolerates the GAN-loss signature-mismatch case (a loss
whose ``forward`` raises "missing … argument" because it needs a discriminator).
The tolerance clause was written as::

    if "not callable" in err or "missing" in err and "argument" in err:

which Python binds as ``("not callable" in err) OR ("missing" in err AND
"argument" in err)`` — so the *standalone* ``"not callable"`` clause silently
passed any loss whose forward raised ``'NoneType' object is not callable`` (a
mis-wired sub-module), a false-green audit verdict (the loss-audit false-GREEN
class). The fix drops the over-broad clause; only the genuine
missing-argument signature is tolerated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.loss_audit import UNPROBED, _test_loss_forward


class _NoneCallableLoss(nn.Module):
    """A genuinely broken loss: a required sub-module was never wired, so the
    forward calls ``None(...)`` and raises ``'NoneType' object is not callable``."""

    def __init__(self) -> None:
        super().__init__()
        self.head = None  # never injected

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head(pred)  # type: ignore[misc]  -> TypeError: NoneType not callable


class _MissingArgLoss(nn.Module):
    """A GAN-style loss that needs a discriminator: calling it with (pred,
    target) raises 'missing 1 required positional argument'. This IS tolerated."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, discriminator
    ) -> torch.Tensor:
        return discriminator(pred).mean()


def test_not_callable_is_reported_as_failure_not_healthy() -> None:
    err = _test_loss_forward(_NoneCallableLoss())
    assert err is not None, (
        "a real 'NoneType is not callable' forward error was reported as a "
        "HEALTHY loss (false-green audit)"
    )
    assert "not callable" in err.lower()


def test_missing_argument_signature_is_unprobed_not_healthy() -> None:
    """A discriminator-needing loss is UNPROBED, not HEALTHY.

    This test previously asserted ``_test_loss_forward(_MissingArgLoss()) is None`` — i.e.
    it *pinned the fail-open*. The probe cannot supply a discriminator, so it raised
    "missing 1 required positional argument", and the audit matched that substring and
    returned ``None`` = healthy. Every loss with a required extra argument therefore passed
    the audit **by failing it**, and the audit could not be trusted to validate a migration
    that moves losses out of ``STRATEGY_MANAGED_LOSSES`` into the probed set.

    The probe now decides from the *signature*, before running: a loss it cannot call is
    reported UNPROBED (a counted warning), never silently green.
    """
    result = _test_loss_forward(_MissingArgLoss())

    assert result is not None, (
        "a loss the probe cannot even call was reported as HEALTHY — this is the "
        "fail-open that let an unverified loss ship green"
    )
    assert result.startswith(UNPROBED)
    assert "discriminator" in result, "the unprobeable parameter should be named"


def test_unprobed_is_not_an_error_either() -> None:
    """UNPROBED is a third state: not a pass, but not a forward failure to be fixed.

    ``audit_losses`` routes it to ``registration.warning``, so it is visible and counted
    without falsely marking a correct-but-context-needing loss as broken.
    """
    unprobed = _test_loss_forward(_MissingArgLoss())
    genuine_error = _test_loss_forward(_NoneCallableLoss())

    assert unprobed is not None and unprobed.startswith(UNPROBED)
    assert genuine_error is not None and not genuine_error.startswith(UNPROBED)
