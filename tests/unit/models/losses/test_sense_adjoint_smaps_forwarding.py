"""Regression tests for SENSE-adjoint loss + ``_call_safe_loss`` wiring.

Background — experiment_11_kspace_cold_diffusion was reporting
``val_sense_adjoint_l1 = 0.0`` for every cascade level. Root cause was
two layered bugs:

1. ``_compute_validation_metrics`` in the diffusion strategy was calling
   ``loss_fn(pred, target)`` directly instead of going through
   ``_call_safe_loss`` — so ``SENSEAdjointL1Loss`` never received
   ``smaps`` and triggered its ``smaps is None`` early-return
   (``return pred.sum() * 0.0``).

2. ``_cached_smaps`` is referenced by ``metrics_mixin._apply_metric_transforms``
   for the ``ifft_sense_adjoint`` transform but is never assigned anywhere
   in the codebase, so the transform silently degrades to per-coil iFFT
   magnitude.

These tests pin the contract:

* ``_call_safe_loss(loss_fn, pred, target, smaps=...)`` must forward
  ``smaps`` to ``SENSEAdjointL1Loss.forward`` and produce a non-zero
  loss for non-trivial inputs.
* The bare ``loss_fn(pred, target)`` call (no smaps) must **RAISE** during
  training. The zero-return is reachable only in eval mode or under an explicit
  ``allow_missing_smaps=True`` opt-out.

2026-07-14 (issue #188): the second bullet used to say the opposite — it pinned
the silent zero-return as expected behaviour, long after the source replaced that
fallback with a hard guard. The tests have been red on ``dev`` ever since,
invisible because GitHub Actions is disabled on this repo.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.complex_losses import SENSEAdjointL1Loss
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    _call_safe_loss,
)


def _real_stacked_kspace(B: int, C: int, H: int, W: int, seed: int) -> torch.Tensor:
    """Random real-stacked k-space tensor ([R0, I0, R1, I1, ...]) layout."""
    torch.manual_seed(seed)
    return torch.randn(B, 2 * C, H, W)


def _real_stacked_smaps(B: int, C: int, H: int, W: int, seed: int) -> torch.Tensor:
    """Random real-stacked S-maps in the same interleaved layout."""
    torch.manual_seed(seed + 1)
    raw = torch.randn(B, 2 * C, H, W)
    # Normalise so RSS across coils ≈ 1 (matches strategy's normalisation step).
    real = raw[:, 0::2]
    imag = raw[:, 1::2]
    rss = torch.sqrt((real**2 + imag**2).sum(dim=1, keepdim=True) + 1e-8)
    raw[:, 0::2] = real / rss
    raw[:, 1::2] = imag / rss
    return raw


def test_sense_adjoint_l1_raises_when_smaps_missing_during_training() -> None:
    """Missing smaps during TRAINING must RAISE — the silent zero is the bug.

    Inverted 2026-07-14 (issue #188). This test used to pin the old
    ``return pred.sum() * 0.0`` fallback as expected behaviour. That fallback IS
    the experiment_11 DC-blob root cause: a data-fidelity term multiplied by zero
    exerts no pressure toward measurement consistency, so the network is free to
    emit a constant blob and still score well. The loss now raises instead
    (CLAUDE.md #9/#15), and the test had been red on ``dev`` ever since — pinning
    the very fallback the source had deliberately removed.
    """
    loss_fn = SENSEAdjointL1Loss(reduction="mean")
    pred = _real_stacked_kspace(2, 4, 32, 32, seed=0)
    target = _real_stacked_kspace(2, 4, 32, 32, seed=1)

    assert loss_fn.training, "nn.Module defaults to train mode; guard is train-gated"
    with pytest.raises(ValueError, match="received smaps=None"):
        loss_fn(pred, target)  # no smaps kwarg


def test_sense_adjoint_l1_zero_is_reachable_only_by_explicit_opt_out() -> None:
    """The zero-return survives ONLY where it is explicitly asked for.

    Two sanctioned contexts (``complex_losses.py``): eval mode (startup /
    no-coil scoring) and an explicit ``allow_missing_smaps=True``. Anything else
    must raise — that is what keeps the fallback from creeping back in silently.
    """
    pred = _real_stacked_kspace(2, 4, 32, 32, seed=0)
    target = _real_stacked_kspace(2, 4, 32, 32, seed=1)

    eval_loss = SENSEAdjointL1Loss(reduction="mean").eval()
    assert eval_loss(pred, target).detach().abs().item() == 0.0

    opted_out = SENSEAdjointL1Loss(reduction="mean")
    opted_out.allow_missing_smaps = True
    assert opted_out.training  # still training — only the opt-out permits zero
    assert opted_out(pred, target).detach().abs().item() == 0.0


def test_call_safe_loss_forwards_smaps_to_sense_adjoint_l1() -> None:
    """Confirms `_call_safe_loss` propagates `smaps` through signature filtering.

    This is the load-bearing assertion: it would have caught the
    experiment_11 bug at unit-test time. With smaps forwarded the loss
    must compute a non-trivial value.
    """
    loss_fn = SENSEAdjointL1Loss(reduction="mean")
    pred = _real_stacked_kspace(2, 4, 32, 32, seed=0)
    target = _real_stacked_kspace(2, 4, 32, 32, seed=1)
    smaps = _real_stacked_smaps(2, 4, 32, 32, seed=2)

    routed = _call_safe_loss(loss_fn, pred, target, smaps=smaps)
    assert routed.detach().abs().item() > 1e-6, (
        f"SENSEAdjointL1Loss with smaps forwarded via _call_safe_loss "
        f"must produce a non-zero residual loss for distinct random "
        f"pred/target inputs; got {routed.item()}."
    )


def test_call_safe_loss_filters_unrelated_kwargs() -> None:
    """Sanity: unrelated kwargs (e.g. ``mask`` / ``timesteps``) do not crash.

    The `_call_safe_loss` filter must drop kwargs the loss doesn't accept.
    SENSEAdjointL1Loss has ``**kwargs`` so it accepts everything, but the
    filter should still gracefully handle unknown extras coming from the
    training/validation contexts.
    """
    loss_fn = SENSEAdjointL1Loss(reduction="mean")
    pred = _real_stacked_kspace(2, 4, 32, 32, seed=0)
    target = _real_stacked_kspace(2, 4, 32, 32, seed=1)
    smaps = _real_stacked_smaps(2, 4, 32, 32, seed=2)

    out = _call_safe_loss(
        loss_fn,
        pred,
        target,
        smaps=smaps,
        mask=torch.ones(2, 1, 32, 32),
        timesteps=torch.zeros(2, dtype=torch.long),
        discriminator=None,
    )
    assert out.detach().abs().item() > 1e-6


def test_zero_residual_drives_sense_adjoint_l1_to_zero() -> None:
    """When pred == target, the SENSE-projected L1 must be zero.

    Catches accidental sign / conjugation bugs in future refactors of the
    SENSE-adjoint pipeline (the loss must respect the identity contract
    even after going through ifft2c + smap projection).
    """
    loss_fn = SENSEAdjointL1Loss(reduction="mean")
    target = _real_stacked_kspace(2, 4, 32, 32, seed=7)
    smaps = _real_stacked_smaps(2, 4, 32, 32, seed=8)

    out = _call_safe_loss(loss_fn, target.clone(), target, smaps=smaps)
    assert (
        out.detach().abs().item() < 1e-5
    ), f"SENSEAdjointL1Loss(pred=target) must vanish; got {out.item()}."
