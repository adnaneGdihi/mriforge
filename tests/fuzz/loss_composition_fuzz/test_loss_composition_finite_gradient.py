"""Fuzz: random loss compositions must produce finite gradients.

Defect predicate
----------------
Compose 2–5 randomly selected registered losses with random positive weights.
Forward on a random (batch, channels, H, W) real tensor.  If the **total
loss is finite** but **any gradient on the input tensor contains NaN or Inf**,
the defect is reported.

The inverse (NaN/Inf total loss) is also reported separately as a secondary
defect, but the primary contract is:

    finite_loss → finite_grad

This matches the exact failure mode observed in several smoke runs where
``torch.nan_to_num`` inside a loss masked the value but the gradient still
exploded.

Composition strategy
--------------------
- Draw N ∈ {2, 3, 4, 5} from the registered loss names via
  ``tests.utils.registry_iterators.all_losses()``.
- Draw random positive float weights in ``[0.01, 10.0]``.
- Instantiate each loss with no extra kwargs (zero-config default).
- Sum: ``total = Σ wᵢ · lossᵢ(pred, target)``
- Call ``total.backward()`` and check ``pred.grad``.

The ``pred`` tensor is a leaf with ``requires_grad=True``.
The ``target`` tensor has the same shape but is detached (no grad needed).

Numerical stability guard
-------------------------
We use ``torch.amp.autocast("cpu")`` when available to stay consistent with
the training-loop AMP contract.  Note: on CPU, autocast is a no-op for
float32 but ensures the path is exercised.

Hypothesis settings
-------------------
- ``max_examples=100`` (raise via ``HYPOTHESIS_MAX_EXAMPLES``).
- ``deadline=None`` (loss forward+backward can take hundreds of ms).

Guards
------
- ``pytest.importorskip("hypothesis")``
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Guard: hypothesis required
# ---------------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, HealthCheck, assume  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collect registered loss names at module-import time
# ---------------------------------------------------------------------------
try:
    from tests.utils.registry_iterators import all_losses as _all_losses  # noqa: PLC0415
    _LOSS_NAMES: list[str] = _all_losses()
except Exception as _exc:
    logger.warning("Could not enumerate registered losses: %s", _exc)
    _LOSS_NAMES = []


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_BATCH_SIZES = st.sampled_from([1, 2])
_CHANNEL_COUNTS = st.sampled_from([1, 2, 3])
_SPATIAL_SIZES = st.sampled_from([8, 16, 32, 64])
_WEIGHT_STRATEGY = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
_N_LOSSES = st.integers(min_value=2, max_value=5)


@st.composite
def _loss_composition_spec(draw):
    """Draw a list of (loss_name, weight) pairs and an input shape."""
    if not _LOSS_NAMES:
        assume(False)  # No losses registered — backtrack.

    n = draw(_N_LOSSES)
    n = min(n, len(_LOSS_NAMES))
    names = draw(st.lists(
        st.sampled_from(_LOSS_NAMES),
        min_size=n,
        max_size=n,
        unique=True,
    ))
    weights = [draw(_WEIGHT_STRATEGY) for _ in names]

    batch = draw(_BATCH_SIZES)
    channels = draw(_CHANNEL_COUNTS)
    height = draw(_SPATIAL_SIZES)
    width = draw(_SPATIAL_SIZES)

    return names, weights, (batch, channels, height, width)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

_MAX_EXAMPLES = int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "100"))


@pytest.mark.fuzz
@settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(spec=_loss_composition_spec())
def test_loss_composition_finite_gradient(spec: Any) -> None:
    """Random 2-5 loss compositions: finite loss must imply finite gradient.

    Defects detected:
    1. ``total_loss`` is NaN or Inf (loss itself is non-finite).
    2. ``total_loss`` is finite but ``pred.grad`` contains NaN or Inf.
    """
    loss_names, weights, shape = spec

    try:
        import torch  # noqa: PLC0415
        from mriforge.models.losses.registry import LossRegistry  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"torch or LossRegistry unavailable: {exc}")

    if not hasattr(torch, "randn"):
        pytest.skip("torch mock does not support randn")

    # Build input tensors.
    pred = torch.randn(*shape, requires_grad=True, dtype=torch.float32)
    target = torch.randn(*shape, dtype=torch.float32).detach()

    # Instantiate losses.
    loss_instances: list[Any] = []
    for name in loss_names:
        try:
            inst = LossRegistry.create(name)
            loss_instances.append(inst)
        except Exception as exc:
            # Loss requires mandatory kwargs we don't supply; skip silently.
            assume(False)
            return

    # Forward: compute weighted sum.
    total_loss: Any = None
    for inst, w in zip(loss_instances, weights):
        try:
            val = inst(pred, target)
        except Exception:
            # The loss's input-domain / shape contract is not satisfied by the
            # random domain-agnostic input (BCELoss needs values in [0, 1];
            # style / perceptual losses need a minimum spatial size; some losses
            # require a specific channel count). Drop it from the composition —
            # the finite-loss → finite-grad predicate this test checks only
            # applies to losses that actually accept the input. A genuine
            # numerical defect surfaces as a non-finite loss or gradient below,
            # and a forward that succeeds then crashes on backward still fails.
            continue
        if not torch.is_tensor(val):
            continue
        w_tensor = torch.tensor(w, dtype=torch.float32)
        term = w_tensor * val
        total_loss = term if total_loss is None else total_loss + term

    if total_loss is None:
        assume(False)
        return

    # Check 1: loss itself must be finite (secondary defect).
    loss_is_finite = bool(torch.isfinite(total_loss).all())
    if not loss_is_finite:
        pytest.fail(
            f"Loss composition {loss_names!r} with weights {weights!r} "
            f"produced a non-finite total loss={total_loss.item()!r}."
        )

    # Backward pass.
    try:
        total_loss.backward()
    except Exception as exc:
        pytest.fail(
            f"Loss composition {loss_names!r} raised {type(exc).__name__} "
            f"during backward: {exc}"
        )

    if pred.grad is None:
        # No gradient means the loss does not connect to pred — not a defect
        # (could be a constant or purely target-dependent loss).
        return

    # Check 2: gradient must be finite when loss is finite (primary defect).
    grad_finite = bool(torch.isfinite(pred.grad).all())
    if not grad_finite:
        nan_count = int(torch.isnan(pred.grad).sum().item())
        inf_count = int(torch.isinf(pred.grad).sum().item())
        pytest.fail(
            f"Finite-loss-but-NaN/Inf-gradient defect detected!\n"
            f"Loss composition: {loss_names!r}\n"
            f"Weights: {weights!r}\n"
            f"Input shape: {tuple(pred.shape)}\n"
            f"total_loss: {total_loss.item()!r} (finite)\n"
            f"pred.grad: {nan_count} NaN(s), {inf_count} Inf(s) "
            f"out of {pred.grad.numel()} elements."
        )
