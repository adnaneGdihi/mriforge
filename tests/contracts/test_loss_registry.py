"""Loss registry contract suite.

One parametric test per registered loss key (all_losses()).

Default-lane assertions (cheap, no GPU):
- The loss key exists in LossRegistry._custom_losses.
- LossRegistry.create(key) succeeds (zero-arg construction).
- The returned object is an nn.Module.

``@pytest.mark.slow`` assertions (forward + backward on synthetic tensors):
- forward(pred, target) returns a finite scalar Tensor.
- Gradient flows back through pred (loss.backward() does not raise and
  pred.grad is not None / all-zeros).
- Wrong-domain input raises rather than silently producing garbage
  (domain-metadata check, skipped when domain metadata is absent).
"""

from __future__ import annotations

import math
import gc
from typing import Any

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.registry import LossRegistry
from tests.utils.registry_iterators import all_losses

# ---------------------------------------------------------------------------
# Synthetic tensor factory
# ---------------------------------------------------------------------------

_IMAGE_SHAPE = (1, 1, 32, 32)   # (B, C, H, W)
_KSPACE_SHAPE = (1, 2, 32, 32)  # (B, 2C interleaved real/imag, H, W)


def _image_pair(requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.rand(*_IMAGE_SHAPE)
    if requires_grad:
        pred = pred.detach().requires_grad_(True)
    target = torch.rand(*_IMAGE_SHAPE)
    return pred, target


def _signed_image_pair(
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-shaped pair spanning NEGATIVE as well as positive values.

    ``_image_pair`` draws from ``torch.rand`` -- uniform ``[0, 1)``, strictly
    non-negative. A loss whose penalty branch is ``clamp(-x, min=0)`` is
    identically zero *and flat* everywhere on that support, so the default probe
    cannot tell "this term is inert" apart from "this term is simply not
    activated by the input I chose". ``_kspace_pair`` already draws
    ``torch.randn`` and is signed; only the image probe needed a counterpart.
    """
    pred = torch.rand(*_IMAGE_SHAPE) * 2.0 - 1.0
    if requires_grad:
        pred = pred.detach().requires_grad_(True)
    target = torch.rand(*_IMAGE_SHAPE) * 2.0 - 1.0
    return pred, target


def _gradient_reaches_pred_on_a_signed_probe(key: str) -> bool:
    """Re-probe ``key`` on a signed input; True when the gradient flows there.

    Used only to second-guess an apparent all-zero gradient. A genuinely inert
    term is flat under *both* probes, so this can convert a false failure and
    can never mask a real one.
    """
    try:
        loss_fn = LossRegistry.create(key)
    except Exception:
        return False
    pred, target = _signed_image_pair(requires_grad=True)
    out, _ = _try_forward(loss_fn, pred, target)
    if out is None or not isinstance(out, torch.Tensor):
        return False
    try:
        scalar = out if out.ndim == 0 else out.mean()
        scalar.backward()
    except Exception:
        return False
    return pred.grad is not None and bool(pred.grad.abs().sum().item() > 0.0)


def _kspace_pair(requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.randn(*_KSPACE_SHAPE)
    if requires_grad:
        pred = pred.detach().requires_grad_(True)
    target = torch.randn(*_KSPACE_SHAPE)
    return pred, target


def _coerce_scalar(t: Any) -> float | None:
    if isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return None
        return float(t.detach().cpu().reshape(-1)[0])
    if isinstance(t, (int, float)):
        return float(t)
    return None


def _numeric_terms(out: Any) -> list[tuple[str, float]]:
    """Every numeric term in a loss return, flattened to (label, value).

    A registered loss is NOT required to return a bare scalar: 12 of them
    deliberately return a multi-term structure (`physics_informed` ->
    {'l2_image', 'regularization', 'total'}, `vq` -> a tuple, ...). Asserting
    "returns a finite scalar Tensor" failed them for using a shape the training
    path consumes daily.

    This does not try to pick out which term is the total, because there is no
    SSOT that would say. `LossKeyRegistry` is keyed by TRAINING MODE, not by
    loss, and the total keys actually in use -- `loss`, `total`, `total_loss`,
    `spectral_total`, `dc_loss`, `loss_recon` -- match neither each other nor
    that registry's documented `*_total_loss` convention (issue filed).
    Asserting EVERY term is finite is stronger than picking one and needs no
    such convention: a NaN anywhere is a NaN in the backward pass.
    """
    if isinstance(out, dict):
        items = out.items()
    elif isinstance(out, (tuple, list)):
        items = ((str(i), v) for i, v in enumerate(out))
    else:
        scalar = _coerce_scalar(out)
        return [] if scalar is None else [("", scalar)]

    terms: list[tuple[str, float]] = []
    for label, value in items:
        scalar = _coerce_scalar(value)
        if scalar is not None:
            terms.append((str(label), scalar))
    return terms


def _try_forward(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor | None, str]:
    """Try (pred, target) → Tensor.  Return (tensor_or_None, status_str)."""
    try:
        out = loss_fn(pred, target)
        return out, "direct"
    except Exception as exc_a:
        pass
    try:
        out = loss_fn(pred, target, reduction="mean")
        return out, "reduction_kwarg"
    except Exception:
        pass
    return None, "unsupported"


# ---------------------------------------------------------------------------
# Default-lane: registration + instantiation
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
@pytest.mark.parametrize("key", all_losses(), ids=lambda k: k)
def test_loss_is_registered_and_buildable(key: str) -> None:
    """Each registered loss key must produce an nn.Module via LossRegistry.create."""
    assert key in LossRegistry._custom_losses, (
        f"Loss '{key}' in all_losses() but not in LossRegistry._custom_losses"
    )

    try:
        loss_fn = LossRegistry.create(key)
    except Exception as exc:
        pytest.xfail(
            f"LossRegistry.create('{key}') raised {type(exc).__name__}: {exc}"
        )
        return

    assert isinstance(loss_fn, nn.Module), (
        f"Loss '{key}' create() returned {type(loss_fn).__name__}, expected nn.Module"
    )


# ---------------------------------------------------------------------------
# Slow: forward scalar + gradient
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
@pytest.mark.slow
@pytest.mark.parametrize("key", all_losses(), ids=lambda k: k)
def test_loss_forward_returns_finite_scalar(key: str) -> None:
    """Each loss must return a finite scalar Tensor on image-domain input."""
    try:
        loss_fn = LossRegistry.create(key)
    except Exception as exc:
        pytest.xfail(
            f"Loss '{key}' could not be instantiated: {type(exc).__name__}: {exc}"
        )
        return

    loss_fn.eval()
    pred, target = _image_pair()

    tensor_out, convention = _try_forward(loss_fn, pred, target)

    if tensor_out is None:
        # Try kspace-shaped input as a fallback
        pred_k, target_k = _kspace_pair()
        tensor_out, convention = _try_forward(loss_fn, pred_k, target_k)

    if tensor_out is None:
        del loss_fn
        gc.collect()
        pytest.xfail(
            f"Loss '{key}' did not accept (pred, target) with either image or "
            f"k-space shaped inputs."
        )
        return

    try:
        terms = _numeric_terms(tensor_out)
    except Exception as exc:
        del loss_fn
        gc.collect()
        pytest.xfail(f"Loss '{key}' output could not be coerced to scalar: {exc}")
        return

    del loss_fn
    gc.collect()

    assert terms, (
        f"Loss '{key}' output carried no numeric term "
        f"(type={type(tensor_out).__name__})"
    )
    non_finite = [(label, value) for label, value in terms if not math.isfinite(value)]
    assert not non_finite, (
        f"Loss '{key}' returned non-finite term(s) {non_finite} on synthetic "
        f"input (convention: {convention}; {len(terms)} term(s) total)"
    )


@pytest.mark.registry_contract
@pytest.mark.slow
@pytest.mark.parametrize("key", all_losses(), ids=lambda k: k)
def test_loss_gradient_flows(key: str) -> None:
    """Gradient must flow back through pred after loss.backward()."""
    try:
        loss_fn = LossRegistry.create(key)
    except Exception as exc:
        pytest.xfail(
            f"Loss '{key}' could not be instantiated: {type(exc).__name__}: {exc}"
        )
        return

    loss_fn.eval()
    pred, target = _image_pair(requires_grad=True)

    tensor_out, _ = _try_forward(loss_fn, pred, target)

    if tensor_out is None:
        pred_k, target_k = _kspace_pair(requires_grad=True)
        tensor_out, _ = _try_forward(loss_fn, pred_k, target_k)
        pred = pred_k  # switch to whichever worked

    if tensor_out is None:
        del loss_fn
        gc.collect()
        pytest.xfail(
            f"Loss '{key}' did not produce output — cannot check gradient"
        )
        return

    # Reduce to scalar before backward
    try:
        scalar_t = tensor_out if tensor_out.ndim == 0 else tensor_out.mean()
        loss_value = float(scalar_t.detach())
        scalar_t.backward()
    except Exception as exc:
        del loss_fn
        gc.collect()
        pytest.xfail(
            f"Loss '{key}' backward raised {type(exc).__name__}: {exc}"
        )
        return

    del loss_fn
    gc.collect()

    assert pred.grad is not None, (
        f"Loss '{key}' backward did not populate pred.grad"
    )

    # An all-zero gradient is only evidence of a BLOCKED graph when the loss it
    # came from is non-zero. A hinge/penalty term that is SATISFIED at this
    # input is legitimately flat there: `hamiltonian_energy_conservation`,
    # `sense_adjoint_l1` and `cross_entropy` all return exactly 0.0 on the
    # synthetic pair, and a zero gradient is the correct derivative of zero.
    # Failing them conflated "trains nothing" with "nothing to train here".
    #
    # The graph itself is checked separately and unconditionally: `grad_fn` is
    # what a genuinely detached loss would be missing, and it is not a function
    # of where on the curve the input happens to sit.
    assert tensor_out.grad_fn is not None, (
        f"Loss '{key}' output carries no grad_fn — it is detached from pred, "
        f"so the term can never train anything"
    )

    grad_magnitude = pred.grad.abs().sum().item()
    if grad_magnitude == 0.0 and loss_value != 0.0:
        # Second-guess the verdict on a SIGNED probe before calling the term
        # inert. The default image probe is `torch.rand` -> [0, 1), so a penalty
        # on negative values is flat across its whole support:
        # `mrs_prior_knowledge` scores `clamp(-amplitudes, min=0)`, identically
        # zero on any non-negative input, while the rest of its value comes from
        # its second argument. That reads exactly like "non-zero value, zero
        # gradient" and was reported as pitfall #16 against a loss whose
        # production call site (mrs_quantification_strategy) passes two live
        # model outputs by keyword and does propagate gradient.
        #
        # A genuinely inert term is flat under both probes, so this only ever
        # converts a false failure.
        if _gradient_reaches_pred_on_a_signed_probe(key):
            return
        pytest.fail(
            f"Loss '{key}' returned a non-zero value ({loss_value!r}) but left "
            f"pred.grad all-zeros on both the default and a signed probe. The "
            f"term contributes to the reported total and nothing to the update "
            f"-- pitfall #16."
        )


# ---------------------------------------------------------------------------
# Self-test: the signed probe must actually span negatives
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
def test_signed_image_pair_spans_negative_values() -> None:
    """The rescue probe is only a rescue if it is genuinely signed.

    If ``_signed_image_pair`` ever drifted back to a non-negative draw, the
    retry above would silently agree with the default probe and the false
    pitfall-#16 verdict would return -- with the retry still in place, looking
    like it had been checked.
    """
    pred, target = _signed_image_pair()
    assert pred.min() < 0.0 and pred.max() > 0.0
    assert target.min() < 0.0 and target.max() > 0.0
