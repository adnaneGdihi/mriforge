"""Regression tests for :func:`_unwrap_tensor_arg` (audit-2026-05-14 F2a).

Background
----------
Several generators (latent / cold diffusion, multi-head VAEs) return a
``dict`` like ``{"pred": tensor, "aux": ...}`` from ``forward``. Before
F2a, those dicts arrived at ``_complex_safe_mse`` / ``_complex_safe_l1``
which called ``.shape`` / ``.device`` on them and raised
``AttributeError: 'dict' object has no attribute 'shape'`` (the
"W6 / W3 dict-shape" warning class — 28 + N occurrences in the
2026-05-14 smoke log).

The fix introduced ``_unwrap_tensor_arg(name, value)`` which probes a
fixed list of common prediction keys (``pred`` / ``image`` / ``x0`` /
…) and falls back to the first tensor-valued entry. This test pins:

1. Each probed key resolves correctly.
2. Tuples / lists are unwrapped to their first element.
3. Dict-of-non-tensors raises a typed ``TypeError`` (CLAUDE.md #9
   fail-loud) — the orchestrator should not see a confusing
   AttributeError from inside MSE.
4. ``_complex_safe_mse`` and ``_complex_safe_l1`` accept dict inputs
   end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.computers.unified_diffusion_reconstruction import (
    _complex_safe_l1,
    _complex_safe_mse,
    _unwrap_tensor_arg,
)


def test_unwrap_passthrough_tensor() -> None:
    """A tensor passes through unchanged."""
    x = torch.randn(2, 3)
    assert _unwrap_tensor_arg("pred", x) is x


@pytest.mark.parametrize(
    "key",
    ["pred", "prediction", "predictions", "image", "image_out",
     "output", "out", "x", "x0", "x_0", "x_pred", "x_hat",
     "x_recon", "sample", "samples", "reconstruction", "denoised"],
)
def test_unwrap_known_prediction_keys(key: str) -> None:
    """Every documented prediction key resolves to its tensor entry."""
    x = torch.randn(2, 3)
    out = _unwrap_tensor_arg("pred", {key: x, "aux": "irrelevant"})
    assert out is x, f"key {key!r} did not unwrap correctly"


def test_unwrap_dict_unknown_key_falls_back_to_first_tensor() -> None:
    """Unknown keys still resolve via the tensor-fallback path."""
    x = torch.randn(2, 3)
    out = _unwrap_tensor_arg("pred", {"weird_name": x})
    assert out is x


def test_unwrap_dict_no_tensor_raises() -> None:
    """Dict-of-non-tensors raises ``TypeError`` (CLAUDE.md #9 fail-loud)."""
    with pytest.raises(TypeError, match="dict with no tensor entry"):
        _unwrap_tensor_arg("pred", {"a": 1, "b": "two"})


def test_unwrap_tuple() -> None:
    """Tuples / lists unwrap to their first element."""
    x = torch.randn(2, 3)
    assert _unwrap_tensor_arg("pred", (x, "aux")) is x
    assert _unwrap_tensor_arg("pred", [x, "aux"]) is x


def test_complex_safe_mse_handles_dict_input() -> None:
    """``_complex_safe_mse({'pred': t}, t)`` returns a scalar tensor."""
    t = torch.randn(2, 4)
    loss = _complex_safe_mse({"pred": t}, t)
    assert loss.dim() == 0, "MSE must return a scalar"
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
        "MSE between identical tensors must be ~0"
    )


def test_complex_safe_mse_handles_dict_on_both_sides() -> None:
    """Dict on both pred and target."""
    t = torch.randn(2, 4)
    loss = _complex_safe_mse({"pred": t}, {"target": t})
    assert loss.dim() == 0


def test_complex_safe_l1_handles_dict_input() -> None:
    """``_complex_safe_l1`` matches the MSE behaviour."""
    t = torch.randn(2, 4)
    loss = _complex_safe_l1({"image": t}, t)
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
