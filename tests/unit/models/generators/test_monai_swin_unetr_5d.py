"""Regression: ``MonaiSwinUNETR.forward`` accepts 5-D TorchIO tensors.

The 2026-05-10 smoke run flagged ``baseline_swin_unetr_3d`` with
``ValueError: too many values to unpack (expected 4)`` at
:py:meth:`mriforge.models.generators.reconstruction_variants.MonaiSwinUNETR.forward`
because the forward did ``B, C, H, W = x.shape`` on TorchIO's 5-D
``[B, C, H, W, D]`` layout. The fix added two paths:

* ``D == 1`` → squeeze the trailing singleton (matches the
  ``cs_mno_operator`` convention; the common 2-D-sliced case).
* ``D > 1`` → fold depth into batch, run the 2-D backbone per slice,
  unfold the output. ``baseline_swin_unetr_3d`` declares
  ``patch_size: [128, 128, 16]`` so this is the relevant case.

The class is named "SwinUNETR" but its internals are 2-D Conv2d /
ConvTranspose2d; this fix makes the 2-D core usable on volumetric
patches by slicing.  A true 3-D baseline would replace those layers
with Conv3d / ConvTranspose3d — out of scope for this regression.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import mriforge.models.generators  # noqa: F401, E402  triggers @register_model
from mriforge.models.generators.reconstruction_variants import MonaiSwinUNETR  # noqa: E402


def _tiny_model() -> MonaiSwinUNETR:
    # Use ``out_channels == dim // 2 == 4`` to side-step a pre-existing
    # channel mismatch in the model's residual path
    # (``dec [B, out_channels] + skip [B, dim//2]`` broadcasts when
    # those don't match).  That mismatch is a separate bug; this
    # regression pins the 5-D handling alone.
    return MonaiSwinUNETR(in_channels=4, out_channels=4, dim=8, depth=1)


def test_forward_accepts_4d_input() -> None:
    """The 4-D path still works after the 5-D extension."""
    model = _tiny_model().eval()
    x = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4, 32, 32)


def test_forward_squeezes_trailing_singleton_depth() -> None:
    """``D == 1`` (TorchIO 2-D-sliced) path: trailing singleton stripped."""
    model = _tiny_model().eval()
    x = torch.randn(2, 4, 32, 32, 1)
    with torch.no_grad():
        y = model(x)
    # Output stays 4-D after the squeeze — same convention as cs_mno_operator.
    assert y.shape == (2, 4, 32, 32)


def test_forward_folds_volumetric_depth_into_batch() -> None:
    """Regression for ``baseline_swin_unetr_3d``: ``patch_size: [H, W, 16]``.

    ``[B=1, C, 32, 32, D=4]`` must round-trip through the 2-D backbone
    by folding D into batch and unfolding back, returning 5-D output
    with depth preserved.
    """
    model = _tiny_model().eval()
    x = torch.randn(1, 4, 32, 32, 4)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 32, 32, 4), (
        "5-D input must produce 5-D output with depth preserved"
    )


def test_forward_5d_per_slice_consistency() -> None:
    """A 5-D forward must match the concatenation of 4-D per-slice forwards.

    This pins the "fold depth into batch" semantic — running the model
    on a volume slice-by-slice must produce the same answer as running
    each slice independently.  Catches accidental cross-slice leakage
    (e.g. if BatchNorm running stats picked up the per-slice batch).
    """
    model = _tiny_model().eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 32, 3)
    with torch.no_grad():
        y_vol = model(x)
        # Run each slice independently and stack.
        slices = [
            model(x[..., d]) for d in range(3)
        ]
        y_per_slice = torch.stack(slices, dim=-1)
    # Allow numerical drift (interp + transformer) but require structural
    # equality, not bitwise.
    assert torch.allclose(y_vol, y_per_slice, atol=1e-4, rtol=1e-4), (
        "5-D fold-unfold result diverges from per-slice 4-D forward — "
        "cross-slice context has leaked into the 2-D backbone."
    )
