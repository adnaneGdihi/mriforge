"""Regression test for F2/E10 (smoke_audit_20260521) — CoilCombineTransform
must reinterpret even-channel real input as the canonical real-stacked
complex multi-coil layout instead of raising.

Root cause: 20 promoted experiments hit
``ValueError: [CoilCombine] input is real with 8 channels — cannot
disambiguate multi-contrast from interleaved coils`` at runtime
because their data builder pre-realstacks complex multi-coil data
into ``[2C, H, W, D]`` real tensors before the transform sees it.
The operator's choice of ``coil_processing_mode='rss_image'`` (or
``sense`` / ``rss``) is itself the disambiguation signal — they
wouldn't request a coil-combination method unless they meant
"treat this as multi-coil complex".

This test pins:

1. **Even-channel real input is accepted** and reinterpreted as
   complex multi-coil before the combine step.
2. **Odd-channel real input still raises** — the historical
   "multi-contrast vs interleaved coils" ambiguity only applies to
   odd channel counts.
3. **1- and 2-channel pass-through is preserved** (existing
   contract — magnitude or already-combined).
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from mriforge.data.transforms.kspace_coil_transforms import CoilCombineTransform


def _subject(tensor: torch.Tensor) -> tio.Subject:
    """Build a minimal torchio subject around a single image tensor."""
    return tio.Subject(image=tio.ScalarImage(tensor=tensor))


def test_even_channel_real_is_reinterpreted_as_complex_multicoil() -> None:
    """8-channel real input → reinterpret as 4 complex coils → combine.

    The 20 promoted experiments (exp_promoted_*, exp_cross_domain_matrix,
    exp_hypernet_universal, exp_inr_universal, exp_slice_profile_sr)
    all hit this path. Before F2, the transform raised; after F2, it
    proceeds with the requested combine method."""
    torch.manual_seed(0)
    real_stacked = torch.randn(8, 16, 16, 4, dtype=torch.float32)
    subject = _subject(real_stacked)

    transform = CoilCombineTransform(method="rss_image", keys=("image",))
    out = transform(subject)

    # rss_image returns ``(1, H, W, D)`` real magnitude.
    assert out["image"].data.shape == (1, 16, 16, 4), (
        f"rss_image must return (1, H, W, D); got {tuple(out['image'].data.shape)}"
    )
    assert out["image"].data.dtype == torch.float32, (
        "rss_image output must be real-valued magnitude"
    )
    assert torch.all(out["image"].data >= 0), (
        "rss_image output must be non-negative magnitude"
    )


def test_odd_channel_real_still_raises() -> None:
    """The original multi-contrast-vs-coil ambiguity argument still
    applies to ODD channel counts (genuinely cannot disambiguate).
    Pinned so a future over-eager generalization doesn't also paper
    over genuine misconfigs."""
    torch.manual_seed(0)
    odd = torch.randn(3, 16, 16, 4, dtype=torch.float32)
    subject = _subject(odd)

    transform = CoilCombineTransform(method="rss_image", keys=("image",))
    with pytest.raises(ValueError, match="odd channel count"):
        transform(subject)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_one_or_two_channel_passthrough_preserved(n_channels: int) -> None:
    """The pre-existing pass-through for 1- and 2-channel real input
    (magnitude / already-combined R+I) must still work after the
    even-channel generalization. Otherwise a magnitude-only YAML
    starts getting reinterpreted as a 0-coil or 1-coil complex tensor."""
    torch.manual_seed(0)
    data = torch.randn(n_channels, 16, 16, 4, dtype=torch.float32)
    subject = _subject(data)

    transform = CoilCombineTransform(method="rss_image", keys=("image",))
    out = transform(subject)

    # Pass-through: shape unchanged.
    assert out["image"].data.shape == (n_channels, 16, 16, 4), (
        f"{n_channels}-channel real input must pass through unchanged; "
        f"got {tuple(out['image'].data.shape)}"
    )


def test_complex_input_still_works() -> None:
    """Genuine complex multi-coil input was the original happy path —
    must keep working after the even-channel-real generalization."""
    torch.manual_seed(0)
    complex_data = torch.randn(4, 16, 16, 4, dtype=torch.cfloat)
    subject = _subject(complex_data)

    transform = CoilCombineTransform(method="rss_image", keys=("image",))
    out = transform(subject)

    assert out["image"].data.shape == (1, 16, 16, 4)
    assert out["image"].data.dtype == torch.float32  # magnitude is real
