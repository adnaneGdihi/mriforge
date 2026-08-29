"""Regression tests for mriforge.models.reconstruction.zerorf.

ZeroRFReconstructor advertises a 3D ``volume_shape`` (D, H, W), but its
``forward`` previously assumed ``D == 1`` and reshaped the reconstructed volume
into ``[1, 1, H, W]`` unconditionally. For any ``D > 1`` that raised a
``RuntimeError`` ("shape '[1,1,H,W]' is invalid for input of size D*H*W").

The fix selects the middle depth slice ``vol[D // 2]`` before reshaping, which
is byte-identical for ``D == 1`` (so the default path / trained-model numerics
are unchanged) and well-defined for ``D > 1``.
"""

import torch

from mriforge.models.reconstruction.zerorf import ZeroRFReconstructor


def _make_model(volume_shape):
    torch.manual_seed(0)
    return ZeroRFReconstructor(
        in_channels=2,
        out_channels=2,
        volume_shape=volume_shape,
        rank=4,
        latent_dim=8,
    )


def test_forward_3d_volume_shape_no_crash():
    """D > 1 must not raise (the pre-fix bug raised a RuntimeError)."""
    model = _make_model((4, 16, 16))
    x = torch.randn(2, 2, 16, 16)
    out = model(x)
    assert out.shape == (2, 2, 16, 16)


def test_forward_d1_unchanged():
    """The D == 1 default path still returns (B, out_channels, H, W)."""
    model = _make_model((1, 16, 16))
    x = torch.randn(3, 2, 16, 16)
    out = model(x)
    assert out.shape == (3, 2, 16, 16)


def test_middle_slice_selected_for_d_gt_1():
    """For D=4 the emitted slice corresponds to reconstruct_volume(...)[2]."""
    model = _make_model((4, 16, 16))
    model.eval()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        params = model.generator(model.z_code)
        vol = model.reconstruct_volume(params)  # [D, H, W]
        out = model(x)
    expected_slice = vol[vol.shape[0] // 2]  # D // 2 == 2 for D=4
    # out is (B, out_channels, H, W) broadcast from the single middle slice.
    assert torch.allclose(out[0, 0], expected_slice)


def test_middle_slice_equals_reshape_for_d1():
    """D == 1: vol[mid].reshape(...) is identical to the old vol.reshape(...)."""
    model = _make_model((1, 16, 16))
    model.eval()
    with torch.no_grad():
        params = model.generator(model.z_code)
        vol = model.reconstruct_volume(params)  # [1, H, W]
    old = vol.reshape(1, 1, vol.shape[-2], vol.shape[-1])
    new = vol[vol.shape[0] // 2].reshape(1, 1, vol.shape[-2], vol.shape[-1])
    assert torch.equal(old, new)
