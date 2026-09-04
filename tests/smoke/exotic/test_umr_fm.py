"""
Unit tests for Universal MRI Foundation Model (UMR-FM).
"""

import pytest
import torch

from spectramr.models.foundation.umr_fm import (
    KSpaceMIMLoss,
    UniversalFoundationModel,
    create_umr_fm,
)
from tests.utils.optional_backends import requires_cuda_for_mamba


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_kspace(device):
    # [B, C, H, W]
    # Small size for speed
    return torch.randn(2, 2, 32, 32).to(device)


def test_mim_loss_masking(device, input_kspace):
    """Verify mask generation and ACS preservation."""
    loss_fn = KSpaceMIMLoss(mask_ratio=0.5, acs_radius=4)
    masked_x, mask = loss_fn(input_kspace)

    assert masked_x.shape == input_kspace.shape
    assert mask.shape == (2, 1, 32, 1)  # [B, 1, H, 1]

    # Check ACS region (center 4 lines) is NOT masked (mask=0)
    center = 16
    acs_region_mask = mask[:, :, center - 4 : center + 4, :]
    assert (acs_region_mask == 0).all(), "ACS region must not be masked"

    # Check some outside ACS region is masked (statistically likely)
    # This is probing randomness, but with 0.5 ratio, sum should be > 0
    outer_mask = torch.cat([mask[:, :, : center - 4, :], mask[:, :, center + 4 :, :]], dim=2)
    assert outer_mask.sum() > 0, "Outer region should have some masking"


@requires_cuda_for_mamba
def test_umr_fm_forward(device, input_kspace):
    """Verify UMR-FM forward pass and loss computation."""
    model = UniversalFoundationModel(
        in_chans=2, embed_dim=16, depths=[2], num_heads=[2], window_size=4
    ).to(device)

    model.train()
    loss, x_rec, mask = model(input_kspace)

    assert loss.dim() == 0  # scalar
    assert x_rec.shape == input_kspace.shape
    assert mask.shape[0] == input_kspace.shape[0]

    # Verify backward
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            break
    else:
        pytest.fail("Gradients not flowing")


@requires_cuda_for_mamba
def test_umr_fm_inference(device, input_kspace):
    """Verify inference mode (no loss returned)."""
    model = UniversalFoundationModel(
        in_chans=2, embed_dim=16, depths=[2], num_heads=[2], window_size=4
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_kspace)
        # Should return (x_rec, mask) tuple
        assert isinstance(out, tuple)
        assert len(out) == 2
        x_rec, mask = out
        assert x_rec.shape == input_kspace.shape


def test_create_umr_fm_pretrained_raises():
    """pretrained=True must raise (pitfall #15: unread knob is not a silent no-op)."""
    with pytest.raises(NotImplementedError):
        create_umr_fm(
            pretrained=True,
            in_chans=2,
            embed_dim=16,
            depths=[2],
            num_heads=[2],
            window_size=4,
        )


def test_create_umr_fm_default_builds():
    """Default path (pretrained=False) returns a random-init foundation model."""
    model = create_umr_fm(
        pretrained=False,
        in_chans=2,
        embed_dim=16,
        depths=[2],
        num_heads=[2],
        window_size=4,
    )
    assert isinstance(model, UniversalFoundationModel)
