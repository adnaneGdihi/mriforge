"""Tests for FieldVelocityUNet (MICCAI MRIxFields2026 field-flow velocity field)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.field_velocity_unet import FieldVelocityUNet
from mriforge.models.registry import MODEL_REGISTRY


def test_registered() -> None:
    assert "field_velocity_unet" in MODEL_REGISTRY
    assert MODEL_REGISTRY["field_velocity_unet"]["mode"] == "field_flow"


def test_forward_shape() -> None:
    m = FieldVelocityUNet(width=16)
    v = m(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 3.0]))
    assert v.shape == (2, 1, 16, 16)


def test_field_path_is_wired() -> None:
    # FiLM is identity-init, so the velocity is field-agnostic until trained.
    # Perturb FiLM weights to confirm the field actually drives the velocity
    # (anti-facade: tau is consumed, not inert).
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=16).eval()
    x = torch.rand(1, 1, 16, 16)
    for p in m.film.parameters():
        p.data += 0.1
    v_lo = m(x, field_strength=torch.tensor([0.1]))
    v_hi = m(x, field_strength=torch.tensor([7.0]))
    assert not torch.allclose(v_lo, v_hi)


def test_complex_raises() -> None:
    m = FieldVelocityUNet(width=8)
    with pytest.raises(ValueError):
        m(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_non_magnitude_channels_raise() -> None:
    with pytest.raises(ValueError):
        FieldVelocityUNet(in_channels=2, out_channels=2)


def test_contrast_conditioning_off_by_default() -> None:
    # Default: contrast-blind (sequence slot is zeros). A `contrast_id` kwarg is
    # accepted (forward is kwarg-tolerant) but has NO effect on the output.
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=16).eval()
    assert m.use_contrast_conditioning is False
    for p in m.film.parameters():
        p.data += 0.1
    x = torch.rand(1, 1, 16, 16)
    b = torch.tensor([3.0])
    v0 = m(x, field_strength=b, contrast_id=torch.tensor([0]))
    v2 = m(x, field_strength=b, contrast_id=torch.tensor([2]))
    assert torch.allclose(v0, v2)  # contrast ignored when the flag is off


def test_contrast_conditioning_sets_sequence_dim() -> None:
    m = FieldVelocityUNet(width=16, use_contrast_conditioning=True, num_contrasts=3)
    assert m.use_contrast_conditioning is True
    assert m.film.sequence_dim == 3


def test_contrast_conditioning_changes_output() -> None:
    # Mechanism-fires (pitfall #16): with conditioning on and FiLM perturbed off
    # its identity init, a different contrast id must change the velocity.
    torch.manual_seed(0)
    m = FieldVelocityUNet(
        width=16, use_contrast_conditioning=True, num_contrasts=3
    ).eval()
    for p in m.film.parameters():
        p.data += 0.1
    x = torch.rand(1, 1, 16, 16)
    b = torch.tensor([3.0])
    v_t1 = m(x, field_strength=b, contrast_id=torch.tensor([0]))
    v_flair = m(x, field_strength=b, contrast_id=torch.tensor([2]))
    assert not torch.allclose(v_t1, v_flair)


def test_contrast_conditioning_requires_contrast_id() -> None:
    # #15: an advertised knob must be wired. Flag on but no contrast_id in the
    # batch is a silent no-op hazard -> raise, never fall back to zeros.
    m = FieldVelocityUNet(width=8, use_contrast_conditioning=True, num_contrasts=3)
    with pytest.raises(ValueError, match="contrast_id"):
        m(torch.rand(1, 1, 8, 8), field_strength=torch.tensor([1.0]))


def test_contrast_id_out_of_range_raises() -> None:
    m = FieldVelocityUNet(width=8, use_contrast_conditioning=True, num_contrasts=3)
    with pytest.raises((ValueError, RuntimeError)):
        m(
            torch.rand(1, 1, 8, 8),
            field_strength=torch.tensor([1.0]),
            contrast_id=torch.tensor([7]),  # >= num_contrasts
        )


def test_off_mode_uses_scalar_placeholder_sequence() -> None:
    # After the 2026-07 refactor onto the shared contrast_conditioning helper,
    # off-mode keeps the single zero-placeholder FiLM slot
    # (contrast_sequence_dim(0, num_contrasts, enabled=False) == 1) — the field-only
    # path is byte-compatible with the pre-refactor architecture.
    m = FieldVelocityUNet(width=16)
    assert m.use_contrast_conditioning is False
    assert m.film.sequence_dim == 1
