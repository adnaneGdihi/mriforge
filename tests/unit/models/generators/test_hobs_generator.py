"""Unit tests for the HoBS (Holographic Bloch-Splatting) generator.

Pins the pitfall-#9 guard on ``sequence_type``: an unknown value must raise
instead of silently skipping the Bloch simulation (the old fallback returned
bare M0, so a mistyped sequence produced a physics-free density render that
still smoke-passed).
"""

import pytest
import torch

from mriforge.models.generators.hobs_generator import HoBSGenerator


def _tiny_model() -> HoBSGenerator:
    return HoBSGenerator(num_gaussians=8)


def test_spin_echo_forward_finite():
    model = _tiny_model()
    out = model(sequence_type="spin_echo")
    assert out.shape == (1, 8, 1)
    assert torch.isfinite(out).all()


def test_inversion_recovery_forward_finite():
    model = _tiny_model()
    out = model(sequence_type="inversion_recovery", ti=400.0)
    assert out.shape == (1, 8, 1)
    assert torch.isfinite(out).all()


def test_default_sequence_type_is_spin_echo():
    model = _tiny_model()
    default_out = model()
    explicit_out = model(sequence_type="spin_echo")
    assert torch.equal(default_out, explicit_out)


def test_unknown_sequence_type_raises():
    model = _tiny_model()
    with pytest.raises(ValueError, match=r"spin_echo.*inversion_recovery"):
        model(sequence_type="gradient_echo")


def test_biophysics_output_type_still_works():
    model = _tiny_model()
    maps = model(sequence_type="spin_echo", output_type="biophysics")
    assert maps.shape == (1, 8, 3)
    assert torch.isfinite(maps).all()
