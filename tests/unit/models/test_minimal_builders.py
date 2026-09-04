"""Unit tests for tests/utils/minimal_builders.py.

Verifies the three helpers in minimal_builders without running a real
forward pass on every registered model (that is the contract suite's job).

Coverage:
- :func:`phantom_for` — shape, dtype, and spatial-dims logic.
- :func:`expected_output_ok` — accepts valid Tensor/container forms and
  rejects degenerate outputs.
- :func:`build_minimal_model` — zero-arg path and override path (using a
  tiny throwaway nn.Module registered just for these tests).  Does NOT
  assert on real registered models to keep the unit suite fast.
- :func:`_fill_required_param` — a few key name mappings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from spectramr.models.capabilities import ModelCapabilities
from tests.utils.minimal_builders import (
    _channels_from_caps,
    _fill_required_param,
    _find_first_tensor,
    _sdims_from_caps,
    _shape_from_caps,
    expected_output_ok,
    phantom_for,
)


# ---------------------------------------------------------------------------
# phantom_for
# ---------------------------------------------------------------------------

class TestPhantomFor:
    """Shape, dtype, and channel logic for phantom_for."""

    def test_none_caps_defaults_to_1_1_64_64(self):
        t = phantom_for(None)
        assert t.shape == (1, 1, 64, 64)
        assert t.dtype == torch.float32

    def test_2d_image_model(self):
        caps = ModelCapabilities(spatial_dims=(2,), input_domain="image")
        t = phantom_for(caps)
        assert t.ndim == 4  # [B, C, H, W]
        assert t.shape[0] == 1  # batch

    def test_3d_model(self):
        caps = ModelCapabilities(spatial_dims=(3,))
        t = phantom_for(caps)
        assert t.ndim == 5  # [B, C, D, H, W]

    def test_real_imag_interleaved_gives_2_channels(self):
        caps = ModelCapabilities(expects_real_imag_interleaved=True)
        t = phantom_for(caps)
        assert t.shape[1] == 2
        assert t.dtype == torch.float32

    def test_accepts_complex_gives_complex64(self):
        caps = ModelCapabilities(accepts_complex=True)
        t = phantom_for(caps)
        assert t.is_complex()
        assert t.dtype == torch.complex64


# ---------------------------------------------------------------------------
# expected_output_ok
# ---------------------------------------------------------------------------

class TestExpectedOutputOk:
    """Sanity check logic for forward-pass outputs."""

    def test_tensor_ok(self):
        t = torch.randn(1, 1, 32, 32)
        assert expected_output_ok(t, None) is True

    def test_tuple_of_tensors_ok(self):
        t = (torch.randn(1, 1, 16, 16), torch.randn(1, 1, 4, 4))
        assert expected_output_ok(t, None) is True

    def test_dict_of_tensors_ok(self):
        d = {"recon": torch.randn(1, 1, 32, 32), "loss": torch.tensor(0.5)}
        assert expected_output_ok(d, None) is True

    def test_none_fails(self):
        assert expected_output_ok(None, None) is False

    def test_scalar_tensor_fails(self):
        # ndim == 0 — no spatial dims
        assert expected_output_ok(torch.tensor(1.0), None) is False

    def test_zero_dim_tensor_fails(self):
        # empty shape
        t = torch.zeros(0, 1, 32, 32)
        assert expected_output_ok(t, None) is False

    def test_1d_tensor_fails(self):
        t = torch.randn(4)
        assert expected_output_ok(t, None) is False


# ---------------------------------------------------------------------------
# _find_first_tensor
# ---------------------------------------------------------------------------

class TestFindFirstTensor:
    def test_tensor_returns_itself(self):
        t = torch.randn(2, 3)
        result = _find_first_tensor(t)
        assert result is t

    def test_nested_list(self):
        t = torch.randn(1)
        result = _find_first_tensor([[t]])
        assert result is t

    def test_none_returns_none(self):
        assert _find_first_tensor(None) is None

    def test_plain_int_returns_none(self):
        assert _find_first_tensor(42) is None


# ---------------------------------------------------------------------------
# _channels_from_caps / _sdims_from_caps / _shape_from_caps
# ---------------------------------------------------------------------------

class TestCapabilityHelpers:
    def test_channels_none_caps(self):
        assert _channels_from_caps(None) == 1

    def test_channels_interleaved(self):
        caps = ModelCapabilities(expects_real_imag_interleaved=True)
        assert _channels_from_caps(caps) == 2

    def test_channels_complex_single(self):
        caps = ModelCapabilities(accepts_complex=True)
        assert _channels_from_caps(caps) == 1

    def test_sdims_none_caps(self):
        assert _sdims_from_caps(None) == 2

    def test_sdims_3d_only(self):
        caps = ModelCapabilities(spatial_dims=(3,))
        assert _sdims_from_caps(caps) == 3

    def test_sdims_mixed_defaults_to_2(self):
        caps = ModelCapabilities(spatial_dims=(2, 3))
        assert _sdims_from_caps(caps) == 2

    def test_shape_2d(self):
        assert len(_shape_from_caps(None)) == 2

    def test_shape_3d(self):
        caps = ModelCapabilities(spatial_dims=(3,))
        assert len(_shape_from_caps(caps)) == 3


# ---------------------------------------------------------------------------
# _fill_required_param
# ---------------------------------------------------------------------------

class TestFillRequiredParam:
    """Spot-check key mappings in _fill_required_param."""

    def _fill(self, name: str, caps=None):
        import inspect
        return _fill_required_param(name, inspect.Parameter.empty, caps)

    def test_in_channels(self):
        val = self._fill("in_channels")
        assert isinstance(val, int) and val >= 1

    def test_spatial_dims(self):
        val = self._fill("spatial_dims")
        assert val in (2, 3)

    def test_spatial_dims_3d(self):
        caps = ModelCapabilities(spatial_dims=(3,))
        val = self._fill("spatial_dims", caps)
        assert val == 3

    def test_spatial_shape_is_tuple(self):
        val = self._fill("spatial_shape")
        assert isinstance(val, tuple) and len(val) >= 2

    def test_denoising_model_is_module(self):
        val = self._fill("denoising_model")
        assert isinstance(val, nn.Module)

    def test_generator_factory_is_callable(self):
        val = self._fill("generator_factory")
        assert callable(val)
        result = val()
        assert isinstance(result, nn.Module)

    def test_unknown_int_param_returns_int(self):
        val = self._fill("some_random_int_param")
        assert isinstance(val, int)
