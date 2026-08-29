"""Unit tests for the divergence-free flow-matching velocity field.

Covers the streamfunction/curl blocks and the
:class:`mriforge.models.generative.divergence_free_flow.DivergenceFreeFlow`
model: registry resolution, default constructibility, ndim/channel guards,
forward shape, and the load-bearing divergence probe ``|div v|_inf`` near
zero (machine precision on the staggered grid up to boundary padding).

Written but not executed here; they run on the cluster.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.streamfunction import (
    CurlOf3DVectorPotential,
    StreamfunctionTo2DVelocity,
)
from mriforge.models.generative.divergence_free_flow import DivergenceFreeFlow
from mriforge.models.registry import MODEL_REGISTRY, get_model_class


def _divergence_2d(v: torch.Tensor) -> torch.Tensor:
    """Discrete divergence d v_x/dx + d v_y/dy in the interior."""
    v_x, v_y = v[:, 0:1], v[:, 1:2]
    dvx_dx = v_x[:, :, :, 1:] - v_x[:, :, :, :-1]
    dvy_dy = v_y[:, :, 1:, :] - v_y[:, :, :-1, :]
    # Align interior region.
    return dvx_dx[:, :, 1:, :] + dvy_dy[:, :, :, 1:]


class TestStreamfunctionTo2DVelocity:
    def test_shape(self):
        block = StreamfunctionTo2DVelocity()
        psi = torch.randn(2, 1, 16, 16)
        v = block(psi)
        assert v.shape == (2, 2, 16, 16)

    def test_wrong_channels_raise(self):
        with pytest.raises(ValueError):
            StreamfunctionTo2DVelocity()(torch.randn(2, 2, 8, 8))

    def test_divergence_near_zero(self):
        block = StreamfunctionTo2DVelocity()
        psi = torch.randn(1, 1, 32, 32)
        v = block(psi)
        div = _divergence_2d(v)
        assert div.abs().max().item() < 1e-5


class TestCurlOf3DVectorPotential:
    def test_shape(self):
        block = CurlOf3DVectorPotential()
        psi = torch.randn(2, 3, 8, 8, 8)
        v = block(psi)
        assert v.shape == (2, 3, 8, 8, 8)

    def test_wrong_channels_raise(self):
        with pytest.raises(ValueError):
            CurlOf3DVectorPotential()(torch.randn(2, 1, 8, 8, 8))


class TestDivergenceFreeFlow:
    def test_registered_name(self):
        assert "divergence_free_flow" in MODEL_REGISTRY
        assert get_model_class("divergence_free_flow") is DivergenceFreeFlow
        assert MODEL_REGISTRY["divergence_free_flow"]["mode"] == "flow_matching"
        caps = MODEL_REGISTRY["divergence_free_flow"]["capabilities"]
        assert caps.spatial_dims == (2, 3)

    def test_constructible_with_no_args(self):
        assert isinstance(DivergenceFreeFlow(), DivergenceFreeFlow)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError):
            DivergenceFreeFlow(ndim=4)

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError):
            DivergenceFreeFlow(ndim=2, out_channels=3)

    def test_forward_shape_2d(self):
        model = DivergenceFreeFlow(in_channels=2, out_channels=2, ndim=2, hidden_channels=16)
        x = torch.randn(2, 2, 16, 16)
        t = torch.rand(2)
        v = model(x, t)
        assert v.shape == (2, 2, 16, 16)
        assert torch.isfinite(v).all()

    def test_forward_shape_3d(self):
        model = DivergenceFreeFlow(in_channels=3, out_channels=3, ndim=3, hidden_channels=8)
        x = torch.randn(1, 3, 8, 8, 8)
        t = torch.rand(1)
        v = model(x, t)
        assert v.shape == (1, 3, 8, 8, 8)

    def test_output_divergence_near_zero(self):
        model = DivergenceFreeFlow(in_channels=2, out_channels=2, ndim=2, hidden_channels=16)
        x = torch.randn(1, 2, 32, 32)
        t = torch.rand(1)
        v = model(x, t)
        div = _divergence_2d(v)
        assert div.abs().max().item() < 1e-5
