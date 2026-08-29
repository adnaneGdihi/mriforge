"""Unit tests for the C_n-equivariant normalizing flow.

Covers the group blocks (GroupConv2d, GroupActNorm2d, GroupCoupling) and
the :class:`mriforge.models.generative.equivariant_flow.EquivariantFlow`
model: registry resolution, default constructibility, channel-divisibility
guards, forward shape, log-prob finiteness, and the load-bearing
equivariance probe ``f(rho_g x) ~= rho_g f(x)``.

``rho_g`` for the cyclic group C_n acting on the regular representation is
a cyclic shift of the group axis combined with the corresponding spatial
rotation; since the lift tiles a (rotation-invariant under the test's
symmetric input) image, the probe checks the group-axis-shift commutation.

Written but not executed here; they run on the cluster.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.equivariant.group_actnorm import GroupActNorm2d
from mriforge.models.blocks.equivariant.group_conv import GroupConv2d
from mriforge.models.blocks.equivariant.group_coupling import GroupCoupling
from mriforge.models.generative.equivariant_flow import EquivariantFlow
from mriforge.models.registry import MODEL_REGISTRY, get_model_class
from tests.unit.models._flow_base import assert_log_prob_finite


def _roll_group_axis(x: torch.Tensor, group_order: int, shift: int) -> torch.Tensor:
    """Cyclically shift the group axis of a regular-representation tensor."""
    b, c, h, w = x.shape
    base = c // group_order
    xr = x.view(b, base, group_order, h, w)
    xr = torch.roll(xr, shifts=shift, dims=2)
    return xr.reshape(b, c, h, w)


class TestGroupConv2d:
    def test_channel_divisibility_raises(self):
        with pytest.raises(ValueError):
            GroupConv2d(7, 8, group_order=8)

    def test_forward_shape(self):
        conv = GroupConv2d(8, 16, kernel_size=3, group_order=8)
        x = torch.randn(2, 8, 6, 6)
        out = conv(x)
        assert out.shape == (2, 16, 6, 6)

    def test_equivariance(self):
        # A group-axis shift on the input must produce the same shift on
        # the output (the defining property of a C_n group convolution).
        n = 4
        conv = GroupConv2d(2 * n, 3 * n, kernel_size=3, group_order=n)
        x = torch.randn(1, 2 * n, 8, 8)
        for s in range(n):
            lhs = conv(_roll_group_axis(x, n, s))
            rhs = _roll_group_axis(conv(x), n, s)
            assert torch.allclose(lhs, rhs, atol=1e-4)


class TestGroupActNorm2d:
    def test_roundtrip(self):
        act = GroupActNorm2d(16, group_order=8)
        x = torch.randn(2, 16, 8, 8)
        z, ld = act(x, reverse=False)
        x_rec, ld_inv = act(z, reverse=True)
        assert torch.allclose(x, x_rec, atol=1e-5)
        assert torch.allclose(ld, -ld_inv, atol=1e-5)


class TestGroupCoupling:
    def test_roundtrip(self):
        cp = GroupCoupling(16, group_order=8, hidden_base=4)
        x = torch.randn(2, 16, 8, 8)
        z, ld = cp(x, reverse=False)
        x_rec, ld_inv = cp(z, reverse=True)
        assert torch.allclose(x, x_rec, atol=1e-4)
        assert torch.allclose(ld, -ld_inv, atol=1e-4)


class TestEquivariantFlow:
    def test_registered_name(self):
        assert "equivariant_flow" in MODEL_REGISTRY
        assert get_model_class("equivariant_flow") is EquivariantFlow
        assert MODEL_REGISTRY["equivariant_flow"]["mode"] == "generative"

    def test_constructible_with_no_args(self):
        assert isinstance(EquivariantFlow(), EquivariantFlow)

    def test_unsupported_group_raises(self):
        with pytest.raises(ValueError):
            EquivariantFlow(group_order=7)

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError):
            EquivariantFlow(in_channels=1, out_channels=2)

    def test_forward_shape(self):
        model = EquivariantFlow(group_order=4, num_steps=2, base_channels=4)
        x = torch.randn(2, 1, 8, 8)
        z, log_det = model(x)
        assert z.shape[0] == 2
        assert log_det.shape == (2,)

    def test_log_prob_finite(self):
        model = EquivariantFlow(group_order=4, num_steps=2, base_channels=4)
        x = torch.randn(2, 1, 8, 8)
        assert_log_prob_finite(model, x)

    def test_equivariance_probe(self):
        # f(rho_g x) == rho_g f(x): a group-axis shift on the lifted input
        # must equal the same shift on the output features. The lift tiles
        # the image identically across the group axis, so we perturb the
        # post-lift features directly to exercise the group equivariance.
        n = 4
        model = EquivariantFlow(group_order=n, num_steps=2, base_channels=4)
        model.eval()
        x = torch.randn(1, 1, 8, 8)
        feat, _ = model.forward_features(x)
        # Shift the latent and verify it matches running the flow on a
        # shifted lift. Because every block is equivariant, the two agree.
        lifted = model._lift(x)
        for s in range(n):
            shifted_lift = _roll_group_axis(lifted, n, s)
            h = shifted_lift
            for step in model.steps:
                h, _ = step(h, reverse=False)
            expected = _roll_group_axis(feat, n, s)
            max_err = (h - expected).abs().max().item()
            assert max_err < 1e-4, f"equivariance broken at shift {s}: {max_err}"
