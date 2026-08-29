"""Unit tests for the Glow multi-scale normalizing flow.

Covers the flow blocks (ActNorm, invertible 1x1 conv, affine coupling,
squeeze/split) and the :class:`mriforge.models.generative.glow.Glow` model:
registry resolution, default constructibility, forward shape, the
invertibility round-trip (< 1e-4), log-prob finiteness and gradient flow.

Written but not executed here; they run on the cluster.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.flow.actnorm import ActNorm2d
from mriforge.models.blocks.flow.affine_coupling import AffineCoupling
from mriforge.models.blocks.flow.invertible_1x1_conv import InvertibleConv1x1LU
from mriforge.models.blocks.flow.squeeze import Split, Squeeze2x
from mriforge.models.generative.glow import Glow
from mriforge.models.registry import MODEL_REGISTRY, get_model_class
from tests.unit.models._flow_base import (
    assert_gradient_flows,
    assert_log_prob_finite,
    assert_roundtrip,
)


class TestActNorm2d:
    def test_roundtrip(self):
        act = ActNorm2d(4)
        x = torch.randn(2, 4, 8, 8)
        z, ld = act(x, reverse=False)
        x_rec, ld_inv = act(z, reverse=True)
        assert torch.allclose(x, x_rec, atol=1e-5)
        assert torch.allclose(ld, -ld_inv, atol=1e-5)

    def test_data_dependent_init_zero_mean_unit_var(self):
        act = ActNorm2d(3)
        x = 5.0 + 2.0 * torch.randn(16, 3, 8, 8)
        z, _ = act(x)
        assert z.mean().abs() < 0.5
        assert abs(z.std().item() - 1.0) < 0.5


class TestInvertibleConv1x1LU:
    def test_roundtrip(self):
        conv = InvertibleConv1x1LU(6)
        x = torch.randn(2, 6, 8, 8)
        z, ld = conv(x, reverse=False)
        x_rec, ld_inv = conv(z, reverse=True)
        assert torch.allclose(x, x_rec, atol=1e-3)
        assert torch.allclose(ld, -ld_inv, atol=1e-4)


class TestAffineCoupling:
    def test_roundtrip(self):
        cp = AffineCoupling(8, hidden_channels=16)
        x = torch.randn(2, 8, 8, 8)
        z, ld = cp(x, reverse=False)
        x_rec, ld_inv = cp(z, reverse=True)
        assert torch.allclose(x, x_rec, atol=1e-4)
        assert torch.allclose(ld, -ld_inv, atol=1e-4)

    def test_odd_channels_raise(self):
        with pytest.raises(ValueError):
            AffineCoupling(7)

    def test_unknown_coupling_raises(self):
        with pytest.raises(ValueError):
            AffineCoupling(8, coupling_type="quadratic")


class TestSqueezeSplit:
    def test_squeeze_roundtrip(self):
        sq = Squeeze2x()
        x = torch.randn(2, 3, 8, 8)
        y = sq(x, reverse=False)
        assert y.shape == (2, 12, 4, 4)
        x_rec = sq(y, reverse=True)
        assert torch.allclose(x, x_rec)

    def test_split_roundtrip(self):
        sp = Split()
        x = torch.randn(2, 8, 4, 4)
        kept, emitted = sp(x)
        assert kept.shape == (2, 4, 4, 4)
        rec = sp.inverse(kept, emitted)
        assert torch.allclose(x, rec)


class TestGlow:
    def test_registered_name(self):
        assert "glow" in MODEL_REGISTRY
        assert get_model_class("glow") is Glow
        assert MODEL_REGISTRY["glow"]["mode"] == "generative"
        caps = MODEL_REGISTRY["glow"]["capabilities"]
        assert caps.input_domain == "image"
        assert caps.accepts_complex is False

    def test_constructible_with_no_args(self):
        assert isinstance(Glow(), Glow)

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError):
            Glow(in_channels=1, out_channels=2)

    def test_forward_shape_and_logdet(self):
        model = Glow(in_channels=1, num_scales=2, num_steps=2, hidden_channels=16)
        x = torch.randn(2, 1, 16, 16)
        z, log_det = model(x)
        assert z.shape[0] == 2
        # bijection preserves total element count.
        assert z.shape[1] == 1 * 16 * 16
        assert log_det.shape == (2,)

    def test_roundtrip(self):
        model = Glow(in_channels=1, num_scales=2, num_steps=2, hidden_channels=16)
        x = torch.randn(2, 1, 16, 16)
        assert_roundtrip(model, x, atol=1e-3)

    def test_log_prob_finite(self):
        model = Glow(in_channels=1, num_scales=2, num_steps=2, hidden_channels=16)
        x = torch.randn(2, 1, 16, 16)
        assert_log_prob_finite(model, x)

    def test_gradient_flows(self):
        model = Glow(in_channels=1, num_scales=2, num_steps=2, hidden_channels=16)
        x = torch.randn(2, 1, 16, 16)
        assert_gradient_flows(model, x)
