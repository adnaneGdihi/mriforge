"""Unit tests for the octave-convolution block and OctaveUNet generator.

Covers:
* :class:`mriforge.models.blocks.octave_conv.OctaveConv2d` four-pathway
  shapes and gradient flow through all of W_hh / W_hl / W_lh / W_ll.
* :class:`mriforge.models.generators.octave_unet.OctaveUNet` forward shape
  and registration as ``octave_conv``.

These tests are written but not executed here (torch may be a MagicMock
shim in CI); they run on the cluster.
"""

import pytest
import torch

from mriforge.models.blocks.octave_conv import OctaveConv2d, _split_channels
from mriforge.models.generators.octave_unet import OctaveUNet
from mriforge.models.registry import MODEL_REGISTRY, get_model_class


class TestOctaveConv2d:
    def test_split_channels_endpoints(self):
        assert _split_channels(64, 0.0) == (64, 0)
        assert _split_channels(64, 1.0) == (0, 64)
        h, low = _split_channels(64, 0.5)
        assert h == 32 and low == 32

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            OctaveConv2d(8, 8, alpha_in=-0.1)
        with pytest.raises(ValueError):
            OctaveConv2d(8, 8, alpha_out=1.5)

    def test_forward_shapes(self):
        oc = OctaveConv2d(16, 32, kernel_size=3, alpha_in=0.5, alpha_out=0.5)
        x_h = torch.randn(2, 8, 32, 32)
        x_l = torch.randn(2, 8, 16, 16)
        y_h, y_l = oc((x_h, x_l))
        assert y_h.shape == (2, 16, 32, 32)
        assert y_l.shape == (2, 16, 16, 16)

    def test_dense_input_when_alpha_in_zero(self):
        oc = OctaveConv2d(4, 16, alpha_in=0.0, alpha_out=0.5)
        x = torch.randn(2, 4, 16, 16)
        y_h, y_l = oc(x)
        assert y_h is not None and y_h.shape == (2, 8, 16, 16)
        assert y_l is not None and y_l.shape == (2, 8, 8, 8)

    def test_dense_output_when_alpha_out_zero(self):
        oc = OctaveConv2d(16, 4, alpha_in=0.5, alpha_out=0.0)
        x_h = torch.randn(2, 8, 16, 16)
        x_l = torch.randn(2, 8, 8, 8)
        y_h, y_l = oc((x_h, x_l))
        assert y_h is not None and y_h.shape == (2, 4, 16, 16)
        assert y_l is None

    def test_gradients_flow_through_all_four_pathways(self):
        oc = OctaveConv2d(16, 16, alpha_in=0.5, alpha_out=0.5)
        x_h = torch.randn(2, 8, 16, 16, requires_grad=True)
        x_l = torch.randn(2, 8, 8, 8, requires_grad=True)
        y_h, y_l = oc((x_h, x_l))
        (y_h.sum() + y_l.sum()).backward()
        for name in ("conv_hh", "conv_hl", "conv_lh", "conv_ll"):
            conv = getattr(oc, name)
            assert conv is not None, f"{name} pathway missing"
            assert conv.weight.grad is not None, f"no grad for {name}"
            assert conv.weight.grad.abs().sum() > 0, f"zero grad for {name}"


class TestOctaveUNet:
    def test_registered_name(self):
        assert "octave_conv" in MODEL_REGISTRY
        assert get_model_class("octave_conv") is OctaveUNet
        assert MODEL_REGISTRY["octave_conv"]["mode"] == "reconstruction"

    def test_constructible_with_no_args(self):
        model = OctaveUNet()
        assert isinstance(model, OctaveUNet)

    @pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
    def test_forward_shape_preserved(self, alpha):
        model = OctaveUNet(in_channels=1, out_channels=1, base_channels=16, alpha=alpha, depth=2)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_forward_survives_every_depth(self, depth):
        """The skip width is a function of DEPTH, so one depth proves nothing.

        Regression for cluster job 8004252. The decoders are built as
        ``_OctaveBlock(chans[i + 1] + chans[i], chans[i])`` -- upsampled deeper
        feature plus a skip of the width BEFORE that stage widened it -- but the
        encoder loop appended ``enc(feat)``, whose width is ``chans[i + 1]``. So
        every concat was ``2 * chans[i + 1]`` instead of ``chans[i+1] + chans[i]``
        and each decoder received more channels than its convolutions declared:
        at ``base=16, depth=2, alpha=0.25`` the first decoder saw 96 high
        channels against a conv built for 72.

        Parametrised over depth because the mismatch is ``chans[i+1] - chans[i]``
        at every level: a single-depth test could pass on a coincidence.
        """
        model = OctaveUNet(
            in_channels=2, out_channels=2, base_channels=8, alpha=0.25, depth=depth
        )
        out = model(torch.randn(1, 2, 64, 64))
        assert out.shape == (1, 2, 64, 64)

    def test_decoder_input_width_matches_its_declaration(self):
        """Pin the ARITHMETIC, not just that a forward pass survives.

        A future change that widened both the skip and the decoder in step would
        keep the forward working while silently doubling the decoder; this
        asserts the relationship the constructor actually declares.
        """
        base, depth = 16, 2
        model = OctaveUNet(base_channels=base, alpha=0.25, depth=depth)
        chans = [base * (2**i) for i in range(depth + 1)]
        for k, i in enumerate(reversed(range(depth))):
            declared = chans[i + 1] + chans[i]
            oc1 = model.decoders[k].oc1
            # the octave pair splits the declared width across the two branches
            actual = oc1.conv_hh.in_channels + oc1.conv_lh.in_channels
            assert actual == declared, (
                f"decoder {k} is built for {actual} in-channels but the skip "
                f"arithmetic gives chans[{i+1}] + chans[{i}] = {declared}"
            )

    def test_gradient_flows_to_input(self):
        model = OctaveUNet(base_channels=16, depth=2)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
