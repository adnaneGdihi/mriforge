"""Unit tests for MIND-SSC Loss."""

import torch

from spectramr.models.losses.mind_ssc_loss import MINDSSCLoss


class TestMINDSSCLoss:
    """Test MINDSSCLoss functionality."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        loss_fn = MINDSSCLoss()
        assert loss_fn.patch_size == 7
        assert loss_fn.search_radius == 3
        assert loss_fn.gaussian_sigma == 1.0
        assert loss_fn.use_self_similarity is True

    def test_forward_pass_shapes(self):
        """Test forward pass output shape/type."""
        loss_fn = MINDSSCLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_gradient_stability_constant_input(self):
        """Test gradient stability with constant input (variance=0).

        This tests the fix for SqrtBackward0 NaN error when local variance is 0.
        """
        loss_fn = MINDSSCLoss()
        # Constant image -> local variance is 0
        pred = torch.ones(1, 1, 32, 32, requires_grad=True)
        target = torch.ones(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        # Check for NaNs in gradient
        assert not torch.isnan(pred.grad).any(), "Gradient contains NaNs!"
        assert not torch.isinf(pred.grad).any(), "Gradient contains Infs!"

    def test_gradient_stability_random_input(self):
        """Test gradient stability with random input."""
        loss_fn = MINDSSCLoss()
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randn(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        assert not torch.isnan(pred.grad).any()

    def test_direction_set_is_symmetric(self):
        """The descriptor neighbourhood must be closed under negation.

        Regression: the old ``directions[:6]`` slice kept an asymmetric subset
        (it dropped ``(1, 0)`` and ``(1, 1)`` while keeping their negatives),
        which biased the descriptor by orientation. The full 8-neighbourhood is
        symmetric: every ``(dx, dy)`` has its mirror ``(-dx, -dy)``.
        """
        loss_fn = MINDSSCLoss()
        dirs = set(loss_fn.directions)
        assert len(dirs) == 8
        for dx, dy in dirs:
            assert (-dx, -dy) in dirs, f"missing mirror of {(dx, dy)}"

    def test_descriptor_has_eight_directional_channels(self):
        """Each input channel yields 8 directional MIND features."""
        loss_fn = MINDSSCLoss()
        desc = loss_fn._compute_mind_descriptor(torch.rand(1, 1, 16, 16))
        assert desc.shape[1] == 8
