"""Unit tests for the optimal-transport velocity field network.

Covers :class:`spectramr.models.ot.optimal_transport.VelocityFieldNetwork`:
- the supported forward path preserves the [B, C, H, W] shape;
- the fragile silent-degradation branch (``layer(... else x)``) has been
  removed, so a channel misconfiguration surfaces as a natural PyTorch
  ``RuntimeError`` instead of silently substituting the original input ``x``
  for a conv expecting ``hidden_dim`` channels (pitfall #9 flavour).

Written but not executed here; they run on the cluster.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.models.ot.optimal_transport import VelocityFieldNetwork


class TestVelocityFieldNetworkForward:
    def test_forward_preserves_shape(self) -> None:
        """The supported path returns a velocity field of the input shape."""
        torch.manual_seed(0)
        net = VelocityFieldNetwork(in_channels=2, hidden_dim=128, num_layers=4)
        x = torch.randn(3, 2, 16, 16)
        t = torch.rand(3)
        out = net(x, t)
        assert out.shape == (3, 2, 16, 16)

    def test_forward_scalar_time(self) -> None:
        """A scalar time is broadcast over the batch without error."""
        torch.manual_seed(1)
        net = VelocityFieldNetwork(in_channels=1, hidden_dim=16, num_layers=3)
        x = torch.randn(2, 1, 8, 8)
        out = net(x, torch.tensor(0.5))
        assert out.shape == (2, 1, 8, 8)


class TestNoSilentDegradationFallback:
    def test_forward_source_has_no_else_x_branch(self) -> None:
        """The silent-substitution branch ``layer(... else x)`` must be gone.

        Source-level lock: the old code masked a channel misconfiguration by
        feeding the original input ``x`` to a conv expecting ``hidden_dim``
        channels. Asserting the substring is absent prevents the fallback
        from being reintroduced.
        """
        src = inspect.getsource(VelocityFieldNetwork.forward)
        assert "else x" not in src
        assert "layer.in_channels" not in src

    def test_channel_misconfig_raises_not_degrades(self) -> None:
        """A deliberately wrong in_channels surfaces a RuntimeError.

        The first conv expects ``in_channels + hidden_dim`` channels (the
        concatenation of x and the time embedding). Feeding an input whose
        channel count does not match the network's declared ``in_channels``
        must raise rather than silently produce an output via the removed
        ``else x`` path.
        """
        torch.manual_seed(2)
        net = VelocityFieldNetwork(in_channels=2, hidden_dim=8, num_layers=3)
        # Network expects 2-channel input; pass 5 channels -> first conv sees
        # 5 + 8 = 13 != 2 + 8 = 10 in_channels and must error out.
        bad_x = torch.randn(1, 5, 8, 8)
        t = torch.rand(1)
        with pytest.raises(RuntimeError):
            net(bad_x, t)


class TestEntropicGromovWasserstein:
    """B-3.2: entropic / debiased Gromov-Wasserstein primitive (T1-#22 OT gap).

    Validates the GW math on the cases where GW is meaningful (unpaired structure matching),
    INCLUDING the isometry-invariance property that makes GW unsuitable for paired registered
    image translation (see the module docstring + the B-3.2 finding).
    """

    def test_invalid_inputs_raise(self) -> None:
        from spectramr.models.ot.optimal_transport import entropic_gromov_wasserstein

        with pytest.raises(ValueError):
            entropic_gromov_wasserstein(torch.zeros(3, 4), torch.zeros(4, 4))
        with pytest.raises(ValueError):
            entropic_gromov_wasserstein(torch.zeros(4, 4), torch.zeros(3, 4))

    def test_coupling_is_valid_plan(self) -> None:
        from spectramr.models.ot.optimal_transport import entropic_gromov_wasserstein

        torch.manual_seed(0)
        c1 = torch.rand(8, 8)
        c1 = c1 + c1.t()
        c2 = torch.rand(8, 8)
        c2 = c2 + c2.t()
        t, gw = entropic_gromov_wasserstein(c1, c2)
        assert t.shape == (8, 8)
        assert (t >= 0).all()
        assert abs(float(t.sum()) - 1.0) < 1e-3  # uniform-marginal plan sums to 1
        assert torch.isfinite(gw)

    def test_debiased_self_distance_is_zero(self) -> None:
        # The DEBIASED divergence is 0 for a structure matched to itself (the raw entropic value
        # is NOT — it has a large input-dependent floor, which is why debiasing is required).
        from spectramr.models.ot.optimal_transport import gromov_wasserstein_feature_loss

        torch.manual_seed(0)
        feat = torch.randn(16, 2)
        div, _ = gromov_wasserstein_feature_loss(feat, feat)
        assert abs(float(div)) < 1e-3

    def test_permutation_invariance(self) -> None:
        # Relabeling (permuting) a descriptor set leaves the intra-domain structure unchanged,
        # so the divergence stays ~0 — GW measures relational structure, not labeling.
        from spectramr.models.ot.optimal_transport import gromov_wasserstein_feature_loss

        torch.manual_seed(0)
        feat = torch.randn(16, 2)
        perm = torch.randperm(16)
        div_perm, _ = gromov_wasserstein_feature_loss(feat, feat[perm])
        assert abs(float(div_perm)) < 1e-2

    def test_discriminates_distinct_structures(self) -> None:
        # Genuinely different intra-domain geometry (a line vs a circle) costs much more than a
        # self-match — the property GW IS good at (unpaired structure matching). + determinism>0.
        from spectramr.models.ot.optimal_transport import gromov_wasserstein_feature_loss

        n = 16
        line = torch.stack([torch.linspace(0, 1, n), torch.zeros(n)], dim=1)
        ang = torch.linspace(0, 6.28, n)
        circle = torch.stack([ang.cos(), ang.sin()], dim=1)
        div_self, _ = gromov_wasserstein_feature_loss(line, line)
        div_lc, det_lc = gromov_wasserstein_feature_loss(line, circle)
        assert float(div_lc) > float(div_self) + 0.05
        assert float(det_lc) > 0.0  # coupling is not the degenerate uniform plan

    def test_isometry_invariance_documents_unsuitability_for_translation(self) -> None:
        # GW's DEFINING property: invariance to isometries (here a translation) of the cloud.
        # A spatially DISPLACED structure scores ~0 — exactly why GW cannot penalise displaced
        # anatomy and is unsuitable for paired registered image translation (B-3.2 finding).
        from spectramr.models.ot.optimal_transport import gromov_wasserstein_feature_loss

        torch.manual_seed(0)
        cloud = torch.randn(16, 2)
        translated = cloud + torch.tensor([5.0, -3.0])  # rigid translation = isometry
        div, _ = gromov_wasserstein_feature_loss(cloud, translated)
        assert abs(float(div)) < 1e-2  # GW is blind to the displacement (by design)

    def test_differentiable_into_features(self) -> None:
        from spectramr.models.ot.optimal_transport import gromov_wasserstein_feature_loss

        torch.manual_seed(0)
        feat_a = torch.randn(12, 2)
        feat_b = torch.randn(12, 2, requires_grad=True)
        div, _ = gromov_wasserstein_feature_loss(feat_a, feat_b)
        div.backward()
        assert feat_b.grad is not None
        assert float(feat_b.grad.abs().sum()) > 0.0
