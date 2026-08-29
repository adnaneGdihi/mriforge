"""Unit tests for HyperMambaBridge and SSMParameters."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.hyper_mamba_bridge import HyperMambaBridge, SSMParameters
from mriforge.models.registry import MODEL_REGISTRY


class TestSSMParameters:
    """Tests for the SSMParameters dataclass."""

    def test_fields(self) -> None:
        """SSMParameters has B_proj, C_proj, Delta_proj."""
        ssm = SSMParameters(
            B_proj=torch.randn(2, 256 * 16),
            C_proj=torch.randn(2, 256 * 16),
            Delta_proj=torch.randn(2, 256),
        )
        assert ssm.B_proj.shape == (2, 256 * 16)
        assert ssm.C_proj.shape == (2, 256 * 16)
        assert ssm.Delta_proj.shape == (2, 256)


class TestHyperMambaBridge:
    """Tests for the HyperMamba bridge network."""

    @pytest.fixture
    def bridge(self) -> HyperMambaBridge:
        return HyperMambaBridge(
            in_channels=2,
            out_channels=2,
            latent_dim=64,
            mamba_d_state=8,
            mamba_d_model=128,
            num_mamba_layers=4,
        )

    def test_forward_output_list_length(self, bridge: HyperMambaBridge) -> None:
        """Forward returns list of SSMParameters with length == num_mamba_layers."""
        fiducial = torch.randn(2, 2, 64, 64)
        ssm_list = bridge(fiducial)
        assert isinstance(ssm_list, list)
        assert len(ssm_list) == 4

    def test_ssm_params_shapes(self, bridge: HyperMambaBridge) -> None:
        """Each SSMParameters has correct B/C/Δ shapes."""
        fiducial = torch.randn(2, 2, 64, 64)
        ssm_list = bridge(fiducial)
        for ssm in ssm_list:
            assert isinstance(ssm, SSMParameters)
            # B_proj: [B, d_model * d_state] = [2, 128 * 8]
            assert ssm.B_proj.shape == (2, 128 * 8)
            # C_proj: same shape
            assert ssm.C_proj.shape == (2, 128 * 8)
            # Delta_proj: [B, d_model] = [2, 128]
            assert ssm.Delta_proj.shape == (2, 128)

    def test_delta_positive(self, bridge: HyperMambaBridge) -> None:
        """Delta_proj should be strictly positive (exp enforced)."""
        fiducial = torch.randn(2, 2, 64, 64)
        ssm_list = bridge(fiducial)
        for ssm in ssm_list:
            assert (ssm.Delta_proj > 0).all(), "Delta must be > 0"

    def test_registered(self) -> None:
        """Model is registered in MODEL_REGISTRY."""
        assert "hyper_mamba_bridge" in MODEL_REGISTRY
        assert MODEL_REGISTRY["hyper_mamba_bridge"]["class"] is HyperMambaBridge

    def test_different_fiducials_different_params(
        self, bridge: HyperMambaBridge
    ) -> None:
        """Different fiducial inputs should produce different SSM params."""
        f1 = torch.randn(1, 2, 64, 64)
        f2 = torch.randn(1, 2, 64, 64) * 3.0
        ssm1 = bridge(f1)
        ssm2 = bridge(f2)
        # At least one parameter should differ
        assert not torch.allclose(ssm1[0].B_proj, ssm2[0].B_proj, atol=1e-6)

    def test_gradient_flow(self, bridge: HyperMambaBridge) -> None:
        """Gradients should flow through the bridge."""
        fiducial = torch.randn(1, 2, 32, 32, requires_grad=True)
        ssm_list = bridge(fiducial)
        loss = sum(ssm.B_proj.sum() + ssm.C_proj.sum() for ssm in ssm_list)
        loss.backward()
        assert fiducial.grad is not None
        assert fiducial.grad.abs().sum() > 0
