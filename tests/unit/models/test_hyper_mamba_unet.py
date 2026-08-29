"""Unit tests for HyperMambaUNet with SSM injection."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.hyper_mamba_unet import HyperMambaUNet
from mriforge.models.registry import MODEL_REGISTRY
from tests.utils.optional_backends import requires_cuda_for_mamba


class TestHyperMambaUNet:
    """Tests for HyperMambaUNet with bridge-conditioned reconstruction."""

    @pytest.fixture
    def model(self) -> HyperMambaUNet:
        return HyperMambaUNet(
            in_channels=2,
            out_channels=2,
            features=[16, 32],
            bridge_latent_dim=32,
            bridge_d_state=4,
            bridge_d_model=64,
            fiducial_channels=2,
            time_embedding_dim=None,
        )

    @requires_cuda_for_mamba
    def test_forward_with_fiducial(self, model: HyperMambaUNet) -> None:
        """Forward with corrupted_fiducial produces correct output shape."""
        x = torch.randn(2, 2, 32, 32)
        fiducial = torch.randn(2, 2, 32, 32)
        out = model(x, corrupted_fiducial=fiducial)
        assert out.shape == (2, 2, 32, 32)

    @requires_cuda_for_mamba
    def test_forward_without_fiducial(self, model: HyperMambaUNet) -> None:
        """Forward without fiducial (vanilla backbone) works."""
        x = torch.randn(2, 2, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 32, 32)

    @requires_cuda_for_mamba
    def test_ssm_params_reach_backbone(self, model: HyperMambaUNet) -> None:
        """SSM params from bridge should be passed to backbone.forward() as ssm_params.

        We verify by confirming that:
        1. Bridge generates SSM params from fiducial
        2. Those params are passed to backbone (ssm_params kwarg exists)
        3. MambaLayer2D receives and processes them (lazy proj gets created)
        """
        x = torch.randn(1, 2, 32, 32)
        fiducial = torch.randn(1, 2, 32, 32)

        # After forward with fiducial, lazy SSM mod proj should be initialized
        model.train()
        _ = model(x, corrupted_fiducial=fiducial)

        # Check that at least one MambaLayer2D in the backbone has its
        # _ssm_mod_proj initialized (proves params were consumed)
        from mriforge.models.generators.mamba_unet import MambaLayer2D

        found_initialized = False
        for m in model.backbone.modules():
            if isinstance(m, MambaLayer2D) and m._ssm_mod_proj is not None:
                found_initialized = True
                break

        assert found_initialized, (
            "No MambaLayer2D had _ssm_mod_proj initialized — "
            "SSM params from bridge are not reaching the backbone"
        )

    def test_freeze_backbone(self, model: HyperMambaUNet) -> None:
        """freeze_backbone() should set backbone params to requires_grad=False."""
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Bridge should still be trainable
        for param in model.bridge.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model: HyperMambaUNet) -> None:
        """unfreeze_backbone() should restore backbone param gradients."""
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_registered(self) -> None:
        """Model is registered in MODEL_REGISTRY."""
        assert "hyper_mamba_unet" in MODEL_REGISTRY
        assert MODEL_REGISTRY["hyper_mamba_unet"]["class"] is HyperMambaUNet

    def test_accepts_kspace_and_image_domains(self) -> None:
        """Regression for dispatch 6944227 (2026-05-25).

        VF / motion-meta strategies IFFT k-space → image internally before
        this backbone runs, so the model genuinely accepts either on-disk
        domain. The registration must advertise both so a kspace-dataset VF
        arm (coil_processing_mode=svd) passes the ``data_model_compatibility``
        audit instead of being rejected as 'kspace data vs image model'.
        """
        from mriforge.models.registry import get_model_capabilities

        caps = get_model_capabilities("hyper_mamba_unet")
        assert caps is not None
        for domain in (caps.input_domain, caps.output_domain):
            assert domain is not None
            assert "image" in domain and "kspace" in domain
        # spatial-dims safety must remain active (caps not fully unannotated).
        assert caps.spatial_dims == (2, 3)

    def test_bridge_gradient_flow(self, model: HyperMambaUNet) -> None:
        """Gradients should flow through bridge when computing loss on SSM params."""
        fiducial = torch.randn(1, 2, 32, 32, requires_grad=True)

        # Run bridge only (not full backbone — that has zero-init which blocks grad)
        ssm_params = model.bridge(fiducial)
        loss = sum(ssm.B_proj.sum() + ssm.C_proj.sum() for ssm in ssm_params)
        loss.backward()
        assert fiducial.grad is not None, "No gradient on fiducial input"
        assert fiducial.grad.abs().sum() > 0, "Fiducial gradient is all zeros"
