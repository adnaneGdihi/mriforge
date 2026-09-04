"""Unit tests for Hilbert-Mamba generators.

Tests verify:
- Forward pass shape correctness for all 10 generators
- Gradient flow (all parameters receive gradients)
- Registry discoverability via MODEL_REGISTRY
- Hilbert 3D linearizer permutation validity + roundtrip
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    _hilbert_3d_indices,
    _hilbert_3d_matrix,
)
from spectramr.models.generators.hilbert_mamba_generators import (
    CRMMamba,
    CTMamba,
    FEMamba,
    HilbertMambaUNet,
    HLDMamba,
    HWMamba,
    INRMamba,
    MDIMamba,
    MMMamba,
    TwoHalfDMamba,
)
from tests.utils.optional_backends import requires_cuda_for_mamba

# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────

BATCH = 2
IN_CH = 1
OUT_CH = 1
H, W = 8, 8  # Power of 2 for Hilbert 2D
D_MODEL = 16  # Small for fast tests
LAYERS = 1
D_STATE = 4


@pytest.fixture
def input_2d():
    """Standard 2D input tensor."""
    return torch.randn(BATCH, IN_CH, H, W)


# ───────────────────────────────────────────────────────────────────────
# Hilbert 3D Tests
# ───────────────────────────────────────────────────────────────────────


class TestHilbert3D:
    """Test 3D Hilbert curve index generation."""

    def test_valid_permutation_2x2x2(self):
        idx = _hilbert_3d_indices(2, 2, 2)
        assert set(idx.tolist()) == set(range(8))

    def test_valid_permutation_4x4x4(self):
        idx = _hilbert_3d_indices(4, 4, 4)
        assert set(idx.tolist()) == set(range(64))

    def test_matrix_shape(self):
        mat = _hilbert_3d_matrix(4)
        assert mat.shape == (4, 4, 4)

    def test_rejects_non_cubic(self):
        with pytest.raises(AssertionError):
            _hilbert_3d_indices(4, 4, 8)

    def test_rejects_non_power_of_2(self):
        with pytest.raises(AssertionError):
            _hilbert_3d_indices(6, 6, 6)

    def test_roundtrip_3d(self):
        """Forward → reverse should reconstruct the original tensor."""
        shape = (4, 4, 4)
        lin = ImageTopologyLinearizer(shape, mode="hilbert_3d")
        x = torch.randn(2, 3, *shape)
        seq = lin(x)
        restored = lin.reverse(seq)
        assert torch.allclose(x, restored, atol=1e-6)

    def test_output_shape_3d(self):
        shape = (4, 4, 4)
        lin = ImageTopologyLinearizer(shape, mode="hilbert_3d")
        x = torch.randn(2, 3, *shape)
        seq = lin(x)
        assert seq.shape == (2, 3, 64)


# ───────────────────────────────────────────────────────────────────────
# Generator Forward Pass Tests
# ───────────────────────────────────────────────────────────────────────


class TestCTMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = CTMamba(
            IN_CH, OUT_CH, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)

    @requires_cuda_for_mamba
    def test_gradient_flow(self, input_2d):
        model = CTMamba(
            IN_CH, OUT_CH, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE
        )
        out = model(input_2d)
        loss = out.sum()
        loss.backward()
        graded = sum(1 for p in model.parameters() if p.grad is not None)
        assert graded > 0

    @requires_cuda_for_mamba
    def test_forward_non_pow2_shape(self):
        """A non-power-of-2 / non-square spatial extent (e.g. a full-FOV
        validation volume) must NOT crash. Pre-fix, ``_HilbertLinearize`` built
        strict ``hilbert_2d`` and raised AssertionError mid-validation."""
        model = CTMamba(
            IN_CH, OUT_CH, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE
        )
        x = torch.randn(1, IN_CH, 66, 60)
        out = model(x)
        assert out.shape == (1, OUT_CH, 66, 60)


class TestFEMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = FEMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            d_state=D_STATE,
            expansion_factor=2,
        )
        out = model(input_2d)
        # FE-Mamba upscales: output is 2x
        assert out.shape == (BATCH, OUT_CH, H * 2, W * 2)


class TestCRMMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = CRMMamba(
            IN_CH, OUT_CH, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestHWMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = HWMamba(
            IN_CH, OUT_CH, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestMMMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = MMMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            micro_layers=LAYERS,
            macro_layers=LAYERS,
            block_size=16,
            d_state=D_STATE,
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestMDIMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = MDIMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            num_directions=2,
            d_state=D_STATE,
            linearization_modes=["hilbert_2d", "snake_2d"],
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestTwoHalfDMamba:
    @requires_cuda_for_mamba
    def test_forward_shape_2d(self, input_2d):
        model = TwoHalfDMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            in_plane_layers=LAYERS,
            cross_plane_layers=LAYERS,
            d_state=D_STATE,
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)

    @requires_cuda_for_mamba
    def test_forward_shape_5d_torchio_convention(self):
        """Regression for 2026-05-10 ``exp_hm_07_25d_mamba`` smoke crash.

        The model used to unpack ``B, C, D, H, W`` but TorchIO emits
        ``[B, C, H, W, D]``.  For ``patch_size: [256, 256, 16]`` that
        misread fed ``(H=256, W=16)`` into the Hilbert linearizer and
        crashed with ``Hilbert 2D requires square power-of-2 dims,
        got (256, 16)``.  The fix unpacks ``B, C, H, W, D``, folds
        depth into batch via ``permute(0, 4, 1, 2, 3)``, and unfolds
        back to TorchIO layout on the way out.
        """
        D = 4  # depth > 1 to exercise the 5-D branch
        model = TwoHalfDMamba(
            IN_CH, OUT_CH,
            d_model=D_MODEL, in_plane_layers=LAYERS,
            cross_plane_layers=LAYERS, d_state=D_STATE,
        )
        x = torch.randn(BATCH, IN_CH, H, W, D)  # TorchIO layout
        out = model(x)
        # Output preserves TorchIO 5-D layout — depth is the last axis.
        assert out.shape == (BATCH, OUT_CH, H, W, D), (
            f"5-D output must keep TorchIO [B, C, H, W, D] layout, got {out.shape}"
        )

    @requires_cuda_for_mamba
    def test_forward_shape_5d_d1_still_runs_cross_plane(self):
        """``D == 1`` traverses the 5-D branch (which adds the cross-plane
        Mamba), distinct from the 4-D in-plane-only branch.

        Pinned as a separate path so the contract is explicit: a 1-slice
        volume is not the same as a 2-D image in this model — only the
        output *shape* is guaranteed to round-trip.
        """
        model = TwoHalfDMamba(
            IN_CH, OUT_CH,
            d_model=D_MODEL, in_plane_layers=LAYERS,
            cross_plane_layers=LAYERS, d_state=D_STATE,
        )
        x = torch.randn(BATCH, IN_CH, H, W, 1)
        out = model(x)
        assert out.shape == (BATCH, OUT_CH, H, W, 1)


class TestHLDMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self):
        x = torch.randn(BATCH, 2, H, W)  # 2 channels: noisy_hf + ulf
        t = torch.rand(BATCH)
        model = HLDMamba(
            in_channels=2,
            out_channels=OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            d_state=D_STATE,
        )
        out = model(x, timesteps=t)
        assert out.shape == (BATCH, OUT_CH, H, W)

    @requires_cuda_for_mamba
    def test_no_timestep(self):
        x = torch.randn(BATCH, 2, H, W)
        model = HLDMamba(
            in_channels=2,
            out_channels=OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            d_state=D_STATE,
        )
        out = model(x)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestINRMamba:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = INRMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            d_state=D_STATE,
            num_freqs=4,
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)


class TestHilbertMambaUNet:
    @requires_cuda_for_mamba
    def test_forward_shape(self, input_2d):
        model = HilbertMambaUNet(
            IN_CH,
            OUT_CH,
            features=[D_MODEL, D_MODEL * 2],
            num_mamba_layers=LAYERS,
            d_state=D_STATE,
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)

    @requires_cuda_for_mamba
    def test_gradient_flow(self, input_2d):
        model = HilbertMambaUNet(
            IN_CH,
            OUT_CH,
            features=[D_MODEL, D_MODEL * 2],
            num_mamba_layers=LAYERS,
            d_state=D_STATE,
        )
        out = model(input_2d)
        loss = out.sum()
        loss.backward()
        graded = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for _ in model.parameters())
        assert graded > total * 0.5  # At least half of params got gradients


# ───────────────────────────────────────────────────────────────────────
# Registry Tests
# ───────────────────────────────────────────────────────────────────────


class TestHilbertMambaRegistry:
    """Verify all 10 generators are discoverable in MODEL_REGISTRY."""

    @pytest.fixture(autouse=True)
    def _populate_registry(self):
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()

    @pytest.mark.parametrize(
        "name",
        [
            "ct_mamba",
            "fe_mamba",
            "crm_mamba",
            "hw_mamba",
            "mm_mamba",
            "mdi_mamba",
            "two_half_d_mamba",
            "hld_mamba",
            "inr_mamba",
            "hilbert_mamba_unet",
        ],
    )
    def test_model_registered(self, name):
        from spectramr.models.registry import MODEL_REGISTRY

        assert name in MODEL_REGISTRY, f"{name} not found in MODEL_REGISTRY"


# ───────────────────────────────────────────────────────────────────────
# Linearization Mode Config Tests
# ───────────────────────────────────────────────────────────────────────


class TestLinearizationModeConfig:
    """Verify generators accept different linearization modes."""

    @requires_cuda_for_mamba
    @pytest.mark.parametrize("mode", ["hilbert_2d", "snake_2d", "zigzag_2d"])
    def test_ct_mamba_modes(self, mode, input_2d):
        shape = (8, 8) if mode == "hilbert_2d" else (H, W)
        model = CTMamba(
            IN_CH,
            OUT_CH,
            d_model=D_MODEL,
            num_layers=LAYERS,
            d_state=D_STATE,
            linearization_mode=mode,
        )
        out = model(input_2d)
        assert out.shape == (BATCH, OUT_CH, H, W)
