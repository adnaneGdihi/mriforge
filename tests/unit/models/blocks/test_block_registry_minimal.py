import pytest
import torch

from mriforge.models.blocks.block_registry import (
    create_block,
    list_registered_blocks,
)
from mriforge.models.blocks.dual_domain_attention_kan import (
    CrossDomainAttention,
    MultiScaleFreqBandAttention,
    RadialBandAttention,
)
from mriforge.models.blocks.graph_attention import (
    GraphAttentionBlock,
    LocalGraphAttentionBlock,
)
from mriforge.models.blocks.kspace_swin_attention import (
    KSpaceRadialSwinBlock,
    RadialWindowAttention,
)
from mriforge.models.blocks.multi_contrast_kspace_attention import (
    MultiContrastKSpaceCrossAttention,
)
from mriforge.models.blocks.structure_tensor_attention import (
    StructureTensorAttention,
)
from mriforge.models.blocks.vf_modules import SpatiotemporalMarkerAttention


def test_create_block_unknown_block_raises():
    with pytest.raises(ValueError):
        create_block("nonexistent_block")


def test_fusion_blocks_instantiate_and_forward():
    # (block_type, kwargs, input_count)
    cases = [
        ("attention_fusion", {"channels": 8, "num_inputs": 2}, 2),
        ("feature_fusion", {"channels": 8, "num_inputs": 2}, 2),
        ("multiscale_fusion", {"channels": 8, "num_scales": 3}, 3),
        ("adaptive_fusion", {"channels": 8, "num_inputs": 2}, 2),
        ("residual_fusion", {"channels": 8, "num_inputs": 2}, 2),
        ("multimodal_fusion", {"channels": 8, "num_modalities": 2}, 2),
    ]

    for block_type, kwargs, n in cases:
        layer = create_block(block_type, **kwargs)
        inputs = [torch.randn(1, 8, 8, 8) for _ in range(n)]
        out = layer(inputs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == inputs[0].shape


# ---------------------------------------------------------------------------
# Previously-orphaned attention blocks (now registry-reachable).
#
# These attention modules were defined and exported but never constructed by
# any registered model/strategy/block (the 2026-06-24 attention-wiring audit).
# Registering them in BLOCK_REGISTRY makes them config-reachable via
# create_block(); the registry contract is *construction* (create_block calls
# block_class(**kwargs)). See docs/attention_wiring_audit.rst.
# ---------------------------------------------------------------------------

# (block_type, kwargs, expected_class)
_NEWLY_REGISTERED_ATTENTION = [
    ("graph_attention", {"in_channels": 8, "out_channels": 8, "heads": 2}, GraphAttentionBlock),
    (
        "local_graph_attention",
        {"in_channels": 8, "out_channels": 8, "k_neighbors": 4, "heads": 2},
        LocalGraphAttentionBlock,
    ),
    (
        "multi_contrast_kspace_cross_attention",
        {"channels": 4, "n_heads": 2, "patch_size": 4},
        MultiContrastKSpaceCrossAttention,
    ),
    ("kspace_radial_swin", {"dim": 8, "num_heads": 2, "window_size": 4}, KSpaceRadialSwinBlock),
    ("radial_window_attention", {"dim": 8, "num_heads": 2, "window_size": 4}, RadialWindowAttention),
    (
        "spatiotemporal_marker_attention",
        {"dim": 8, "num_heads": 2, "num_timepoints": 4},
        SpatiotemporalMarkerAttention,
    ),
    # 2026-06-27 follow-up: standalone mechanisms previously reachable only as
    # hard-coded sub-components of a generator-used parent block. Non-trivial
    # forward contracts (complex / two-input / structure-tensor side input), so
    # these are construction-only here — forward is the caller's concern.
    ("structure_tensor_attention", {"dim": 8, "num_heads": 2}, StructureTensorAttention),
    ("radial_band_attention", {"channels": 8, "num_heads": 2}, RadialBandAttention),
    (
        "multiscale_freq_band_attention",
        {"channels": 8, "num_heads": 2},
        MultiScaleFreqBandAttention,
    ),
    ("cross_domain_attention", {"channels": 8, "num_heads": 2}, CrossDomainAttention),
]


@pytest.mark.parametrize("block_type, kwargs, expected_cls", _NEWLY_REGISTERED_ATTENTION)
def test_newly_registered_attention_blocks_resolve(block_type, kwargs, expected_cls):
    """Each formerly-orphaned attention block is now in the registry and builds."""
    assert block_type in list_registered_blocks()
    block = create_block(block_type, **kwargs)
    assert isinstance(block, expected_cls)


def test_trellis_self_attention_2d_registered():
    """The trellis 2D self-attention block is conditionally registered.

    It depends on the optional trellis attention import (guarded by
    ImportError in block_registry); when available it must be reachable.
    """
    from mriforge.models.blocks import block_registry as br

    if br.TrellisMultiHeadAttention is None:  # optional dep absent
        pytest.skip("trellis attention optional dependency unavailable")
    assert "trellis_self_attention_2d" in list_registered_blocks()
    block = create_block("trellis_self_attention_2d", channels=8, num_heads=2)
    from mriforge.models.blocks.trellis_attention import TrellisSelfAttention2D

    assert isinstance(block, TrellisSelfAttention2D)


def test_newly_registered_attention_blocks_forward_smoke():
    """Forward smoke for the blocks with a known plain input contract."""
    # Radial attention / swin: (B, L, C) tokens + (H, W) grid, L == H*W.
    x_tok = torch.randn(1, 16, 8)
    rwa = create_block("radial_window_attention", dim=8, num_heads=2, window_size=4)
    assert rwa(x_tok, 4, 4).shape == x_tok.shape
    rsb = create_block("kspace_radial_swin", dim=8, num_heads=2, window_size=4)
    assert rsb(x_tok, 4, 4).shape == x_tok.shape

    # Multi-contrast cross-attention: (B, n_contrasts, C, H, W), H,W % patch == 0.
    x_mc = torch.randn(1, 2, 4, 8, 8)
    mcc = create_block(
        "multi_contrast_kspace_cross_attention", channels=4, n_heads=2, patch_size=4
    )
    assert mcc(x_mc).shape == x_mc.shape
