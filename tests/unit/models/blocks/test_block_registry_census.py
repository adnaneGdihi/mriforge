"""Census ratchet for ``BLOCK_REGISTRY`` — the reusable-block reachability gate.

The repo rule is "zero *static* consumers != dead": a block reachable through
``BLOCK_REGISTRY`` / ``create_block(name)`` (config-dispatched) is USED even if no
Python file imports it directly. What was missing (per the WS-6/WS-7 exploration:
"no fitness check polices dark blocks") is a gate that pins the reachable surface
so a block silently going dark — dropped from the registry by an import
regression or a removed ``@register_block`` — becomes a CI failure, not an
invisible catalog loss.

This suite asserts the structural invariants (every entry is an ``nn.Module``
class; ``create_block`` raises on an unknown name — no silent fallback, pitfall
#9) and pins the full roster. Adding or removing a block is then a deliberate
edit to ``EXPECTED_BLOCKS`` below.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from spectramr.models.blocks.block_registry import (
    BLOCK_REGISTRY,
    create_block,
    list_registered_blocks,
)

pytestmark = pytest.mark.unit

# The full reachable-block roster as of the WS-7 sweep (2026-07-02). A block
# leaving this set (registry regression) or joining it (new block) is a
# deliberate act — update this pin in the same change.
EXPECTED_BLOCKS: frozenset[str] = frozenset(
    {
        "adaptive_fusion",
        "attention_fusion",
        "bloch_signal_encoder",
        "cross_attention",
        "cross_domain_attention",
        "diffusion_attention",
        "feature_fusion",
        "graph_attention",
        "kspace_radial_swin",
        "layer_norm",
        "local_graph_attention",
        "multi_contrast_kspace_cross_attention",
        "multimodal_fusion",
        "multiscale_freq_band_attention",
        "multiscale_fusion",
        "placeholder_identity",
        "preact_residual_block",
        "radial_band_attention",
        "radial_window_attention",
        "residual_block",
        "residual_fusion",
        "self_attention",
        "spatial_attention",
        "spatiotemporal_marker_attention",
        "structure_tensor_attention",
        "swin_block",
        "trellis_absolute_position_embedder",
        "trellis_attention",
        "trellis_cross_attention",
        "trellis_self_attention_2d",
        "trellis_self_attention_rms",
        "trellis_transformer_block",
        "unet_adagn_double_conv",
        "unet_adagn_down",
        "unet_adagn_up",
        "unet_complex_adagn_double_conv",
        "unet_complex_adagn_down",
        "unet_complex_adagn_up",
        "unet_complex_double_conv",
        "unet_complex_down",
        "unet_complex_up",
        "unet_double_conv",
        "unet_down",
        "unet_out_conv",
        "unet_up",
        "unified_residual_block",
        "window_attention",
    }
)


def test_registry_is_nonempty_above_floor() -> None:
    assert len(BLOCK_REGISTRY) >= 45, (
        f"BLOCK_REGISTRY has {len(BLOCK_REGISTRY)} blocks, below the floor — a "
        "block family likely went dark (import regression / removed decorator)."
    )


def test_every_entry_is_an_nn_module_class() -> None:
    bad = {
        name: type(cls).__name__
        for name, cls in BLOCK_REGISTRY.items()
        if not (isinstance(cls, type) and issubclass(cls, nn.Module))
    }
    assert not bad, f"BLOCK_REGISTRY entries that are not nn.Module classes: {bad}"


def test_create_block_unknown_raises_no_silent_fallback() -> None:
    """An unknown block name must raise, never degrade to a default (pitfall #9)."""
    with pytest.raises(ValueError):
        create_block("__definitely_not_a_registered_block__")


def test_roster_matches_pin() -> None:
    """The reachable roster equals the pin — a dark/removed block trips this.

    ``trellis_self_attention_2d`` is conditionally registered behind an optional
    trellis-attention import; allow it to be absent when that dep is missing.
    """
    live = set(list_registered_blocks())
    optional = {"trellis_self_attention_2d"}
    missing = (EXPECTED_BLOCKS - optional) - live
    extra = live - EXPECTED_BLOCKS
    assert not missing, f"expected blocks missing from the registry (went dark): {sorted(missing)}"
    assert not extra, (
        f"new blocks not in the census pin: {sorted(extra)} — add them to "
        "EXPECTED_BLOCKS deliberately."
    )
