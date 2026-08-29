"""Regression tests for ``TRELLISGenerator`` renderer_type validation.

Pitfall #9/#15: ``renderer_type`` is an exposed ``__init__`` knob with a
documented set {'gaussian', 'mesh', 'volume'}. Both ``_build_renderer_decoder``
and ``_build_slat_renderer`` dispatch with ``if 'gaussian' / elif 'mesh' /
else: <volume>``, so any unrecognized value silently built the volume renderer.
Post-fix: ``__init__`` validates the set at assignment time and RAISES, while
the three valid values still resolve to their expected renderer classes.

Validation is performed before the heavy ViT backbone is built, so the negative
case stays cheap; valid cases use small feature dims to keep CPU runtime tiny.
"""

from __future__ import annotations

import pytest

from mriforge.models.generators.trellis_generator import (
    TrellisGaussianRenderer,
    TRELLISGenerator,
    TrellisMeshRenderer,
    VolumeRenderer,
)


def test_unknown_renderer_type_raises() -> None:
    with pytest.raises(ValueError, match="renderer_type"):
        TRELLISGenerator(renderer_type="nerf")


def _build(renderer_type: str) -> TRELLISGenerator:
    return TRELLISGenerator(
        in_channels=1,
        out_channels=1,
        input_resolution=(16, 16),
        patch_size=8,
        stride=8,
        feature_dim=16,
        num_layers=1,
        num_heads=2,
        num_stages=1,
        num_gaussians=4,
        renderer_type=renderer_type,
    )


@pytest.mark.parametrize(
    "renderer_type,renderer_cls",
    [
        ("gaussian", TrellisGaussianRenderer),
        ("mesh", TrellisMeshRenderer),
        ("volume", VolumeRenderer),
    ],
)
def test_valid_renderer_type_resolves(renderer_type, renderer_cls) -> None:
    gen = _build(renderer_type)
    assert gen.renderer_type == renderer_type
    # The else-branch is now provably 'volume' since invalid values raise.
    assert isinstance(gen._build_slat_renderer(), renderer_cls)
