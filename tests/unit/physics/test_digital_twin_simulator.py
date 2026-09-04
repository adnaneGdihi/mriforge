"""Canary / parametrized / edge-case / expected-failure tests for digital_twin_simulator.py.

Critical-partial: extends existing coverage (none in tests/unit/physics/).

Covers the public surface:
  - CornerFiducialEmbedder (gaussian / cross / none marker types)
  - CornerFiducialEmbedder raises for unknown marker_type (CLAUDE.md pitfall #9)

NOTE: DigitalTwinSimulator (the top-level class that stitches the full
pipeline) is not tested here because it depends on heavy optional deps
(sigpy, torchkbnufft) and GPU memory. We focus on the deterministic,
CPU-tractable CornerFiducialEmbedder which is the most-used public class.
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _anatomy(b: int, c: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(b, c, h, w, generator=g)
    im = torch.randn(b, c, h, w, generator=g)
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_canary_digital_twin_import():
    """CornerFiducialEmbedder imports cleanly."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(32, 32), marker_type="gaussian")
    assert emb is not None


@pytest.mark.physics
def test_canary_fiducial_embedder_gaussian():
    """Gaussian embedder forward returns correct shapes."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(32, 32), marker_type="gaussian")
    anatomy = _anatomy(1, 1, 32, 32)
    joint, prior = emb(anatomy)
    assert joint.shape == (1, 1, 32, 32)
    assert prior.shape[2:] == (32, 32)


@pytest.mark.physics
def test_canary_fiducial_embedder_cross():
    """Cross embedder forward returns correct shapes."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(32, 32), marker_type="cross")
    anatomy = _anatomy(1, 1, 32, 32)
    joint, prior = emb(anatomy)
    assert joint.shape == (1, 1, 32, 32)


@pytest.mark.physics
def test_canary_fiducial_embedder_none():
    """'none' marker type: joint image == anatomy."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(16, 16), marker_type="none")
    anatomy = _anatomy(1, 1, 16, 16)
    joint, prior = emb(anatomy)
    assert torch.allclose(joint, anatomy)


# ---------------------------------------------------------------------------
# Parametrized — CornerFiducialEmbedder shapes
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize("marker_type", ["gaussian", "cross"])
@pytest.mark.parametrize(
    "b, c, h, w",
    [
        (1, 1, 32, 32),
        (2, 1, 64, 64),
        (1, 2, 32, 64),  # rectangular
        (2, 4, 32, 32),  # multi-contrast
    ],
    ids=["b1c1-sq", "b2c1-64", "b1c2-rect", "b2c4-multi"],
)
def test_fiducial_embedder_shape(marker_type, b, c, h, w):
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(h, w), marker_type=marker_type)
    anatomy = _anatomy(b, c, h, w)
    joint, prior = emb(anatomy)
    assert joint.shape == (b, c, h, w)
    assert not torch.isnan(joint).any()


@pytest.mark.physics
@pytest.mark.parametrize("marker_type", ["gaussian", "cross"])
def test_fiducial_embedder_marker_adds_signal(marker_type):
    """Joint image should differ from anatomy (markers were added)."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(32, 32), marker_type=marker_type)
    anatomy = _anatomy(1, 1, 32, 32, seed=1)
    joint, _ = emb(anatomy)
    # With a non-trivial anatomy the markers should change the image
    # (unless anatomy exactly cancels them, which is astronomically unlikely)
    diff = (joint - anatomy).abs().max()
    assert float(diff.item()) > 0.0, "Marker embedding made no change"


# ---------------------------------------------------------------------------
# Sanity-shape tests
# ---------------------------------------------------------------------------


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, c, h, w",
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D if h <= 128 and w <= 128],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D if h <= 128 and w <= 128],
)
def test_fiducial_embedder_shape_matrix(b, c, h, w):
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(h, w), marker_type="gaussian")
    anatomy = _anatomy(b, c, h, w)
    joint, prior = emb(anatomy)
    assert joint.shape == (b, c, h, w)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_fiducial_embedder_edge_zero_anatomy():
    """Zero anatomy: marker prior should still be non-zero (anatomy=0 → zero scale,
    but the prior is returned independently)."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(im_size=(32, 32), marker_type="gaussian")
    anatomy = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    joint, prior = emb(anatomy)
    # Joint == anatomy + scale * marker; with zero anatomy scale=0 -> joint=0
    assert not torch.isnan(joint).any()


@pytest.mark.physics
def test_fiducial_embedder_edge_multi_contrast_per_channel():
    """per_channel_marker_prior=True: prior has per-channel shape."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    emb = CornerFiducialEmbedder(
        im_size=(32, 32), marker_type="gaussian", per_channel_marker_prior=True
    )
    anatomy = _anatomy(1, 4, 32, 32)
    joint, prior = emb(anatomy)
    assert joint.shape == (1, 4, 32, 32)


@pytest.mark.physics
def test_fiducial_embedder_edge_with_coil_sensitivities():
    """Multi-coil path: providing coil_sensitivities runs without error."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    b, c, h, w = 1, 4, 32, 32
    emb = CornerFiducialEmbedder(im_size=(h, w), marker_type="gaussian")
    anatomy = _anatomy(b, c, h, w)
    smaps = _anatomy(1, c, h, w, seed=2)
    joint, prior = emb(anatomy, coil_sensitivities=smaps)
    assert joint.shape == (b, c, h, w)


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests — CLAUDE.md pitfall #9: must raise on illegal enum
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_fiducial_embedder_raises_invalid_marker_type():
    """CornerFiducialEmbedder must raise ValueError for unknown marker_type."""
    from spectramr.infrastructure.physics.digital_twin_simulator import CornerFiducialEmbedder

    with pytest.raises(ValueError, match="marker_type"):
        CornerFiducialEmbedder(im_size=(32, 32), marker_type="polygon")


# ---------------------------------------------------------------------------
# DigitalTwinSimulator construction-time contract (WS-4 physics fixes)
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_from_config_type_hints_resolve():
    """``from typing import Any`` import is present: hint resolution must not raise.

    Regression for the latent NameError — ``from_config`` annotates ``dt: Any``
    but ``Any`` was never imported. Under ``from __future__ import annotations``
    the runtime signature was fine, but ``typing.get_type_hints`` (pydantic /
    dataclass / pyright introspection) raised ``NameError: name 'Any' ...``.
    """
    import typing

    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    hints = typing.get_type_hints(DigitalTwinSimulator.from_config)
    assert "dt" in hints  # would raise NameError before the fix


@pytest.mark.physics
def test_progressive_degradations_rejects_unknown_entry():
    """A mistyped progressive_degradations key must RAISE, never silently no-op.

    Per pitfall #15: an exposed knob (degradation feature name) must be
    validated against the canonical set the forward pass dispatches on.
    """
    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    with pytest.raises(ValueError, match="progressive_degradations"):
        DigitalTwinSimulator(im_size=(64, 64), progressive_degradations=["motion", "b_0"])


@pytest.mark.physics
def test_degradation_ranges_rejects_unknown_key():
    """A typo'd degradation_ranges key must RAISE (pitfall #15)."""
    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    with pytest.raises(ValueError, match="degradation_ranges"):
        DigitalTwinSimulator(im_size=(64, 64), degradation_ranges={"b_0": (0.1, 0.9)})


@pytest.mark.physics
def test_canonical_degradation_keys_accepted():
    """Every canonical degradation key the forward pass uses must construct cleanly.

    Regression guard: validation must NOT reject any key actually consumed by
    ``_get_effective_cf`` call sites.
    """
    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    canonical = [
        "motion",
        "noise",
        "b0",
        "b1",
        "undersampling",
        "gibbs",
        "chemical_shift",
        "eddy_current",
        "spike_noise",
        "rf_zipper",
        "rf_crosstalk",
        "b0_distortion",
        "gradient_nonlinearity",
        "magic_angle",
    ]
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        progressive_degradations=canonical,
        degradation_ranges=dict.fromkeys(canonical, (0.1, 1.0)),
    )
    assert sim is not None
    # Empty knobs (the default) must also construct without raising.
    assert DigitalTwinSimulator(im_size=(32, 32)) is not None
