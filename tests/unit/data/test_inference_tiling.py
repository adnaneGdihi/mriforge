"""Tests for sliding-window inference tiling (Phase 2).

Coverage:

1. **Identity round-trip** — feeding a known volume's tiles through an
   identity model, then aggregating, recovers the input to float
   precision. Pins the numerical correctness of tile-then-Hann-aggregate.
2. **Overlap validation** — overlap >= patch_size, odd overlap, and
   negative overlap are all rejected with clear messages.
3. **Auto-overlap heuristic** — when ``patch_overlap=None`` is given,
   ``_auto_overlap`` returns even values bounded by patch_size.
4. **Per-strategy opt-in audit** — ``check_grid_tiling_supported``
   accepts compatible strategies and rejects incompatible ones with
   actionable messages.
"""
from __future__ import annotations

import pytest
import torch

from mriforge.data.builders.inference_tiling import (
    GridInferenceHandle,
    _auto_overlap,
    _validate_overlap,
    build_grid_inference_handle,
)
from mriforge.data.builders.inference_tiling_audit import (
    check_grid_tiling_supported,
    list_grid_tiling_compatible_strategies,
)


# ── Numerical correctness ───────────────────────────────────────────────────


def _build_subject(shape=(1, 64, 64, 16), seed=0):
    """Tiny TorchIO Subject with deterministic content."""
    import torchio as tio

    g = torch.Generator().manual_seed(seed)
    vol = torch.randn(*shape, generator=g)
    return tio.Subject(image=tio.ScalarImage(tensor=vol)), vol


def test_identity_round_trip_hann_aggregator() -> None:
    """For an identity model (pred = input), the Hann aggregator
    must reconstruct the original volume within float tolerance."""
    import torchio as tio

    subj, vol = _build_subject()
    handle = build_grid_inference_handle(
        subject=subj,
        patch_size=(32, 32, 8),
        patch_overlap=(8, 8, 2),
        aggregator="hann",
    )
    for patch in handle.sampler:
        pred = patch["image"][tio.DATA]
        handle.aggregator.add_batch(
            pred.unsqueeze(0), patch[tio.LOCATION].unsqueeze(0)
        )
    out = handle.aggregator.get_output_tensor()
    assert out.shape == vol.shape
    assert (out - vol).abs().max().item() < 1e-3


def test_identity_round_trip_average_aggregator() -> None:
    """``average`` aggregation is also exact for identity inputs."""
    import torchio as tio

    subj, vol = _build_subject()
    handle = build_grid_inference_handle(
        subject=subj,
        patch_size=(32, 32, 8),
        patch_overlap=(0, 0, 0),  # no overlap → trivially exact
        aggregator="average",
    )
    for patch in handle.sampler:
        pred = patch["image"][tio.DATA]
        handle.aggregator.add_batch(
            pred.unsqueeze(0), patch[tio.LOCATION].unsqueeze(0)
        )
    out = handle.aggregator.get_output_tensor()
    assert (out - vol).abs().max().item() < 1e-5


def test_2d_tiling_works_with_d_equals_1() -> None:
    """2D inference uses ``patch_size=(H, W, 1)`` — must still tile correctly."""
    import torchio as tio

    subj, vol = _build_subject(shape=(1, 64, 64, 1))
    handle = build_grid_inference_handle(
        subject=subj,
        patch_size=(32, 32, 1),
        patch_overlap=(8, 8, 0),
        aggregator="hann",
    )
    for patch in handle.sampler:
        pred = patch["image"][tio.DATA]
        handle.aggregator.add_batch(
            pred.unsqueeze(0), patch[tio.LOCATION].unsqueeze(0)
        )
    out = handle.aggregator.get_output_tensor()
    assert out.shape == vol.shape


# ── Overlap validation ──────────────────────────────────────────────────────


def test_overlap_equal_to_patch_size_raises() -> None:
    """``overlap == patch_size`` would silently produce zero patches in
    TorchIO. We surface that as a clear ValueError."""
    with pytest.raises(ValueError, match="strictly less than"):
        _validate_overlap(patch_size=(32, 32, 8), patch_overlap=(32, 8, 4))


def test_overlap_greater_than_patch_size_raises() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        _validate_overlap(patch_size=(32, 32, 8), patch_overlap=(40, 8, 4))


def test_negative_overlap_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        _validate_overlap(patch_size=(32, 32, 8), patch_overlap=(-1, 0, 0))


def test_odd_overlap_raises() -> None:
    """TorchIO requires even overlap for symmetric tiling."""
    with pytest.raises(ValueError, match="even"):
        _validate_overlap(patch_size=(32, 32, 8), patch_overlap=(7, 4, 2))


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="3-tuples"):
        _validate_overlap(patch_size=(32, 32), patch_overlap=(8, 8))  # type: ignore[arg-type]


# ── Auto-overlap heuristic ──────────────────────────────────────────────────


def test_auto_overlap_returns_even_eighth() -> None:
    """1/8 of patch_size, rounded down to even, never above patch_size."""
    o = _auto_overlap((128, 128, 32))
    assert o == (16, 16, 4)
    # All values are even.
    assert all(v % 2 == 0 for v in o)
    # All values strictly less than patch_size.
    assert all(o[i] < (128, 128, 32)[i] for i in range(3))


def test_auto_overlap_handles_tiny_patches() -> None:
    """For patches < 8 voxels, 1/8 rounds to 0 — valid."""
    o = _auto_overlap((4, 4, 1))
    assert o == (0, 0, 0)


def test_auto_overlap_used_when_none_passed() -> None:
    """``build_grid_inference_handle(patch_overlap=None)`` triggers
    the heuristic."""
    subj, _ = _build_subject()
    handle = build_grid_inference_handle(
        subject=subj,
        patch_size=(32, 32, 8),
        patch_overlap=None,  # → auto
        aggregator="hann",
    )
    assert handle.patch_overlap == (4, 4, 0)  # 32//8=4, 8//8=1→0(rounded down to even)


# ── Constructor argument plumbing ───────────────────────────────────────────


def test_handle_records_resolved_shape_metadata() -> None:
    subj, _ = _build_subject()
    handle = build_grid_inference_handle(
        subject=subj,
        patch_size=(16, 16, 4),
        patch_overlap=(4, 4, 2),
        aggregator="hann",
    )
    assert isinstance(handle, GridInferenceHandle)
    assert handle.patch_size == (16, 16, 4)
    assert handle.patch_overlap == (4, 4, 2)
    assert handle.aggregator_mode == "hann"


def test_invalid_aggregator_kind_raises() -> None:
    subj, _ = _build_subject()
    with pytest.raises(ValueError, match="Unknown aggregator"):
        build_grid_inference_handle(
            subject=subj,
            patch_size=(16, 16, 4),
            aggregator="bogus",  # type: ignore[arg-type]
        )


def test_non_3d_patch_size_raises() -> None:
    """Temporal/4D tiling is Phase 4c, not this module."""
    subj, _ = _build_subject()
    with pytest.raises(ValueError, match="3D patch_size required"):
        build_grid_inference_handle(
            subject=subj,
            patch_size=(16, 16),  # type: ignore[arg-type]
            aggregator="hann",
        )


# ── Per-strategy opt-in audit ───────────────────────────────────────────────


def test_audit_no_op_for_non_grid_sampler() -> None:
    """``sampler.type`` other than 'grid' (or None) → audit passes
    regardless of strategy support."""

    class DiffusionLike:
        supports_grid_tiling = False

    # All three of these should be no-ops:
    check_grid_tiling_supported(DiffusionLike, "full")
    check_grid_tiling_supported(DiffusionLike, "uniform")
    check_grid_tiling_supported(DiffusionLike, None)


def test_audit_accepts_strategy_that_opts_in() -> None:
    class FastReconStrategy:
        supports_grid_tiling = True

    check_grid_tiling_supported(FastReconStrategy, "grid")  # no raise


def test_audit_rejects_strategy_that_does_not_opt_in() -> None:
    """The error must NAME the strategy and point at the fix.

    Failing fast at config-load is the CLAUDE.md #9 principle — we
    cannot silently produce wrong outputs by naively tiling a
    diffusion paradigm.
    """

    class LatentDiffusionLike:
        supports_grid_tiling = False

    with pytest.raises(ValueError) as exc_info:
        check_grid_tiling_supported(LatentDiffusionLike, "grid")
    msg = str(exc_info.value)
    assert "LatentDiffusionLike" in msg
    assert "supports_grid_tiling" in msg
    # The error tells the user how to fix it.
    assert "(a)" in msg and "(b)" in msg


def test_audit_uses_strategy_name_override() -> None:
    class LatentDiffusionLike:
        supports_grid_tiling = False

    with pytest.raises(ValueError, match="MyCustomName"):
        check_grid_tiling_supported(
            LatentDiffusionLike, "grid", strategy_name="MyCustomName"
        )


def test_list_grid_tiling_compatible_filters_classes() -> None:
    class A:
        supports_grid_tiling = True
    class B:
        supports_grid_tiling = False
    class C:
        supports_grid_tiling = True

    A.__name__ = "A"
    B.__name__ = "B"
    C.__name__ = "C"

    result = list_grid_tiling_compatible_strategies([A, B, C])
    assert result == ["A", "C"]


# ── Real-strategy compatibility table ───────────────────────────────────────


def test_real_strategies_opt_in_correctly() -> None:
    """Pin which inference strategies opt in to grid tiling — flipping
    these requires a deliberate decision (and an implementation, if
    flipping a False to True)."""
    from mriforge.infrastructure.inference.gan_inference_strategy import (
        GANInferenceStrategy,
    )
    from mriforge.infrastructure.inference.reconstruction_inference_strategy import (
        ReconstructionInferenceStrategy,
    )
    from mriforge.infrastructure.inference.vae_inference_strategy import (
        VAEInferenceStrategy,
    )
    from mriforge.infrastructure.inference.domain_adaptation_inference_strategy import (
        DomainAdaptationInferenceStrategy,
    )
    from mriforge.infrastructure.inference.latent_diffusion_inference_strategy import (
        LatentDiffusionInferenceStrategy,
    )
    from mriforge.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )
    from mriforge.infrastructure.inference.mae_inference_strategy import (
        MAEInferenceStrategy,
    )

    compatible = [
        GANInferenceStrategy,
        ReconstructionInferenceStrategy,
        VAEInferenceStrategy,
        DomainAdaptationInferenceStrategy,
    ]
    incompatible = [
        LatentDiffusionInferenceStrategy,
        ColdDiffusionInferenceStrategy,
        MAEInferenceStrategy,
    ]
    for c in compatible:
        assert c.supports_grid_tiling is True, (
            f"{c.__name__} regressed: supports_grid_tiling should be True"
        )
    for c in incompatible:
        assert c.supports_grid_tiling is False, (
            f"{c.__name__} flipped to True — confirm per-tile inference "
            "is correctly implemented before opting in."
        )
