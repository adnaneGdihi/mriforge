"""PMPS Phase-0 tests — Bloch round-trip at 64 mT + concomitant correction.

These tests exercise the concomitant-aware Bloch forward pass added in
:mod:`spectramr.infrastructure.physics.differentiable_bloch` for the PMPS
plan (Phase 0 = Idea 1).

What we verify here:

* The forward signal at 64 mT is finite, non-negative-magnitude, and
  responds to changes in :math:`T_1`/:math:`T_2` in the right direction
  (longer T1 → smaller magnetisation recovery → smaller SPGR magnitude
  given fixed TR).
* The :math:`1/B_0` scaling of the concomitant phase: at fixed gradient
  amplitude and FOV the magnitude of the phase modulation at the volume
  edge should scale as :math:`1/B_0` to within numerical tolerance.
* ``include_concomitant=None`` is auto-resolved (true below 0.5 T,
  false above) so a YAML that omits it cannot silently drop the
  correction at ULF.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.differentiable_bloch import (  # noqa: E402
    DifferentiableBlochLayer,
    _resolve_include_concomitant,
)
from spectramr.infrastructure.physics.maxwell_concomitant import (  # noqa: E402
    maxwell_concomitant_field_hz,
)


@pytest.fixture
def bloch_layer() -> DifferentiableBlochLayer:
    """Concrete SPGR Bloch layer (used everywhere)."""
    return DifferentiableBlochLayer(sequence_type="SPGR")


def _phantom(t1_ms: float, t2_ms: float, pd: float = 1.0, hw: int = 16) -> torch.Tensor:
    """Build a uniform (1, 1, H, W) phantom for a given tissue."""
    return torch.full((1, 1, hw, hw), t1_ms), torch.full((1, 1, hw, hw), t2_ms), torch.full((1, 1, hw, hw), pd)


class TestConcomitantAutoResolve:
    """Schema-side guard: the auto-rule must fire below 0.5 T."""

    def test_explicit_true_is_respected(self) -> None:
        assert _resolve_include_concomitant(True, 3.0) is True

    def test_explicit_false_is_respected(self) -> None:
        assert _resolve_include_concomitant(False, 0.064) is False

    def test_auto_on_at_ulf(self) -> None:
        # Hyperfine 64 mT, M4Raw 0.3 T, 0.4 T: all should auto-on.
        for b0 in (0.064, 0.3, 0.4, 0.499):
            assert _resolve_include_concomitant(None, b0) is True, (
                f"include_concomitant should auto-on at {b0} T"
            )

    def test_auto_off_at_clinical_field(self) -> None:
        for b0 in (0.5, 1.5, 3.0, 7.0):
            assert _resolve_include_concomitant(None, b0) is False, (
                f"include_concomitant should auto-off at {b0} T"
            )

    def test_threshold_override(self) -> None:
        # User can override the auto threshold (e.g. push 0.3 T into
        # "clinical" if they have a specific concomitant-aware recon).
        assert _resolve_include_concomitant(None, 0.3, threshold_t=0.1) is False


class TestBlochAt64mT:
    """Forward signal sanity at the Hyperfine 64 mT operating point."""

    def test_forward_finite_and_nonneg_magnitude(
        self, bloch_layer: DifferentiableBlochLayer
    ) -> None:
        t1, t2, pd = _phantom(t1_ms=950.0, t2_ms=100.0)
        out = bloch_layer(
            t1_map=t1,
            t2_map=t2,
            pd_map=pd,
            tr=10.0,
            te=2.0,
            flip_angle_rad=math.radians(15.0),
            field_strength_t=0.064,  # ULF
            gradient_amplitudes_mT_per_m=(5.0, 0.0, 0.0),
            voxel_size_mm=(1.0, 1.0, 1.0),
        )
        # Concomitant auto-engages at 64 mT → complex output.
        assert out.is_complex(), "ULF Bloch output should be complex (concomitant phase applied)."
        assert torch.isfinite(out).all(), "Bloch output has NaN/Inf at 64 mT."
        assert out.abs().min() >= 0.0
        assert out.abs().max() > 0.0, "Bloch output is identically zero."

    def test_t1_monotonicity(self, bloch_layer: DifferentiableBlochLayer) -> None:
        """Longer T1 → less recovery → smaller |signal| (fixed TR=10 ms)."""
        t1_short = 300.0
        t1_long = 2000.0
        kwargs = dict(
            tr=10.0,
            te=2.0,
            flip_angle_rad=math.radians(15.0),
        )
        t1_s, t2, pd = _phantom(t1_ms=t1_short, t2_ms=100.0)
        out_s = bloch_layer(t1_map=t1_s, t2_map=t2, pd_map=pd, **kwargs)
        t1_l, _, _ = _phantom(t1_ms=t1_long, t2_ms=100.0)
        out_l = bloch_layer(t1_map=t1_l, t2_map=t2, pd_map=pd, **kwargs)
        assert out_s.abs().mean() > out_l.abs().mean(), (
            "Expected shorter T1 to produce larger SPGR magnitude."
        )


class TestConcomitantScaling:
    """Verify the 1/B0 scaling of the Maxwell concomitant phase."""

    @pytest.mark.parametrize("b0_t", [0.064, 0.3, 1.5, 3.0])
    def test_1_over_b0_scaling(self, b0_t: float) -> None:
        """`maxwell_concomitant_field_hz` should scale as 1/B0."""
        ref_b0 = 1.0
        spatial = (1, 16, 16)
        voxel = (1.0, 1.0, 1.0)
        g = (10.0, 0.0, 0.0)
        ref = maxwell_concomitant_field_hz(
            spatial_shape=spatial,
            voxel_size_mm=voxel,
            gradient_amplitudes_mT_per_m=g,
            field_strength_t=ref_b0,
        )
        scaled = maxwell_concomitant_field_hz(
            spatial_shape=spatial,
            voxel_size_mm=voxel,
            gradient_amplitudes_mT_per_m=g,
            field_strength_t=b0_t,
        )
        # Expect scaled ≈ ref * (ref_b0 / b0_t) up to float tol.
        expected = ref * (ref_b0 / b0_t)
        # Use relative tolerance keyed off the max magnitude.
        denom = max(expected.abs().max().item(), 1e-12)
        rel_err = (scaled - expected).abs().max().item() / denom
        assert rel_err < 1e-4, (
            f"1/B0 scaling violated at {b0_t} T (relative error {rel_err:.2e})."
        )

    def test_concomitant_dominant_at_64mT(
        self, bloch_layer: DifferentiableBlochLayer
    ) -> None:
        """At 64 mT the concomitant phase must be non-trivial at the edge."""
        t1, t2, pd = _phantom(t1_ms=950.0, t2_ms=100.0)
        out = bloch_layer(
            t1_map=t1,
            t2_map=t2,
            pd_map=pd,
            tr=10.0,
            te=20.0,  # exaggerate so the phase has time to accumulate
            flip_angle_rad=math.radians(15.0),
            field_strength_t=0.064,
            gradient_amplitudes_mT_per_m=(5.0, 0.0, 5.0),
            voxel_size_mm=(1.0, 1.0, 1.0),
            include_concomitant=True,
        )
        assert out.is_complex()
        # Phase at the volume corner should differ measurably from the centre.
        center_phase = torch.angle(out[0, 0, out.shape[-2] // 2, out.shape[-1] // 2])
        corner_phase = torch.angle(out[0, 0, 0, 0])
        delta = (corner_phase - center_phase).abs().item()
        assert delta > 1e-3, (
            f"Concomitant phase modulation at 64 mT corner is too small "
            f"({delta:.3e}); the correction is likely not being applied."
        )
