"""III.5 — Phase-Extraction Three-Tier Deployment Pipeline (ULF-PR-18)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class PhaseExtractionThreeTierPipeline(Phase2Pipeline):
    """Workflow that extracts phase at three levels of fidelity, each
    a viable deployment target depending on hardware:

    1. ``cheap_demodulation``  — coarse demodulation via measured B0
                                 (no learned model — works on any hardware).
    2. ``inverse_bloch_phase`` — multi-parameter mapping with phase
                                 residual smoothness — needs multi-contrast.
    3. ``field_probe_coupled`` — gold-standard demodulation using on-line
                                 field-probe traces. Needs probe hardware.
    """

    def __init__(self) -> None:
        super().__init__(
            name="phase_extraction_three_tier_pipeline",
            description=(
                "Three deployment tiers for ULF phase extraction — each a "
                "viable target depending on hardware: cheap demod, "
                "inverse-Bloch phase, field-probe-coupled. ULF-PR-18 / III.5."
            ),
            stages=[
                PipelineStage(
                    name="tier1_cheap_demod",
                    strategy="reconstruction",
                    config_overrides={
                        "physics.b0_correction.enabled": True,
                        "training.max_iterations": 50_000,
                    },
                ),
                PipelineStage(
                    name="tier2_inverse_bloch_phase",
                    strategy="inverse_bloch_phase",
                    depends_on=["tier1_cheap_demod"],
                ),
                PipelineStage(
                    name="tier3_field_probe_coupled",
                    strategy="field_probe_coupled",
                    depends_on=["tier2_inverse_bloch_phase"],
                ),
            ],
        )


__all__ = ["PhaseExtractionThreeTierPipeline"]
