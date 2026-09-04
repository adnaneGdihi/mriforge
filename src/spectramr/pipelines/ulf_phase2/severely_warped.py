"""III.3 — Severely-Warped Volumes Pipeline (ULF-PR-16)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class SeverelyWarpedPipeline(Phase2Pipeline):
    """Workflow for ULF acquisitions with bulk subject motion:

    1. ``concomitant_aware_recon`` — recover from concomitant-field-induced
                                     phase errors at low B0.
    2. ``diffeomorphic_recon``     — joint reconstruction + smooth
                                     deformation field across slices.
    3. ``cycle_bloch``             — refine residual contrast warp via
                                     a digital-twin cycle.
    """

    def __init__(self) -> None:
        super().__init__(
            name="severely_warped_pipeline",
            description=(
                "ULF-warp recovery: concomitant-aware recon → diffeomorphic "
                "registration → cycle-Bloch digital twin. ULF-PR-16 / III.3."
            ),
            stages=[
                PipelineStage(
                    name="concomitant_aware_recon",
                    strategy="caur",
                    config_overrides={"physics.concomitant.enabled": True},
                ),
                PipelineStage(
                    name="diffeomorphic_recon",
                    strategy="diffeomorphic_recon",
                    depends_on=["concomitant_aware_recon"],
                ),
                PipelineStage(
                    name="cycle_bloch_refine",
                    strategy="cycle_bloch",
                    depends_on=["diffeomorphic_recon"],
                ),
            ],
        )


__all__ = ["SeverelyWarpedPipeline"]
