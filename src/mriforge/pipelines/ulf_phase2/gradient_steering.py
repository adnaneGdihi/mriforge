"""III.6 — Gradient-Steering Forward-Operator Composition (ULF-PR-19)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class GradientSteeringPipeline(Phase2Pipeline):
    """Workflow for ULF systems with high gradient nonlinearity:

    1. ``girf_aware``          — fit a per-axis GIRF correction filter.
    2. ``concomitant_aware``   — apply the Maxwell phase correction at
                                 low B0.
    3. ``codesign_pilot``      — joint-optimise the trajectory under the
                                 corrected forward operator (PILOT).

    Composes three forward-operator corrections that are individually
    well-defined; the pipeline builds the operator once and reuses it.
    """

    def __init__(self) -> None:
        super().__init__(
            name="gradient_steering_pipeline",
            description=(
                "Gradient-steering forward-operator composition: GIRF + "
                "concomitant + codesign-PILOT. ULF-PR-19 / III.6."
            ),
            stages=[
                PipelineStage(
                    name="girf_calibration",
                    strategy="girf_aware",
                ),
                PipelineStage(
                    name="concomitant_aware_recon",
                    strategy="caur",
                    config_overrides={"physics.concomitant.enabled": True},
                    depends_on=["girf_calibration"],
                ),
                PipelineStage(
                    name="codesign_pilot",
                    strategy="pilot",
                    config_overrides={"acquisition.codesign.enabled": True},
                    depends_on=["concomitant_aware_recon"],
                ),
            ],
        )


__all__ = ["GradientSteeringPipeline"]
