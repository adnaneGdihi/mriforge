"""III.4 — k-Space Generation + Sequence Testing Pipeline (ULF-PR-17)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class KspaceGenSequenceTestPipeline(Phase2Pipeline):
    """Workflow that uses generative k-space to test new sequences:

    1. ``coord_kspace_gen``    — train a coordinate-conditioned k-space
                                 generator on existing acquisitions.
    2. ``sequence_simulation`` — sample synthetic acquisitions for a
                                 candidate sequence (TR/TE/FA/B0).
    3. ``downstream_eval``     — run the simulated synth through a fixed
                                 downstream-task model and report task
                                 metrics.
    """

    def __init__(self) -> None:
        super().__init__(
            name="kspace_gen_sequence_test_pipeline",
            description=(
                "Generative k-space sequence-testing workflow: train a "
                "coord-conditioned k-space generator, simulate the candidate "
                "sequence, evaluate downstream task. ULF-PR-17 / III.4."
            ),
            stages=[
                PipelineStage(
                    name="coord_kspace_gen",
                    strategy="coord_kspace_gen",
                ),
                PipelineStage(
                    name="sequence_simulation",
                    strategy="reconstruction",  # acts as inference-only forward
                    config_overrides={"training.max_iterations": 0},
                    depends_on=["coord_kspace_gen"],
                ),
                PipelineStage(
                    name="downstream_eval",
                    strategy="calibration",
                    config_overrides={
                        "certification.conformal.enabled": True,
                    },
                    depends_on=["sequence_simulation"],
                ),
            ],
        )


__all__ = ["KspaceGenSequenceTestPipeline"]
