"""III.2 — Off-Grid Amorphous k-Space Pipeline (ULF-PR-15)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class OffGridAmorphousPipeline(Phase2Pipeline):
    """Workflow for non-Cartesian, possibly perturbed trajectories:

    1. ``inr_kspace_fit``  — coordinate-MLP fits the off-grid samples per
                             subject (requires no GT).
    2. ``bald_active_acq`` — closed-loop refinement: BALD picks the next
                             query point from the unsampled set.
    3. ``girf_calibration`` — fits a per-axis GIRF correction filter.
    """

    def __init__(self) -> None:
        super().__init__(
            name="off_grid_amorphous_pipeline",
            description=(
                "Non-Cartesian / amorphous-trajectory workflow: k-space INR "
                "fit → BALD active refinement → GIRF calibration. ULF-PR-15 / III.2."
            ),
            stages=[
                PipelineStage(
                    name="inr_kspace_fit",
                    strategy="kspace_inr",
                ),
                PipelineStage(
                    name="bald_active_acq",
                    strategy="bald",
                    config_overrides={"acquisition.active.enabled": True},
                    depends_on=["inr_kspace_fit"],
                ),
                PipelineStage(
                    name="girf_calibration",
                    strategy="girf_aware",
                    depends_on=["bald_active_acq"],
                ),
            ],
        )


__all__ = ["OffGridAmorphousPipeline"]
