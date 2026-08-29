"""III.1 — Noise-Dominant Reconstruction Pipeline (ULF-PR-14)."""

from __future__ import annotations

from ._base import Phase2Pipeline, PipelineStage


class NoiseDominantPipeline(Phase2Pipeline):
    """Three-stage workflow for SNR-limited ULF acquisitions:

    1. ``denoise_pretrain``   — Bloch-consistent self-supervised denoising
                                on raw k-space (no GT needed).
    2. ``noise_adaptive_recon`` — score-based recon conditioned on the
                                  per-sample noise estimate.
    3. ``conformal_calibration`` — emit a per-pixel uncertainty interval
                                   so downstream QA can flag low-SNR
                                   regions.
    """

    def __init__(self) -> None:
        super().__init__(
            name="noise_dominant_pipeline",
            description=(
                "Three-stage SNR-limited ULF workflow: Bloch-consistent "
                "denoising → noise-adaptive score recon → conformal "
                "calibration. ULF-PR-14 / III.1."
            ),
            stages=[
                PipelineStage(
                    name="denoise_pretrain",
                    strategy="bloch_consistent_denoising",
                    config_overrides={"training.max_iterations": 50_000},
                ),
                PipelineStage(
                    name="noise_adaptive_recon",
                    strategy="noise_adaptive_score",
                    config_overrides={"training.max_iterations": 100_000},
                    depends_on=["denoise_pretrain"],
                ),
                PipelineStage(
                    name="conformal_calibration",
                    strategy="calibration",
                    config_overrides={
                        "certification.conformal.enabled": True,
                        "certification.conformal.alpha": 0.1,
                    },
                    depends_on=["noise_adaptive_recon"],
                ),
            ],
        )


__all__ = ["NoiseDominantPipeline"]
