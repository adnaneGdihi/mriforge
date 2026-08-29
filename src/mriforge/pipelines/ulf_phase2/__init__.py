"""ULF radiologist roadmap Phase-2 engineering pipelines (ULF-PR-14..19).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md
      §Phase 2 — Per-Problematic Engineering.

Each pipeline composes existing strategies + components into a
multi-stage deployment workflow:

- III.1 Noise-dominant            → :class:`NoiseDominantPipeline`
- III.2 Off-grid amorphous        → :class:`OffGridAmorphousPipeline`
- III.3 Severely-warped volumes   → :class:`SeverelyWarpedPipeline`
- III.4 k-space gen + seq testing → :class:`KspaceGenSequenceTestPipeline`
- III.5 Phase-extraction 3-tier   → :class:`PhaseExtractionThreeTierPipeline`
- III.6 Gradient-steering compose → :class:`GradientSteeringPipeline`
"""

from .gradient_steering import GradientSteeringPipeline
from .kspace_gen_sequence_test import KspaceGenSequenceTestPipeline
from .noise_dominant import NoiseDominantPipeline
from .off_grid_amorphous import OffGridAmorphousPipeline
from .phase_extraction_three_tier import PhaseExtractionThreeTierPipeline
from .severely_warped import SeverelyWarpedPipeline

__all__ = [
    "GradientSteeringPipeline",
    "KspaceGenSequenceTestPipeline",
    "NoiseDominantPipeline",
    "OffGridAmorphousPipeline",
    "PhaseExtractionThreeTierPipeline",
    "SeverelyWarpedPipeline",
]
