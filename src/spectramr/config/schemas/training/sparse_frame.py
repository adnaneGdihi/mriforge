"""Config schema for the sparsifying-frame coherence-optimisation strategy (A-6.5).

Read by ``SparseFrameStrategy`` via ``config.training.sparse_frame``. The headline
one-knob ablation is ``lambda_coherence`` (0 = tightness+reconstruction-only
control; >0 = also optimise mutual incoherence of the learned atoms).
``lambda_parseval`` (frame tightness) is held fixed across the ablation so the
only varied knob is the coherence penalty (#17 clean ablation).
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class SparseFrameConfig(StrictSchema):
    """Knobs for learnable-frame coherence optimisation (A-6.5 SCO-Frame)."""

    lambda_coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Weight on the mutual-coherence penalty (Frobenius energy of the off-diagonal "
            "Gram of unit-normalised atoms). 0 = tightness+reconstruction-only control; >0 = "
            "coherence-optimised treatment. Linear scale; [0.01, 0.1] is typical for frame "
            "learning (upper bound 100 guards against a divergent config)."
        ),
    )
    lambda_parseval: float = Field(
        default=0.01,
        ge=0.0,
        le=100.0,
        description=(
            "Weight of the always-on Parseval frame-tightness penalty (held fixed across the "
            "ablation). Linear scale; upper bound 100 guards against a divergent config."
        ),
    )
