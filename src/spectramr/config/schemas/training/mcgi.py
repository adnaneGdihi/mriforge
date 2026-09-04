r"""Schema for MCGI: Monotone-Contrast-Group-Invariant encoder (M2).

Defines the ``training.mcgi`` sub-block for arms selecting
:class:`~spectramr.models.encoders.mcgi_encoder.MCGIEncoder`. MCGI rides the
existing ``supervised`` paradigm -- there is no bespoke strategy, because the
invariance is *structural* (it lives in the rank transform in front of the
backbone), not something the training loop enforces.

The knobs here are the ones that decide whether the invariance claim is exact:

* ``hard_rank_eval`` -- the exact empirical-CDF rank at inference. Soft-ranking
  at inference makes the invariance approximate, which the Tier-1
  ``mcgi_invariance_declared`` check flags as an ERROR (pitfall #16: a facade
  mechanism that still carries the exactness claim).
* ``symmetrize_order_reversal`` -- adjoins the involution
  :math:`\sigma:\mathbf x\mapsto-\mathbf x`, extending invariance from
  :math:`G_+` to the full :math:`G_+\rtimes\mathbb Z_2` needed for order-reversing
  contrast pairs (T1w vs T2w). Leaving it off is legitimate for single-polarity
  cohorts, but then the arm is only :math:`G_+`-invariant.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MCGIConfig(BaseModel):
    """Hyperparameters for the monotone-contrast-group-invariant encoder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    soft_rank_temperature: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "Temperature of the differentiable soft rank used during training. "
            "Smaller is closer to the exact rank but has a sharper gradient."
        ),
    )
    symmetrize_order_reversal: bool = Field(
        default=True,
        description=(
            "Pool over x -> -x, extending invariance to G_+ x Z_2 (needed when "
            "the cohort mixes order-reversing contrasts such as T1w and T2w)."
        ),
    )
    hard_rank_eval: bool = Field(
        default=True,
        description=(
            "Use the exact hard rank at inference. Required for the exactness "
            "claim (Tier-1 mcgi_invariance_declared)."
        ),
    )
    backbone: str = Field(
        default="conv",
        description=(
            "Backbone Psi behind the rank transform. The invariance guarantee is "
            "independent of this choice -- it sits behind R."
        ),
    )


__all__ = ["MCGIConfig"]
