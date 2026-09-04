"""Low-Rank + Sparse (RPCA) training sub-schema.

Declarative config for ``training_mode: low_rank_sparse`` — the
differentiable Robust-PCA splitting ``X = L + S`` performed by
:class:`~spectramr.infrastructure.training.strategies.low_rank_sparse_strategy.LowRankSparseStrategy`.

This module defines the self-contained ``training.low_rank_sparse`` sub-block
that the strategy reads at construction time
(``config.training.low_rank_sparse.lambda_nuclear`` etc.). It is mounted on
:class:`~spectramr.config.schemas.training.base.TrainingStrategyConfigSchema`
(mirrors the SSDU / CS-MNO wiring) and must stay import-cycle-free: it imports
nothing from ``base`` / ``reconstruction``.

The top-level paradigm schema selected by ``create_training_config`` —
``TrainingConfigLowRankSparse`` — lives in this package's ``__init__`` (where
``TrainingConfigReconstruction`` is already imported) to avoid a
base ← low_rank_sparse ← reconstruction ← base cycle.

The five RPCA knobs (``tau_low_rank``, ``tau_sparse``, ``lambda_nuclear``,
``lambda_sparse``, ``lambda_consistency``) default to the values the strategy
previously hard-coded as class attributes, so behaviour is unchanged when no
YAML block is supplied. Setting any of them in YAML now flows through to the
strategy (CLAUDE.md pitfall #15 — every advertised knob is read, validated,
and stamped into provenance). Out-of-range values raise at load time
(pitfall #9 — no silent degradation to a default).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LowRankSparseTrainingConfigSchema(BaseModel):
    """RPCA (low-rank + sparse) decomposition hyperparameters.

    Frozen, ``extra="forbid"`` sub-block read by ``LowRankSparseStrategy``.
    Defaults equal the strategy's historical hard-coded class attributes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    tau_low_rank: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Singular-value soft-threshold (SVT) applied to the temporal "
            "stack to extract the low-rank component L."
        ),
    )
    tau_sparse: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Element-wise soft-threshold applied to the residual X - L to "
            "extract the sparse component S."
        ),
    )
    lambda_nuclear: float = Field(
        default=1e-3,
        ge=0.0,
        description="Weight on the nuclear-norm surrogate term (sum of thresholded singular values).",
    )
    lambda_sparse: float = Field(
        default=1e-2,
        ge=0.0,
        description="Weight on the L1 (sparsity) term over the sparse component S.",
    )
    lambda_consistency: float = Field(
        default=1e-3,
        ge=0.0,
        description="Weight on the Frobenius consistency term ||X - (L + S)||_F^2.",
    )


__all__ = ["LowRankSparseTrainingConfigSchema"]
