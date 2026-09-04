"""Tests for the low-rank + sparse (RPCA) training sub-schema.

Regression for the 2026-05-31 unwired-knob audit finding: the five RPCA
hyperparameters (tau_low_rank, tau_sparse, lambda_nuclear, lambda_sparse,
lambda_consistency) were hard-coded as ``LowRankSparseStrategy`` class
attributes with no schema, no mount on ``TrainingStrategyConfigSchema``, and
no ``_MODE_DISPATCH`` entry — so they could not be set from YAML
(CLAUDE.md pitfalls #9 / #15).

These tests assert (a) the sub-block mounts and validates, (b) a distinctive
YAML value survives validation, (c) out-of-range / extra values raise rather
than degrade, and (d) the paradigm schema is registered for dispatch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training import (
    TrainingConfigLowRankSparse,
    TrainingConfigReconstruction,
    _MODE_DISPATCH,
)
from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.low_rank_sparse import (
    LowRankSparseTrainingConfigSchema,
)


class TestLowRankSparseSchema:
    def test_defaults_equal_historical_literals(self):
        s = LowRankSparseTrainingConfigSchema()
        assert s.tau_low_rank == 0.05
        assert s.tau_sparse == 0.05
        assert s.lambda_nuclear == 1e-3
        assert s.lambda_sparse == 1e-2
        assert s.lambda_consistency == 1e-3

    def test_distinctive_value_flows_through(self):
        # The whole point of wiring the knob: a YAML value is preserved.
        s = LowRankSparseTrainingConfigSchema(lambda_nuclear=0.4242)
        assert s.lambda_nuclear == 0.4242

    @pytest.mark.parametrize(
        "field",
        [
            "tau_low_rank",
            "tau_sparse",
            "lambda_nuclear",
            "lambda_sparse",
            "lambda_consistency",
        ],
    )
    def test_negative_value_raises(self, field):
        # pitfall #9: out-of-range must raise at load time, never degrade.
        with pytest.raises(ValidationError):
            LowRankSparseTrainingConfigSchema(**{field: -1.0})

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            LowRankSparseTrainingConfigSchema(bogus_knob=1.0)

    def test_subblock_mounts_on_training_strategy_schema(self):
        assert "low_rank_sparse" in TrainingStrategyConfigSchema.model_fields
        cfg = TrainingStrategyConfigSchema(
            low_rank_sparse={"lambda_nuclear": 0.123}
        )
        assert cfg.low_rank_sparse is not None
        assert cfg.low_rank_sparse.lambda_nuclear == 0.123

    def test_mode_dispatch_registered(self):
        # ``low_rank_sparse`` must resolve to its dedicated paradigm schema in
        # the dispatch table (pre-fix it fell through to reconstruction).
        assert _MODE_DISPATCH.get("low_rank_sparse") == "TrainingConfigLowRankSparse"

    def test_paradigm_schema_is_reconstruction_subclass(self):
        # The strategy subclasses ReconstructionTrainingStrategy, so the
        # paradigm schema subclasses TrainingConfigReconstruction (and thus
        # inherits the recon-objective gating).
        assert issubclass(TrainingConfigLowRankSparse, TrainingConfigReconstruction)

    def test_paradigm_schema_validates_and_flows_subblock(self):
        # Validated directly (the way create_training_config dispatches it):
        # the typed low_rank_sparse sub-block flows through under training.
        # (objectives.reconstruction.* was removed in v6.0 — loss weights now
        # live under losses.reconstruction.* — so the paradigm schema validates
        # with no objectives block.)
        # Asserted on the schema that OWNS the mount. This previously validated
        # ``TrainingConfigLowRankSparse`` with a ``{"training": {...}}`` wrapper --
        # i.e. ``training.training`` -- a self-nesting that never existed. That
        # class declares neither ``training`` nor ``low_rank_sparse``, so under
        # ``extra="forbid"`` it could only ever raise; the knobs are mounted on
        # ``TrainingStrategyConfigSchema``, which is what ``settings.training``
        # resolves to.
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        block = TrainingStrategyConfigSchema.model_validate(
            {"low_rank_sparse": {"lambda_nuclear": 0.321}}
        )
        assert block.low_rank_sparse is not None
        assert block.low_rank_sparse.lambda_nuclear == 0.321

        # ...and the paradigm schema still constructs on its own terms, which is
        # the other half of what this test was checking.
        cfg = TrainingConfigLowRankSparse.model_validate({})
        assert isinstance(cfg, TrainingConfigLowRankSparse)
