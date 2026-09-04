"""The stub must track the schema, or it is just a second schema.

`tests/utils/data_config_stub.py` exists because a hand-rolled stand-in silently
became a shape no real config produces, and 31 tests stayed green over code that
had stopped working. `block_config_stub` is the same guard for `validation:` and
`logging:` — the two other blocks the 15-phase decomposition split — and it was
written after the opposite failure: 19 tests went RED because their inline stubs
still spelled the leaves flat (#714).

Red is the better failure of the two, but it cost more than it looks. Two of the
19 are the only guards for the #371/#390 wrong-channel visualization family, so
a rename disarmed a live mechanism check while reading as fixture noise.

These tests pin the property that makes the module worth having: coverage is
DERIVED from the schema, so a phase that adds a sub-block cannot leave the stub
behind.
"""

from __future__ import annotations

import re

import pytest

from tests.utils.block_config_stub import (
    LoggingConfigStub,
    ValidationConfigStub,
    sub_blocks,
)


class TestCoverageIsDerivedNotListed:
    @pytest.mark.parametrize(
        ("stub", "schema_path"),
        [
            (ValidationConfigStub, "spectramr.config.schemas.validation:ValidationConfigSchema"),
            (LoggingConfigStub, "spectramr.config.schemas.logging:LoggingConfigSchema"),
        ],
    )
    def test_every_schema_sub_block_is_present(self, stub, schema_path):
        """The anti-drift property. A hand-kept list would pass today and rot."""
        import importlib

        mod, cls = schema_path.split(":")
        schema = getattr(importlib.import_module(mod), cls)
        from pydantic import BaseModel

        expected = {
            name
            for name, info in schema.model_fields.items()
            if isinstance(info.annotation, type)
            and issubclass(info.annotation, BaseModel)
        }
        assert expected, "schema exposes no sub-blocks — this test has gone vacuous"
        assert expected <= set(vars(stub()))

    def test_blocks_carry_real_schema_defaults_not_restated_ones(self):
        """Restating a default is how ``max_resident_volumes`` reached int(None)."""
        from spectramr.config.schemas.logging import LoggingSnapshotsConfigSchema

        assert (
            LoggingConfigStub().snapshots.max_calls
            == LoggingSnapshotsConfigSchema.model_fields["max_calls"].default
        )


class TestOverridesGoThroughTheRealSchema:
    def test_leaf_override_applies(self):
        cfg = LoggingConfigStub(snapshots={"enabled": False, "interval_steps": 7})
        assert cfg.snapshots.enabled is False
        assert cfg.snapshots.interval_steps == 7

    def test_the_pre_decomposition_spelling_raises(self):
        """`validation.metrics` -> `validation.scoring.compute` (phase-10a).

        The stub must REJECT the old name. Accepting it as a stray attribute is
        exactly how the flat stubs kept a retired spelling alive.
        """
        with pytest.raises(KeyError, match=re.escape("no sub-block 'metrics'")):
            ValidationConfigStub(metrics=["psnr"])

    def test_an_unknown_leaf_raises_from_the_real_schema(self):
        with pytest.raises(Exception, match=r"(?i)extra|valid"):
            ValidationConfigStub(scoring={"not_a_real_leaf": 1})

    def test_sub_blocks_accepts_a_constructed_block(self):
        from spectramr.config.schemas.validation import ValidationScoringConfigSchema

        block = ValidationScoringConfigSchema(compute=["psnr"])
        assert sub_blocks(
            ValidationConfigStub._SCHEMA, scoring=block
        )["scoring"] is block
