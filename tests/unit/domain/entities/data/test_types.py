"""Tests for ``domain/entities/data/types.py`` — the optimizer-vocabulary re-export.

``OptimizerType`` here used to be an independent
``Literal["adam", "sgd", "adamw"]``: a *third* optimizer vocabulary alongside the
``OptimizerType`` enum in ``config/schemas/enums.py`` and the name table in
``OptimizerRegistry``, with none of the three designated as the SSOT. Nothing
referenced it, so the three could disagree indefinitely while each agreed with
itself — the same shape of drift that made the workflow axis-table keys dead.

These tests assert the **collapse**, not the contents: that this name now *is*
the config-layer enum object rather than a copy that happens to overlap. A test
asserting `"adam" in OptimizerType` would pass for a copy too, which is exactly
what must not be allowed back.
"""

from __future__ import annotations


class TestOptimizerTypeIsTheConfigEnum:
    def test_is_the_same_object_as_the_config_layer_enum(self) -> None:
        """Identity, not equality — a re-export cannot drift, a copy can."""
        from spectramr.config.schemas.enums import OptimizerType as CanonicalOptimizerType
        from spectramr.domain.entities.data.types import OptimizerType

        assert OptimizerType is CanonicalOptimizerType

    def test_is_still_exported(self) -> None:
        """The public alias survives the collapse (it is in ``__all__``)."""
        from spectramr.domain.entities.data import types

        assert "OptimizerType" in types.__all__
        assert getattr(types, "OptimizerType", None) is not None

    def test_carries_the_full_vocabulary_not_the_old_three(self) -> None:
        """The old Literal advertised 3 names; the enum advertises the real set."""
        from spectramr.domain.entities.data.types import OptimizerType

        values = {m.value for m in OptimizerType}
        # The three the old Literal knew about, plus names it could never express.
        assert {"adam", "sgd", "adamw"} <= values
        assert {"lars", "lamb", "lion", "sam"} <= values
        assert len(values) > 3

    def test_no_second_literal_vocabulary_remains_in_the_module(self) -> None:
        """Guard the regression: a fresh ``Literal[...]`` here would re-fork the set.

        Deliberately a source check. The defect was not a wrong value, it was the
        *existence* of a parallel definition, and no behavioural assertion can see
        that — a second Literal would agree with the enum on the day it was written.
        """
        import inspect

        from spectramr.domain.entities.data import types

        src = inspect.getsource(types)
        assert "OptimizerType = Literal[" not in src, (
            "domain/entities/data/types.py defines its own OptimizerType Literal "
            "again. The vocabulary SSOT is config/schemas/enums.OptimizerType; "
            "re-export it instead of restating it."
        )
