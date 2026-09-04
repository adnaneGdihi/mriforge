"""Regression tests pinning the two-enum split for field-strength representations.

Locks ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5 + the
operator-dispatch-enum exemption in CLAUDE.md pitfall #12.

Two enums coexist because they encode different things:

- :class:`~spectramr.infrastructure.physics.tissue_properties.TissueFieldStrength`
  — string keys (``"64mT"``, ``"1.5T"``, ``"3.0T"``) into the tissue-property
  lookup table.

- :class:`~spectramr.infrastructure.physics.field_simulation.FieldStrength`
  — numeric T-values (``0.064``, ``1.5``, ``3.0``, …) used in off-resonance
  / B0 physics computations.

Phase 5 of the SSOT cleanup audit flagged these as "duplicates" — but
they aren't duplicates in semantics, just in name. This test pins the
explicit-rename design so a future contributor doesn't accidentally
reintroduce the visual collision by re-importing one as ``FieldStrength``.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.field_simulation import (
    FieldStrength as NumericFieldStrength,
)
from spectramr.infrastructure.physics.tissue_properties import TissueFieldStrength


def test_tissue_field_strength_uses_string_tags() -> None:
    """The tissue-property enum's values are string tags (lookup keys)."""
    assert TissueFieldStrength.mT64.value == "64mT"
    assert TissueFieldStrength.T1_5.value == "1.5T"
    assert TissueFieldStrength.T3_0.value == "3.0T"


def test_numeric_field_strength_carries_t_values() -> None:
    """The off-resonance enum's values are numeric T (tesla)."""
    assert NumericFieldStrength.T3_0.value == pytest.approx(3.0)
    assert NumericFieldStrength.T1_5.value == pytest.approx(1.5)
    assert NumericFieldStrength.T0_064.value == pytest.approx(0.064)


def test_two_enums_are_distinct_classes() -> None:
    """The two enums must NOT be the same class — they encode different things."""
    assert TissueFieldStrength is not NumericFieldStrength


def test_tissue_field_strength_is_not_re_exported_as_field_strength() -> None:
    """``tissue_properties.FieldStrength`` is no longer an exported symbol.

    The previous (audit-flagged) layout exposed both enums under the same
    name in adjacent modules. Catching a reintroduction of that ambiguity
    is the point of this test.
    """
    from spectramr.infrastructure.physics import tissue_properties

    assert not hasattr(tissue_properties, "FieldStrength"), (
        "tissue_properties.FieldStrength was re-introduced — "
        "use TissueFieldStrength to avoid the visual collision with "
        "field_simulation.FieldStrength."
    )


def test_tissue_property_registry_still_resolves_64mT() -> None:
    """The renamed enum still keys the registry correctly (no broken refactor)."""
    from spectramr.infrastructure.physics.tissue_properties import (
        TissuePropertyRegistry,
    )

    props = TissuePropertyRegistry.get_properties(TissueFieldStrength.mT64)
    # The registry returns a dict[TissueType, RelaxationProperties].
    assert props, "Empty registry — refactor broke the lookup."
    # Spot-check a known key from the original registry.
    from spectramr.infrastructure.physics.tissue_properties import TissueType

    assert TissueType.GM in props
