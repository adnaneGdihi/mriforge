"""Regression test for the post-audit-04-F-S-013 cleanup.

``ConfigValidator`` used to expose three class attributes
(``REQUIRED_FIELDS`` / ``MUST_NOT_BE_NONE`` / ``LEGACY_FIELDS``) that
nothing read. ``REQUIRED_FIELDS`` still referenced the v6.0-removed
``objectives`` field, so any reader would have rejected every v6.0
config. The class attributes have been deleted to keep the validator
from advertising a contract it does not enforce (CLAUDE.md pitfall #9).

If a future change re-introduces one of those attributes, this test
flags it so the contract gets wired into ``ValidatorRegistry``
properly rather than living as dead state.
"""

from __future__ import annotations

from mriforge.infrastructure.validation.config_validation import ConfigValidator


def test_no_dead_class_attributes() -> None:
    leaked = [
        name
        for name in ("REQUIRED_FIELDS", "MUST_NOT_BE_NONE", "LEGACY_FIELDS")
        if hasattr(ConfigValidator, name)
    ]
    assert not leaked, (
        f"ConfigValidator reintroduced dead class attribute(s) {leaked}. "
        "These had zero readers and REQUIRED_FIELDS referenced the "
        "removed `objectives` field — see TODO/audit/04_schemas.md "
        "F-S-013. If a real consumer needs this contract, wire it into "
        "ValidatorRegistry, not into a class attribute."
    )
