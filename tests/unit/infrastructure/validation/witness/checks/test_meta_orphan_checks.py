"""The orphan detector, against the real 126-method ConfigHealthChecker.

Sensitivity pair: it must pass on the real class (every check either invoked or
allowlisted) AND fire when a check is defined but not called. A detector only
ever tested against a healthy input is indistinguishable from one that never
fires -- which is how the #550 gate shipped.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks.meta_orphan_checks import (
    KNOWN_INERT,
    defined_check_methods,
    health_checker_has_no_orphan_checks,
    invoked_check_methods,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def test_the_real_health_checker_has_no_unexplained_orphans():
    """CONTROL: green on the real class."""
    verdict = health_checker_has_no_orphan_checks(WitnessSubject.for_ci(None, {}))
    assert verdict.passed, verdict.message


def test_it_actually_parsed_a_realistic_number_of_checks():
    """Guards the AST walk itself: 0 invoked would pass vacuously.

    If `invoked_check_methods` silently returned an empty set, every check would
    look orphaned; if `defined_check_methods` returned empty, nothing would. Both
    failure modes are caught by asserting the magnitudes are plausible.
    """
    defined, invoked = defined_check_methods(), invoked_check_methods()
    assert len(defined) > 100, f"only found {len(defined)} check_* methods"
    assert len(invoked & defined) > 100, "the AST walk found almost no wired calls"


def test_every_allowlist_entry_is_a_real_method():
    """A stale allowlist hides the next real orphan."""
    defined = defined_check_methods()
    for name in KNOWN_INERT:
        assert name in defined, f"KNOWN_INERT names {name!r}, which does not exist"


def test_it_fires_when_a_check_is_defined_but_never_invoked(monkeypatch):
    """DEFECT: inject an orphan and require the detector to notice."""
    from spectramr.infrastructure.validation.witness.checks import meta_orphan_checks as m

    monkeypatch.setattr(m, "defined_check_methods", lambda: {"check_a", "check_ghost"})
    monkeypatch.setattr(m, "invoked_check_methods", lambda: {"check_a"})
    monkeypatch.setattr(m, "KNOWN_INERT", {})

    verdict = m.health_checker_has_no_orphan_checks(WitnessSubject.for_ci(None, {}))
    assert verdict.passed is False
    assert "check_ghost" in verdict.message
    assert verdict.class_ids == ("S11.3",)


def test_it_warns_when_the_allowlist_goes_stale(monkeypatch):
    from spectramr.infrastructure.validation.witness.checks import meta_orphan_checks as m

    monkeypatch.setattr(m, "defined_check_methods", lambda: {"check_a"})
    monkeypatch.setattr(m, "invoked_check_methods", lambda: {"check_a"})
    monkeypatch.setattr(m, "KNOWN_INERT", {"check_a": "reason"})

    verdict = m.health_checker_has_no_orphan_checks(WitnessSubject.for_ci(None, {}))
    assert verdict.passed is False
    assert str(verdict.severity) == "warning"
