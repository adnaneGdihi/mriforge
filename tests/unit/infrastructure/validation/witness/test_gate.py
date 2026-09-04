"""Gate behaviour: one entry point, and a crashed detector is a finding."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness import gate as gate_mod
from spectramr.infrastructure.validation.witness.gate import (
    WitnessGateError,
    assert_no_errors,
    run_witnesses,
    scheduled_witnesses,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    Witness,
    WitnessRegistry,
    WitnessVerdict,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


@pytest.fixture
def isolated_registry(monkeypatch):
    reg = WitnessRegistry()
    monkeypatch.setattr(gate_mod, "get_witness_registry", lambda: reg)
    return reg


def _w(name, fn, *, tiers=(Tier.T1,), subjects=(Subject.CONFIG,)):
    return Witness(
        name=name,
        fn=fn,
        category="test",
        stage=Stage.PARSE,
        tiers=frozenset(tiers),
        subjects=frozenset(subjects),
    )


def test_a_witness_that_raises_becomes_a_failed_verdict(isolated_registry):
    """A crashed detector means the arm is UNVERIFIED, not clean."""

    def explode(_s):
        raise RuntimeError("kaboom")

    isolated_registry.register(_w("bad", explode))
    verdicts = run_witnesses(WitnessSubject.for_ci(None, {}), tiers=frozenset({Tier.T1}))
    assert len(verdicts) == 1
    assert verdicts[0].passed is False
    assert verdicts[0].category == "witness_crash"
    assert "kaboom" in verdicts[0].message


def test_one_crash_does_not_abort_the_sweep(isolated_registry):
    def explode(_s):
        raise RuntimeError("x")

    isolated_registry.register(_w("bad", explode))
    isolated_registry.register(_w("good", lambda s: WitnessVerdict("good", True, "ok")))
    verdicts = run_witnesses(WitnessSubject.for_ci(None, {}), tiers=frozenset({Tier.T1}))
    assert {v.witness_name for v in verdicts} == {"bad", "good"}


def test_witness_needing_an_absent_subject_is_not_scheduled(isolated_registry):
    """A CI subject must never build a model just to discover it could not."""
    isolated_registry.register(_w("needs_model", lambda s: None, subjects=(Subject.MODULE_TREE,)))
    scheduled = list(
        scheduled_witnesses(WitnessSubject.for_ci(None, {}), tiers=frozenset({Tier.T1}))
    )
    assert scheduled == []


def test_tier_filter_excludes_expensive_witnesses(isolated_registry):
    isolated_registry.register(_w("cheap", lambda s: WitnessVerdict("cheap", True, "")))
    isolated_registry.register(
        _w("costly", lambda s: WitnessVerdict("costly", True, ""), tiers=(Tier.T3,))
    )
    names = {
        w.name
        for w in scheduled_witnesses(WitnessSubject.for_ci(None, {}), tiers=frozenset({Tier.T1}))
    }
    assert names == {"cheap"}


def test_a_list_returning_witness_is_flattened(isolated_registry):
    isolated_registry.register(
        _w(
            "many",
            lambda s: [WitnessVerdict("many", True, "a"), WitnessVerdict("many", True, "b")],
        )
    )
    assert len(run_witnesses(WitnessSubject.for_ci(None, {}), tiers=frozenset({Tier.T1}))) == 2


def test_assert_no_errors_raises_only_on_failing_errors():
    assert_no_errors([WitnessVerdict("w", True, "ok", severity=Severity.ERROR)])
    assert_no_errors([WitnessVerdict("w", False, "meh", severity=Severity.WARNING)])
    with pytest.raises(WitnessGateError, match="witness error"):
        assert_no_errors([WitnessVerdict("w", False, "bad", severity=Severity.ERROR)])
