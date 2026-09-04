r"""Tests for the validation-badge document (ULF-PR-26 / V.3).

Targets ``spectramr.infrastructure.reporting.validation_badge``.

Categories:

- ``GateResult`` dataclass round-trips through dict
- ``ValidationBadge.from_gates`` flags missing gates explicitly
- All-passed badge is ``eligible = True``
- Any failed gate disqualifies the badge (audit-gate semantic)
- Custom ``required_gates`` list is respected (ablation experiments
  may opt out of specific gates)
- ``to_json`` produces a valid JSON document
- Non-``GateResult`` value rejected at construction
- ``DEFAULT_REQUIRED_GATES`` matches the plan's six gates
"""

from __future__ import annotations

import json

import pytest

from spectramr.infrastructure.reporting.validation_badge import (
    BadgeAuditResult,
    DEFAULT_REQUIRED_GATES,
    GateResult,
    ValidationBadge,
    validation_badge_gates_complete,
)


# ---------------------------------------------------------------------------
# DEFAULT_REQUIRED_GATES contract (matches the plan)
# ---------------------------------------------------------------------------


def test_default_required_gates_match_plan() -> None:
    """The default required-gate list matches the six the plan codifies."""
    expected = (
        "iqm",
        "downstream",
        "hallucination",
        "conformal_coverage",
        "pathology_recall",
        "pac_bayes",
    )
    assert DEFAULT_REQUIRED_GATES == expected


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


def test_gate_result_to_dict_round_trip() -> None:
    """``GateResult.to_dict`` carries all fields."""
    g = GateResult(
        passed=True, details="bound = 0.05 < target = 0.10",
        threshold=0.10, observed=0.05,
    )
    d = g.to_dict()
    assert d == {
        "passed": True,
        "details": "bound = 0.05 < target = 0.10",
        "threshold": 0.10,
        "observed": 0.05,
    }


def test_gate_result_optional_threshold_observed() -> None:
    """Threshold / observed default to ``None``."""
    g = GateResult(passed=True, details="binary check")
    assert g.threshold is None
    assert g.observed is None


# ---------------------------------------------------------------------------
# ValidationBadge — eligibility logic
# ---------------------------------------------------------------------------


def _all_pass_gates() -> dict[str, GateResult]:
    """Helper: gates dict with every required gate passing."""
    return {
        name: GateResult(passed=True, details="ok")
        for name in DEFAULT_REQUIRED_GATES
    }


def test_all_pass_eligible() -> None:
    """All required gates passing → ``eligible = True``."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    assert badge.eligible is True
    assert badge.missing_gates == ()
    assert badge.failed_gates == ()


def test_missing_required_gate_disqualifies() -> None:
    """Missing a required gate disqualifies the badge."""
    gates = _all_pass_gates()
    del gates["pac_bayes"]
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is False
    assert "pac_bayes" in badge.missing_gates


def test_failed_required_gate_disqualifies() -> None:
    """A failed required gate disqualifies the badge."""
    gates = _all_pass_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR exceeded β")
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is False
    assert badge.failed_gates == ("hallucination",)


def test_extra_gates_do_not_disqualify() -> None:
    """Extra gates beyond the required set do not affect eligibility."""
    gates = _all_pass_gates()
    gates["custom_iqm_extension"] = GateResult(passed=False, details="ignored")
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is True


def test_custom_required_gates_respected() -> None:
    """A custom required-gates tuple overrides the default set."""
    gates = {"iqm": GateResult(passed=True, details="ok")}
    badge = ValidationBadge.from_gates(
        gates, required_gates=("iqm",)
    )
    assert badge.eligible is True


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_non_dict_gates_rejected() -> None:
    """``gates`` must be a dict."""
    with pytest.raises(TypeError, match="gates"):
        ValidationBadge.from_gates(gates=[("iqm", True)])  # type: ignore[arg-type]


def test_non_gate_result_value_rejected() -> None:
    """Each gate value must be a ``GateResult``."""
    with pytest.raises(TypeError, match="GateResult"):
        ValidationBadge.from_gates(gates={"iqm": True})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dict_includes_required_meta() -> None:
    """Top-level dict has ``eligible``, ``missing_gates``, ``failed_gates``."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    d = badge.to_dict()
    assert "eligible" in d
    assert "missing_gates" in d
    assert "failed_gates" in d
    assert "gates" in d


def test_to_json_round_trips() -> None:
    """``to_json`` produces a valid JSON document with the eligibility flag."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    text = badge.to_json()
    parsed = json.loads(text)
    assert parsed["eligible"] is True
    assert "iqm" in parsed["gates"]


def test_to_json_failure_case() -> None:
    """Failure case round-trips correctly through JSON."""
    gates = _all_pass_gates()
    gates["pac_bayes"] = GateResult(passed=False, details="vacuous bound")
    badge = ValidationBadge.from_gates(gates)
    parsed = json.loads(badge.to_json())
    assert parsed["eligible"] is False
    assert parsed["failed_gates"] == ["pac_bayes"]
    assert parsed["gates"]["pac_bayes"]["passed"] is False


# ---------------------------------------------------------------------------
# `.write(path)` disk emitter + `validation_badge_gates_complete` audit hook
# ---------------------------------------------------------------------------


def _all_passing_gates() -> dict[str, GateResult]:
    return {
        name: GateResult(passed=True, details=f"{name} OK")
        for name in DEFAULT_REQUIRED_GATES
    }


def test_write_creates_json_file_with_full_payload(tmp_path) -> None:
    """``ValidationBadge.write(path)`` emits a JSON file equal to ``to_dict()``."""
    badge = ValidationBadge.from_gates(_all_passing_gates())
    out = tmp_path / "subdir" / "badge.json"
    written = badge.write(out)
    assert written == out
    assert out.exists()
    assert json.loads(out.read_text()) == badge.to_dict()


def test_audit_hook_passes_when_all_required_gates_green() -> None:
    """All required gates pass → audit hook returns ``passed=True``."""
    badge = ValidationBadge.from_gates(_all_passing_gates())
    result = validation_badge_gates_complete(badge)
    assert isinstance(result, BadgeAuditResult)
    assert result.passed is True
    assert result.missing_gates == ()
    assert result.failed_gates == ()
    assert "clinical-pilot eligible" in result.message


def test_audit_hook_flags_missing_gate() -> None:
    """Missing required gates surface in the audit result."""
    gates = _all_passing_gates()
    del gates["pac_bayes"]  # required gate omitted
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert "pac_bayes" in result.missing_gates
    assert "missing" in result.message
    assert "pac_bayes" in result.message


def test_audit_hook_flags_failed_gate() -> None:
    """Failed required gates surface in the audit result."""
    gates = _all_passing_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR > bound")
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert result.failed_gates == ("hallucination",)
    assert "failed" in result.message
    assert "hallucination" in result.message


def test_audit_hook_reports_both_missing_and_failed_gates() -> None:
    """If both missing and failed gates exist, both surface in the message."""
    gates = _all_passing_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR > bound")
    del gates["iqm"]
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert "iqm" in result.missing_gates
    assert "hallucination" in result.failed_gates
    assert "missing" in result.message
    assert "failed" in result.message
