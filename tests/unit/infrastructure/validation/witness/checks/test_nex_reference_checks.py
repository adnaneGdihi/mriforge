"""The ``nex_reference_route`` witness and the predicate it shares with the corpus test.

Planted violations first (non-negotiable 15): a witness is only a gate for the
shape it has been watched to fail on.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness.checks.nex_reference_checks import (
    NEX_ROUTES,
    nex_reference_route,
    nex_reference_route_defect,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    get_witness_registry,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _subject(data: dict | None) -> WitnessSubject:
    raw = {"data": data} if data is not None else {}
    return WitnessSubject.for_ci(None, raw)


# --- the predicate -----------------------------------------------------------


@pytest.mark.parametrize(
    "route", ["kspace", "bart_kspace", "ismrmrd_kspace", "oracle_bssfp", "nifti_paired"]
)
def test_use_repetitions_true_off_route_is_a_defect(route: str) -> None:
    """The planted violation: the vf/diffusion shape (53 + 8 arms on 2026-09-02)."""
    defect = nex_reference_route_defect(route, True)
    assert defect is not None
    assert route in defect and "m4raw" in defect


def test_use_repetitions_true_on_the_m4raw_route_is_honest() -> None:
    assert nex_reference_route_defect("m4raw", True) is None


@pytest.mark.parametrize("declared", [None, False])
def test_absent_or_false_is_never_a_defect(declared) -> None:
    """None means "the route decides" (schema default); False is the declared
    identity/debug mode. Neither advertises a NEX target."""
    assert nex_reference_route_defect("kspace", declared) is None
    assert nex_reference_route_defect("m4raw", declared) is None


def test_nex_routes_is_the_advertised_set() -> None:
    """Pins the claim about ``DatasetInstantiator``: one route averages reps."""
    assert frozenset({"m4raw"}) == NEX_ROUTES


# --- the witness ---------------------------------------------------------------


def test_witness_fails_with_error_severity_on_the_planted_shape() -> None:
    verdict = nex_reference_route(_subject({"dataset_type": "kspace", "use_repetitions": True}))
    assert verdict.passed is False
    assert verdict.severity is Severity.ERROR
    assert verdict.stage is Stage.DECLARE
    assert "data.use_repetitions" in verdict.yaml_keys
    assert "m4raw" in verdict.message


def test_witness_passes_on_the_m4raw_route() -> None:
    verdict = nex_reference_route(_subject({"dataset_type": "m4raw", "use_repetitions": True}))
    assert verdict.passed is True


def test_witness_passes_when_the_knob_is_absent_and_says_the_route_decides() -> None:
    verdict = nex_reference_route(_subject({"dataset_type": "kspace"}))
    assert verdict.passed is True
    assert "route decides" in verdict.message


def test_witness_tolerates_a_config_with_no_data_block() -> None:
    assert nex_reference_route(_subject(None)).passed is True


def test_witness_is_registered_after_discovery() -> None:
    """Registered is not reachable: the package walk must have imported this
    module for a config to ever meet the witness (reachability contract)."""
    import spectramr.infrastructure.validation.witness  # noqa: F401  (triggers discovery)

    witness = get_witness_registry().get("nex_reference_route")
    assert witness is not None
    assert Subject.CONFIG in witness.subjects
    assert witness.severity is Severity.ERROR
