"""Corpus-wide NEX-reference route guard (cohort review 2026-09-02, T0.1).

Every tracked arm under ``experiments/inprogress`` that declares
``data.use_repetitions: true`` must sit on a route that builds the
repetition-averaged target. The predicate is owned by the
``nex_reference_route`` witness; this test is only its corpus driver, so the
audit (changed arms, on the cluster) and CI (every arm, raw YAML) cannot
disagree about what a NEX declaration means.

It replaced ``test_kspace_filling_cohort_consistency::
test_nex_reference_requires_m4raw_dataset_type``, whose predicate was scoped to
one cohort while 79 arms in other cohorts carried the same defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.infrastructure.validation.witness.checks.nex_reference_checks import (
    nex_reference_route_defect,
)
from tests.utils.corpus import tracked_yamls

_ARMS = tracked_yamls("experiments/inprogress")


def _arm_id(path: Path) -> str:
    parts = path.parts
    return "/".join(parts[parts.index("inprogress") + 1 :]) if "inprogress" in parts else path.name


@pytest.mark.skipif(not _ARMS, reason="experiments/inprogress not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_nex_declaration_sits_on_a_route_that_builds_it(arm: Path) -> None:
    doc = yaml.safe_load(arm.read_text()) or {}
    data = doc.get("data") or {}
    if not isinstance(data, dict) or data.get("use_repetitions") is not True:
        pytest.skip("arm does not advertise a repetition-averaged target")
    defect = nex_reference_route_defect(data.get("dataset_type"), data.get("use_repetitions"))
    assert defect is None, f"{_arm_id(arm)}: {defect}"
