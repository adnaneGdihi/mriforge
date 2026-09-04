"""Every mrixfields2026 arm declares an honest, renderable reporting block.

Anti-facade guard (CLAUDE.md pitfall #16): a `reporting.figures` list must contain only
figure ids that are actually REGISTERED — no arm may advertise a figure with no producer
(the ~13 fig_c*/fig_b*/gen_vae/certificate figures that silently soft-skip). The task
preset must match the arm's cohort directory (task1/task3 -> synthesis image translation,
task2 -> reconstruction restoration, baselines -> gan).

The block is validated against ``ReportingSettings`` directly (extra="forbid" catches a
mistyped field; enum coercion catches a bad task/style), isolated from the rest of the
config so an unrelated pre-existing load issue in some arm cannot mask a reporting bug.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml as pyyaml

from spectramr.config.schemas.reporting import ReportingSettings
from tests.utils.corpus import tracked_yamls

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ARMS_DIR = _REPO_ROOT / "experiments" / "inprogress" / "mrixfields2026"
ARMS = tracked_yamls(_ARMS_DIR)

# Cohort directory -> the reporting task preset it must use.
_TASK_BY_DIR = {
    "task1": "synthesis",
    "task2": "reconstruction",
    "task3": "synthesis",
    "baselines": "gan",
}


@pytest.fixture(scope="module")
def registered_ids() -> set[str]:
    from spectramr.infrastructure.reporting.plotters import PLOTTERS

    assert PLOTTERS, "plotter registry is empty — import did not populate PLOTTERS"
    return set(PLOTTERS)


def test_arms_discovered() -> None:
    # Guard against a bad glob silently covering zero arms.
    assert len(ARMS) >= 70, f"expected the full mrixfields cohort, found {len(ARMS)}"


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_arm_has_honest_reporting_block(
    arm: pathlib.Path, registered_ids: set[str]
) -> None:
    doc = pyyaml.safe_load(arm.read_text())
    rep = doc.get("reporting")
    assert rep, f"{arm.name}: missing a reporting block"

    settings = ReportingSettings(**rep)  # schema-validates the block (raises on bad field)
    assert settings.enabled, f"{arm.name}: reporting.enabled must be true"

    expected_task = _TASK_BY_DIR.get(arm.parent.name)
    assert expected_task is not None, f"{arm.name}: unexpected cohort dir {arm.parent.name}"
    assert settings.task.value == expected_task, (
        f"{arm.name}: reporting.task={settings.task.value!r} != {expected_task!r} "
        f"(cohort dir {arm.parent.name})"
    )

    figs = settings.figures or []
    assert figs, f"{arm.name}: figures list is empty (curate the renderable set, not null)"
    unknown = [f for f in figs if f not in registered_ids]
    assert not unknown, f"{arm.name}: advertises unregistered figure ids {unknown} (facade #16)"
