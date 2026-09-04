"""Planted violations for the local required-lane runner (non-negotiable 15).

``scripts/ci/run_required_locally.py`` executes the blocking lane by hand, because
GitHub Actions is disabled on this repository and ``pr-required.yml`` therefore describes
a lane that never runs. That makes the runner **the** gate, so its own failure modes
matter more than a convenience script's would, and each is planted here rather than
demonstrated once by hand.

The plants follow ``tests/unit/architecture/test_required_lane_composition_plants.py``:
drive the pure derivation directly, one plant per *shape* a rule can take, plus a
negative control and a mutation test -- a plant no mutation kills is not a demonstration.

The shapes, and why each is separate:

``uses:``-only job
    A job that looks wired and executes nothing. It must **raise**, never report zero
    steps as green. This is the shape that lets a derivation silently go vacuous.

absent tool
    ``UNRUNNABLE``, never ``PASS``, and a non-zero exit. ``docs/known_limitations.rst``
    records the audit ladder printing a check that *declined to run* with the same green
    tick as one that passed; a runner that reported "pip-audit is not installed" as a
    pass would rebuild that blindness one level up (non-negotiable 18).

failure in a non-final step
    Must not be masked by a later success in the same job.

a job added to the workflow
    Must appear with no edit to the runner. This is the property that makes it a
    derivation rather than a transcription (non-negotiable 17).

unsupported ``if:`` / unresolvable ``${{ }}``
    Must raise. Guessing would either skip a real check or -- as actually happened while
    writing this -- pass the literal expression through, so the step failed deep inside
    ``git diff`` as ``exit status 128`` instead of as a missing value.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _REPO_ROOT / "scripts" / "ci" / "run_required_locally.py"
_REAL_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "pr-required.yml"


def _load() -> ModuleType:
    """Import the runner by path -- ``scripts/`` is not a package (non-negotiable 5)."""
    spec = importlib.util.spec_from_file_location("_lane_runner_under_test", _RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # registered before exec: dataclasses resolve here
    spec.loader.exec_module(module)
    return module


def _workflow(jobs: dict) -> dict:
    return {"name": "planted", "on": {"pull_request": None}, "jobs": jobs}


def _write(tmp_path: Path, workflow: dict) -> Path:
    path = tmp_path / "planted.yml"
    path.write_text(yaml.safe_dump(workflow, sort_keys=False))
    return path


# --------------------------------------------------------------------------- plants


def test_a_uses_only_job_raises_rather_than_reporting_zero_steps_green() -> None:
    """The vacuous-derivation shape: a job that looks wired and executes nothing."""
    mod = _load()
    planted = _workflow({
        "looks-wired": {"steps": [
            {"uses": "actions/checkout@v4"},
            {"uses": "./.github/actions/setup-env"},
        ]},
    })
    with pytest.raises(SystemExit) as excinfo:
        mod.plan_jobs(planted)
    assert "0 executable steps" in str(excinfo.value)


def test_a_provisioning_only_job_also_raises() -> None:
    """Same shape, second spelling: every ``run:`` is a ``pip install``.

    Kept separate from the ``uses:``-only plant because it reaches the raise down a
    different branch -- the steps ARE ``run:`` steps, and are removed by the
    provisioning filter rather than by the ``uses:`` filter.
    """
    mod = _load()
    planted = _workflow({"install-only": {"steps": [{"run": "pip install ruff"}]}})
    with pytest.raises(SystemExit) as excinfo:
        mod.plan_jobs(planted)
    assert "0 executable steps" in str(excinfo.value)


def test_an_empty_selection_raises_rather_than_reporting_an_empty_lane_green() -> None:
    mod = _load()
    planted = _workflow({"a": {"steps": [{"run": "true"}]}})
    with pytest.raises(SystemExit) as excinfo:
        mod.plan_jobs(planted, only={"no-such-job"})
    assert "empty lane" in str(excinfo.value)


def test_an_absent_tool_is_unrunnable_not_passed(tmp_path: Path) -> None:
    """The state distinction the whole runner exists to preserve."""
    mod = _load()
    planted = _workflow({
        "needs-a-ghost": {"steps": [
            {"run": "pip install definitely-not-a-real-tool-9f3a"},
            {"name": "would use it", "run": "true"},
        ]},
    })
    [plan] = mod.plan_jobs(planted)
    results = mod.run_job(plan, _REPO_ROOT, {"PATH": ""}, echo=False)

    assert [r.state for r in results] == [mod.UNRUNNABLE]
    assert "definitely-not-a-real-tool-9f3a" in results[0].detail
    # and it must not be reportable as success
    assert mod.report(results, allow_unrunnable=False) == 2
    assert mod.report(results, allow_unrunnable=True) == 0


def test_a_command_not_found_is_reclassified_from_fail_to_unrunnable(tmp_path: Path) -> None:
    """Exit 127 is 'could not run', which is not the same fact as 'failed'."""
    mod = _load()
    planted = _workflow({
        "ghost-command": {"steps": [{"name": "call a ghost", "run": "definitely-not-a-real-tool-9f3a"}]},
    })
    [plan] = mod.plan_jobs(planted)
    results = mod.run_job(plan, _REPO_ROOT, dict(PATH="/usr/bin:/bin"), echo=False)
    assert [r.state for r in results] == [mod.UNRUNNABLE]


def test_a_failure_in_a_non_final_step_is_not_masked_by_a_later_success() -> None:
    mod = _load()
    planted = _workflow({
        "mixed": {"steps": [
            {"name": "fails", "run": "exit 3"},
            {"name": "passes", "run": "true"},
        ]},
    })
    [plan] = mod.plan_jobs(planted)
    results = mod.run_job(plan, _REPO_ROOT, dict(PATH="/usr/bin:/bin"), echo=False)
    assert [r.state for r in results] == [mod.FAIL, mod.PASS]
    assert mod.report(results, allow_unrunnable=False) == 1


def test_a_pipeline_failure_is_not_swallowed_by_its_last_command() -> None:
    """``bash -c 'false | true'`` exits 0 without ``pipefail``; the runner sets it."""
    mod = _load()
    planted = _workflow({"piped": {"steps": [{"name": "piped", "run": "false | true"}]}})
    [plan] = mod.plan_jobs(planted)
    results = mod.run_job(plan, _REPO_ROOT, dict(PATH="/usr/bin:/bin"), echo=False)
    assert [r.state for r in results] == [mod.FAIL]


def test_a_job_added_to_the_workflow_appears_without_editing_the_runner() -> None:
    """Derivation, not transcription (non-negotiable 17)."""
    mod = _load()
    before = {p.name for p in mod.plan_jobs(_workflow({"a": {"steps": [{"run": "true"}]}}))}
    after = {p.name for p in mod.plan_jobs(_workflow({
        "a": {"steps": [{"run": "true"}]},
        "brand-new": {"steps": [{"run": "true"}]},
    }))}
    assert after - before == {"brand-new"}


def test_the_aggregator_job_is_not_executed() -> None:
    """``required`` only reads other jobs' ``result`` contexts; this runner IS that."""
    mod = _load()
    planted = _workflow({
        "a": {"steps": [{"run": "true"}]},
        "required": {"needs": ["a"], "steps": [{"run": "exit 1"}]},
    })
    assert [p.name for p in mod.plan_jobs(planted)] == ["a"]


def test_an_unsupported_if_expression_raises_rather_than_guessing() -> None:
    mod = _load()
    with pytest.raises(SystemExit) as excinfo:
        mod._should_run({"if": "github.event_name == 'push'", "run": "true"}, {})
    assert "unsupported step `if:`" in str(excinfo.value)


def test_an_unresolvable_workflow_expression_raises_rather_than_passing_it_through() -> None:
    """The bug this caught for real: the literal reached ``git diff`` as a revision."""
    mod = _load()
    with pytest.raises(SystemExit) as excinfo:
        mod._resolve_env("BASE", "${{ github.event.pull_request.base.sha }}", {})
    assert "no local equivalent" in str(excinfo.value)

    # ...and with a locally-derived value present, that value wins over the literal.
    assert mod._resolve_env("BASE", "${{ github.event.pull_request.base.sha }}",
                            {"BASE": "deadbeef"}) == "deadbeef"


# ------------------------------------------------------------------- negative control


def test_the_real_workflow_derives_cleanly() -> None:
    """Negative control: the shipped lane must produce a non-trivial plan."""
    mod = _load()
    plans = mod.plan_jobs(mod.load_workflow(_REAL_WORKFLOW))
    names = {p.name for p in plans}

    assert "required" not in names, "the aggregator must not be executed locally"
    assert {"lint-diff", "guards", "architecture", "unit-collect",
            "physics", "yaml-audit", "security"} <= names
    assert all(p.steps for p in plans)
    # the requirement inference must find the two provisioning steps the lane declares
    assert "ruff" in dict((p.name, p.requires) for p in plans)["lint-diff"]
    assert "pip-audit" in dict((p.name, p.requires) for p in plans)["security"]


def test_every_job_in_the_real_workflow_is_either_planned_or_an_aggregator() -> None:
    """No job may be dropped silently -- that is the vacuous-lane failure, one level up."""
    mod = _load()
    workflow = mod.load_workflow(_REAL_WORKFLOW)
    planned = {p.name for p in mod.plan_jobs(workflow)}
    assert set(workflow["jobs"]) - planned == set(mod.AGGREGATOR_JOBS)


# ------------------------------------------------------------------------- mutation


def test_removing_the_empty_job_guard_makes_the_uses_only_plant_go_uncaught() -> None:
    """A plant no mutation kills is not a demonstration.

    Re-derives the ``uses:``-only plant with the guard neutralised. If it still 'fails',
    the plant was being caught by something other than the rule it claims to test.
    """
    mod = _load()
    planted = _workflow({"looks-wired": {"steps": [{"uses": "actions/checkout@v4"}]}})

    real_plan_jobs = mod.plan_jobs

    def without_guard(workflow: dict, only: set[str] | None = None) -> list:
        plans = []
        for name, job in workflow["jobs"].items():
            if name in mod.AGGREGATOR_JOBS:
                continue
            plan = mod.JobPlan(name=name)
            for step in job.get("steps", []) or []:
                if "run" in step:
                    plan.steps.append(step)
            plans.append(plan)           # <-- the deleted guard
        return plans

    assert without_guard(planted)[0].steps == [], "mutant must accept the empty job"
    with pytest.raises(SystemExit):
        real_plan_jobs(planted)          # the real rule still catches it
