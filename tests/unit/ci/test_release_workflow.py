"""Structural invariants of ``.github/workflows/release.yml``.

The release lane cannot be executed here -- no runner, no PyPI, and the act it
performs is one-shot: PyPI never re-issues a filename, so a lane that publishes
the wrong thing costs a version number rather than a re-run. That makes it the
third blindness shape from ``.agent/rules/detectors.md``: a detector whose
subject cannot be driven. The response is to make the properties *decidable from
the document* and plant each one against a mutated copy -- pinning the parsed
structure, never the prose, so a reworded comment does not turn a lane red and a
reworded lane does not stay green.

Each predicate below is fed a mutated workflow in its own test. A predicate that
only ever sees the real file is a predicate nobody has watched fail.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3] / ".github" / "workflows" / "release.yml"
)

# PyYAML resolves the bare key `on` to the boolean True (YAML 1.1), so the
# trigger block is not reachable as doc["on"]. Reading it that way is how a
# trigger assertion comes back vacuously clean.
_ON = True


def _load(path: Path = WORKFLOW) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Predicates -- each returns the offending items, empty meaning clean.
# --------------------------------------------------------------------------


def oidc_granted_workflow_wide(doc: dict[str, Any]) -> bool:
    """``id-token: write`` at workflow level hands an upload credential to every job.

    Including the job that executes the build backend and everything the tree's
    build dependencies drag in. The grant belongs to the job that publishes.
    """
    return (doc.get("permissions") or {}).get("id-token") == "write"


def publishing_jobs_reachable_without_a_tag(doc: dict[str, Any]) -> list[str]:
    """Any job that can reach the index must be gated to a real tag push.

    ``workflow_dispatch`` exists so the artefacts can be rehearsed. A rehearsal
    that can publish is not a rehearsal.
    """
    publishers = {"pypi", "github-release"}
    return sorted(
        name
        for name, body in (doc.get("jobs") or {}).items()
        if name in publishers and "github.event_name == 'push'" not in str(body.get("if", ""))
    )


def verification_step_missing(doc: dict[str, Any]) -> bool:
    """The build job must *invoke* build_dist.py, not merely ship beside it.

    Non-negotiable 16: a capability is not delivered until the production path
    calls it. A lane that runs bare ``python -m build`` has every check written
    and none of them running.
    """
    steps = (doc.get("jobs", {}).get("build", {}) or {}).get("steps", [])
    return not any("build_dist.py" in str(s.get("run", "")) for s in steps)


def tag_not_pinned_to_the_artefacts(doc: dict[str, Any]) -> bool:
    """On the tag path the ref must be handed to ``--expect-version``.

    Without it the lane happily builds 0.1.0 and publishes it under a v0.1.1
    release, and that filename is then spent.
    """
    steps = (doc.get("jobs", {}).get("build", {}) or {}).get("steps", [])
    return not any("--expect-version" in str(s.get("run", "")) for s in steps)


def interpolated_run_scripts(doc: dict[str, Any]) -> list[str]:
    """``${{ }}`` inside a ``run:`` body is the script-injection seam.

    Event-supplied text is pasted into the shell before the shell sees it. The
    value belongs in ``env:``, where it arrives as data.
    """
    offenders = []
    for job_name, body in (doc.get("jobs") or {}).items():
        for i, step in enumerate(body.get("steps", []) or []):
            if "${{" in str(step.get("run", "")):
                offenders.append(f"{job_name}[{i}]")
    return offenders


# --------------------------------------------------------------------------
# The real file.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def doc() -> dict[str, Any]:
    return _load()


def test_the_workflow_parses_and_is_tag_driven(doc) -> None:
    assert "tags" in doc[_ON]["push"], "release.yml must fire on tags, not branches"
    assert "workflow_dispatch" in doc[_ON], "no way to rehearse an irreversible publish"


def test_oidc_is_not_granted_workflow_wide(doc) -> None:
    assert not oidc_granted_workflow_wide(doc)
    assert doc["jobs"]["pypi"]["permissions"]["id-token"] == "write", (
        "the publishing job still needs the token it was scoped to"
    )


def test_no_publishing_job_is_reachable_from_a_dry_run(doc) -> None:
    assert publishing_jobs_reachable_without_a_tag(doc) == []


def test_the_build_job_runs_the_verifier(doc) -> None:
    assert not verification_step_missing(doc)


def test_the_tag_is_pinned_to_the_artefacts(doc) -> None:
    assert not tag_not_pinned_to_the_artefacts(doc)


def test_no_run_script_interpolates_event_text(doc) -> None:
    assert interpolated_run_scripts(doc) == []


def test_the_verifier_the_lane_names_actually_exists(doc) -> None:
    """A path in a `run:` is a string; nothing else checks that it resolves."""
    assert (WORKFLOW.parents[2] / "scripts" / "release" / "build_dist.py").is_file()


# --------------------------------------------------------------------------
# Plants -- every predicate above, watched failing on a document that violates it.
# --------------------------------------------------------------------------


def test_plant_oidc_hoisted_to_workflow_level(doc) -> None:
    bad = copy.deepcopy(doc)
    bad["permissions"]["id-token"] = "write"
    assert oidc_granted_workflow_wide(bad)


def test_plant_a_publishing_job_with_its_gate_removed(doc) -> None:
    bad = copy.deepcopy(doc)
    del bad["jobs"]["pypi"]["if"]
    assert publishing_jobs_reachable_without_a_tag(bad) == ["pypi"]


def test_plant_a_gate_that_names_the_wrong_event(doc) -> None:
    """`if: always()` and a gate on the wrong event both read as 'a gate is present'."""
    bad = copy.deepcopy(doc)
    bad["jobs"]["github-release"]["if"] = "always()"
    assert publishing_jobs_reachable_without_a_tag(bad) == ["github-release"]


def test_plant_the_lane_reverted_to_a_bare_build(doc) -> None:
    """The shape this file exists for: `python -m build`, straight to publish."""
    bad = copy.deepcopy(doc)
    bad["jobs"]["build"]["steps"] = [{"run": "python -m build"}]
    assert verification_step_missing(bad)
    assert tag_not_pinned_to_the_artefacts(bad)


def test_plant_verification_that_runs_but_never_pins_the_tag(doc) -> None:
    """Calling the verifier without the tag is the half-wired shape, not the absent one."""
    bad = copy.deepcopy(doc)
    bad["jobs"]["build"]["steps"] = [{"run": "python scripts/release/build_dist.py"}]
    assert not verification_step_missing(bad)
    assert tag_not_pinned_to_the_artefacts(bad)


def test_plant_a_ref_interpolated_into_a_run_body(doc) -> None:
    bad = copy.deepcopy(doc)
    bad["jobs"]["build"]["steps"] = [
        {"run": "python scripts/release/build_dist.py --expect-version ${{ github.ref_name }}"}
    ]
    assert interpolated_run_scripts(bad) == ["build[0]"]


def test_plant_the_boolean_on_key_confusion() -> None:
    """`doc["on"]` is None on every workflow here -- an assertion through it is vacuous."""
    doc = _load()
    assert doc.get("on") is None
    assert isinstance(doc[_ON], dict)
