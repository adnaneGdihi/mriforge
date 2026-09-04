"""Every workflow trigger is a pull request.

Guards one policy: every workflow trigger is a pull request. The full test suite runs on
a cluster rather than on hosted runners, so a `schedule:` cron or a branch `push:` in
`.github/workflows/` spends hosted-runner minutes on work nobody asked for, on hardware
far worse suited to it.

The policy is stated here rather than cited from a page, because the page that carried
it is internal and does not ship: a failure message that names a file the reader does
not have sends them looking for it instead of at the workflow.

Two mechanics make this worth a gate rather than a convention:

* `schedule` / `push` / `workflow_dispatch` / `pull_request_target` / `issue_comment` are
  read from the **default branch**, so a cron added here is invisible on `dev` and fires
  from `main` -- and, in the other direction, `nightly.yml` carried a cron for its whole
  life and never fired once because it was never published to `main`.
* A `branches:` filter on a `pull_request` trigger matches the PR's *base*, and PRs here
  target **both** `dev` (feature work) and `main` (dependabot bumps, `dev`->`main`
  publish PRs). Pinning it to `main` disabled every check on every `dev` PR until
  2026-07-12 while the PR page rendered clean; pinning it to `dev` would do the same to
  the `main` ones. With two live bases the only safe filter is none. That invariant has
  no other home, so it lives here too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# Events that are a pull request, or are scoped to one.
_PR_EVENTS = frozenset(
    {
        "pull_request",
        "pull_request_target",
        "pull_request_review",
        "pull_request_review_comment",
    }
)

# A button in the Actions tab. Never autonomous, so it is allowed anywhere.
_MANUAL_EVENTS = frozenset({"workflow_dispatch"})

# The only non-PR events in the repo, each initiated by a human, with the reason. A new
# entry here is a deliberate exception to the policy and belongs in the same commit as
# the workflow that needs it.
_ALLOWED_NON_PR_EVENTS: dict[str, dict[str, str]] = {
    "release.yml": {
        "push": "tag-only: fires on `git push origin vX.Y.Z`, the publish step",
    },
    "test-release.yml": {
        # NOT the artefact: this lane does `pip install -e .` and `make test-release`,
        # so it exercises the checkout. release.yml is what verifies what is published.
        "push": "tag-only: runs the full test lane against the tagged source tree",
        # `release: types: [published]` was removed 2026-09-04, and this entry with
        # it -- GitHub suppresses runs triggered by a GITHUB_TOKEN-published release
        # (so it never fired on the automated path), while on a manual publish it
        # queued a second full-length run behind the tag's. `_stale_entries` is what
        # made the pair inseparable, which is the point of it.
    },
    "claude.yml": {
        "issues": "the @claude bot; the job body requires an @claude mention",
        "issue_comment": "the @claude bot; the job body requires an @claude mention",
    },
}

_ALLOWLIST = _REPO_ROOT / "scripts" / "release" / "public_allowlist.txt"

# Workflows this repository HAS but the public distribution deliberately denies.
#
# An _ALLOWED_NON_PR_EVENTS entry naming one of these is not stale in a tree that
# does not ship the workflow -- it is an entry whose workflow was removed by SCOPE.
# A bare ``path.exists()`` cannot tell the two apart, and they need opposite
# treatment: rot must fail, a scope decision must not. Absent is a state to
# report, never a state to infer.
#
# The excuse is not free-floating. Each name here must appear as a `!` denial in
# scripts/release/public_allowlist.txt, which ships in both trees -- so deleting
# the denial without deleting this entry fails, and the two cannot drift apart
# unnoticed. The excuse covers ABSENCE only: a workflow that is present is
# checked in full, exactly as if it were not listed here.
_NOT_DISTRIBUTED: dict[str, str] = {
    "claude.yml": (
        "issue_comment-triggered and consumes secrets.CLAUDE_CODE_OAUTH_TOKEN; a "
        "long-lived OAuth token behind a comment trigger is an exposure path on a "
        "public repository, and the secret does not exist there."
    ),
}


def _denied_in_public_allowlist(allowlist_text: str) -> set[str]:
    """Workflow basenames the public allowlist denies with a `!` line."""
    denied: set[str] = set()
    for raw in allowlist_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("!.github/workflows/"):
            denied.add(line.rsplit("/", 1)[-1])
    return denied


def _stale_entries(
    allowed: dict[str, dict[str, str]],
    workflow_dir: Path,
    not_distributed: dict[str, str],
    denied: set[str],
) -> list[str]:
    """The pure core of the stale check, so it can be planted against directly."""
    stale: list[str] = []
    for name, events in allowed.items():
        path = workflow_dir / name
        if not path.exists():
            if name not in not_distributed:
                stale.append(f"{name}: workflow no longer exists")
            elif name not in denied:
                stale.append(
                    f"{name}: excused by _NOT_DISTRIBUTED but the public allowlist "
                    f"does not deny it -- the excuse names a scope decision that "
                    f"scripts/release/public_allowlist.txt no longer makes"
                )
            continue
        declared = _triggers(path)
        stale.extend(
            f"{name}: allowlists `{event}`, which the workflow no longer declares"
            for event in events
            if event not in declared
        )
    return stale


# The public distribution's workflows are stored HERE, not in .github/workflows/,
# so that GitHub does not also run them against this repository. That storage
# location put them outside this file's scan root -- and a scan root is an
# unaudited constant: no violation planted in .github/workflows/ can reveal that
# a second directory of workflows exists and is unchecked. The `branches:` filter
# this module exists to forbid would have been invisible in exactly the files
# that get published, until someone read the exported tree.
_OVERLAY_WORKFLOW_DIR = (
    _REPO_ROOT / "scripts" / "release" / "public_overlay" / ".github" / "workflows"
)


def _workflow_paths() -> list[Path]:
    """Every workflow this repository OWNS, wherever it is stored.

    Both directories, deliberately. The overlay copies are published verbatim, so
    the policy applies to them identically; the only difference is which repo
    runs them.
    """
    dirs = [_WORKFLOW_DIR, _OVERLAY_WORKFLOW_DIR]
    paths = sorted(
        p for d in dirs if d.is_dir() for ext in ("*.yml", "*.yaml") for p in d.glob(ext)
    )
    assert paths, f"no workflows found under any of {dirs}"
    return paths


def _triggers(path: Path) -> dict[str, Any]:
    """The `on:` block, normalised to {event: config}.

    YAML 1.1 resolves the bare key `on` to the boolean `True`, so a plain `doc["on"]`
    raises KeyError on every workflow in the repo. Read both spellings.
    """
    doc = yaml.safe_load(path.read_text())
    block = doc.get("on", doc.get(True))
    assert block is not None, f"{path.name} has no `on:` block"
    if isinstance(block, str):
        return {block: None}
    if isinstance(block, list):
        return dict.fromkeys(block)
    return block


# ``.github/`` is a dropped root in the public export -- Workstream F ports the
# two PR workflows into the new repo rather than exporting this tree's ten.
# Without this skip, ``_workflow_paths()`` below raises during IMPORT, and a
# collection error is not a test failure: it aborts the whole pytest session
# with "Interrupted: 1 error during collection", so ``pytest tests/`` in the
# export collects NOTHING AT ALL. One shipped file made the entire suite
# unrunnable, and the workaround was an ``--ignore`` flag a public user has no
# way to know they need.
#
# The distinction the assert in ``_workflow_paths`` draws is preserved rather
# than weakened: a MISSING directory is a scope decision and skips visibly; a
# PRESENT but empty one is a defect and still raises.
if not _WORKFLOW_DIR.is_dir():
    pytest.skip(
        f"no workflow directory at {_WORKFLOW_DIR}: .github/ is not part of this tree",
        allow_module_level=True,
    )

_WORKFLOWS = _workflow_paths()
# Not p.name: both directories hold a `pr-required.yml`, and two parametrised
# cases sharing an id are indistinguishable in a failure report.
_IDS = [
    (f"overlay:{p.name}" if p.is_relative_to(_OVERLAY_WORKFLOW_DIR) else p.name) for p in _WORKFLOWS
]


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_no_workflow_runs_on_a_schedule(path: Path) -> None:
    assert "schedule" not in _triggers(path), (
        f"{path.name} declares a `schedule:` cron. CI here is pull-request-only, and "
        f"the full suite runs on a cluster rather than on hosted runners. Use "
        f"`workflow_dispatch:` if it needs to be runnable on demand."
    )


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_push_triggers_are_tag_only(path: Path) -> None:
    push = _triggers(path).get("push")
    if push is None:
        return
    allowed = _ALLOWED_NON_PR_EVENTS.get(path.name, {})
    assert "push" in allowed, (
        f"{path.name} triggers on `push:`. Only the tag-driven release lanes may, and "
        f"they are listed in _ALLOWED_NON_PR_EVENTS."
    )
    assert isinstance(push, dict) and "tags" in push and "branches" not in push, (
        f"{path.name} triggers on a branch push ({push!r}). A push trigger is allowed "
        f"only when scoped to tags -- a branch push is autonomous CI."
    )


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_non_pull_request_triggers_are_allowlisted(path: Path) -> None:
    allowed = _ALLOWED_NON_PR_EVENTS.get(path.name, {})
    unexpected = {
        event
        for event in _triggers(path)
        if event not in _PR_EVENTS | _MANUAL_EVENTS and event not in allowed
    }
    assert not unexpected, (
        f"{path.name} triggers on {sorted(unexpected)}, which is neither a pull-request "
        f"event nor an allowlisted human-initiated one. Every trigger here is a pull "
        f"request; extend _ALLOWED_NON_PR_EVENTS only for an event a human starts."
    )


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_pull_request_triggers_have_no_branches_filter(path: Path) -> None:
    """The 2026-07-12 invariant: a `branches:` filter matches the PR's BASE branch."""
    for event in ("pull_request", "pull_request_target"):
        config = _triggers(path).get(event)
        if not isinstance(config, dict):
            continue
        assert "branches" not in config, (
            f"{path.name} filters `{event}` on branches={config['branches']!r}. That "
            f"matches the PR's BASE branch, and PRs here target BOTH `dev` and `main` "
            f"-- pinning it to `main` is what silently disabled every check until "
            f"2026-07-12, and pinning it to `dev` would do the same to the `main` PRs. "
            f"A skipped workflow renders as no check at all, not as a red one."
        )


def test_allowlist_has_no_stale_entries() -> None:
    """An exception that no workflow still needs is a licence nobody is using."""
    allowlist_text = _ALLOWLIST.read_text(encoding="utf-8") if _ALLOWLIST.exists() else ""
    stale = _stale_entries(
        _ALLOWED_NON_PR_EVENTS,
        _WORKFLOW_DIR,
        _NOT_DISTRIBUTED,
        _denied_in_public_allowlist(allowlist_text),
    )
    assert not stale, "stale _ALLOWED_NON_PR_EVENTS entries:\n  " + "\n  ".join(stale)


def test_every_not_distributed_entry_is_denied_by_the_public_allowlist() -> None:
    """The excuse and the denial are one decision; neither may move alone."""
    assert _ALLOWLIST.exists(), f"{_ALLOWLIST} is missing -- the excuse cannot be checked"
    denied = _denied_in_public_allowlist(_ALLOWLIST.read_text(encoding="utf-8"))
    missing = sorted(set(_NOT_DISTRIBUTED) - denied)
    assert not missing, (
        "_NOT_DISTRIBUTED names workflows the public allowlist does not deny: " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Plants against the stale check itself (non-negotiable 15).
#
# The check grew a branch that EXCUSES an absence, and an excuse is the one kind
# of change that makes a detector quieter rather than louder. So each rule and
# each shape it can take is planted here and required to turn the core red --
# committed as tests, not run once by hand. The negative controls matter equally:
# a check that fires on everything is as useless as one that fires on nothing.
# ---------------------------------------------------------------------------

_PR_ONLY_WORKFLOW = "on:\n  pull_request:\n"
_TAG_PUSH_WORKFLOW = "on:\n  push:\n    tags: ['v*']\n"


def _workflows(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    for name, body in files.items():
        (d / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return d


def test_an_absent_workflow_with_no_excuse_is_stale(tmp_path: Path) -> None:
    """The original rule. Deleting a workflow must not leave its licence behind."""
    stale = _stale_entries({"ghost.yml": {"push": "why"}}, _workflows(tmp_path), {}, set())
    assert stale == ["ghost.yml: workflow no longer exists"]


def test_an_excused_absence_that_the_allowlist_does_not_deny_is_stale(
    tmp_path: Path,
) -> None:
    """The excuse is anchored to the denial: remove the denial and it fails."""
    stale = _stale_entries(
        {"ghost.yml": {"push": "why"}},
        _workflows(tmp_path),
        {"ghost.yml": "scope decision"},
        denied=set(),
    )
    assert len(stale) == 1
    assert "the public allowlist does not deny it" in stale[0]


def test_an_excused_absence_that_the_allowlist_denies_is_clean(tmp_path: Path) -> None:
    """Negative control: the whole point of the branch is that this case passes."""
    assert (
        _stale_entries(
            {"ghost.yml": {"push": "why"}},
            _workflows(tmp_path),
            {"ghost.yml": "scope decision"},
            denied={"ghost.yml"},
        )
        == []
    )


def test_the_excuse_does_not_leak_into_a_present_workflow(tmp_path: Path) -> None:
    """The excuse covers ABSENCE only.

    This is the shape that would make the change a net loss: a name listed in
    _NOT_DISTRIBUTED that is nonetheless PRESENT must be checked exactly as if it
    were not listed at all, or listing a workflow there would silently stop
    auditing its triggers in the tree that actually runs it.
    """
    stale = _stale_entries(
        {"ghost.yml": {"issue_comment": "why"}},
        _workflows(tmp_path, ghost__yml=_PR_ONLY_WORKFLOW),
        {"ghost.yml": "scope decision"},
        denied={"ghost.yml"},
    )
    assert len(stale) == 1
    assert "no longer declares" in stale[0]


def test_a_present_workflow_that_still_declares_its_event_is_clean(
    tmp_path: Path,
) -> None:
    """Negative control for the branch above."""
    assert (
        _stale_entries(
            {"ghost.yml": {"push": "tag-only"}},
            _workflows(tmp_path, ghost__yml=_TAG_PUSH_WORKFLOW),
            {},
            set(),
        )
        == []
    )


def test_denial_parser_reads_a_real_denial_and_ignores_a_commented_one() -> None:
    """Both shapes, because the allowlist is a commented file by construction.

    A denial carrying a trailing comment must still count, and a denial that is
    itself commented out must not -- the second is how a reverted decision would
    otherwise keep excusing an entry forever.
    """
    denied = _denied_in_public_allowlist(
        "\n".join(
            [
                "!.github/workflows/real.yml",
                "!.github/workflows/trailing.yml   # with a reason",
                "# !.github/workflows/commented.yml",
                "  !.github/workflows/indented.yml",
                ".github/                          # an ALLOW, not a denial",
                "!tests/unit/ci/not_a_workflow.py",
            ]
        )
    )
    assert denied == {"real.yml", "trailing.yml", "indented.yml"}


def test_the_overlay_workflows_are_actually_scanned() -> None:
    """Anti-vacuity for the widened scan root.

    Without this, deleting `_OVERLAY_WORKFLOW_DIR` from `_workflow_paths` leaves
    every test above green -- the parametrised cases simply stop existing, and a
    test that no longer runs reports nothing at all. Assert the published
    workflows are in the set, by name.
    """
    if not _OVERLAY_WORKFLOW_DIR.is_dir():
        pytest.skip(f"{_OVERLAY_WORKFLOW_DIR} is absent -- not the tree that owns it")
    scanned = {p for p in _WORKFLOWS if p.is_relative_to(_OVERLAY_WORKFLOW_DIR)}
    on_disk = {p for ext in ("*.yml", "*.yaml") for p in _OVERLAY_WORKFLOW_DIR.glob(ext)}
    assert on_disk, f"{_OVERLAY_WORKFLOW_DIR} exists but holds no workflow"
    assert scanned == on_disk


# --------------------------------------------------------------------------- #
# A badge is a claim about a workflow, and nothing was checking it.
#
# README.md carried an Actions badge for `test.yml` -- a workflow that does not
# exist and never has. GitHub renders such a badge as a grey "no status" image
# rather than an error, so it reads as "CI configured" indefinitely. The workflow
# names are right here, so the claim is checkable.
# --------------------------------------------------------------------------- #

_BADGE_WORKFLOW = re.compile(r"/actions/workflows/([A-Za-z0-9._-]+\.ya?ml)")


def _workflows_named_in(text: str) -> set[str]:
    return set(_BADGE_WORKFLOW.findall(text))


def test_every_workflow_a_readme_badge_names_actually_exists() -> None:
    readme = _REPO_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover - README always ships
        pytest.skip("no README.md in this tree")
    named = _workflows_named_in(readme.read_text(encoding="utf-8"))
    have = {p.name for p in _WORKFLOWS}
    missing = sorted(named - have)
    assert not missing, (
        "README.md links an Actions badge to workflow(s) that do not exist: "
        + ", ".join(missing)
        + ". GitHub renders a badge for a missing workflow as a grey 'no status' "
        "image, not as an error, so it reads as working CI forever."
    )


def test_the_badge_scan_finds_the_badge_that_is_there() -> None:
    """Anti-vacuity: a regex that matches nothing passes the test above."""
    readme = _REPO_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover
        pytest.skip("no README.md in this tree")
    assert _workflows_named_in(readme.read_text(encoding="utf-8")), (
        "no Actions badge found in README.md -- either the badge was removed, or "
        "the pattern stopped matching and the check above is now vacuous"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://github.com/o/r/actions/workflows/ci.yml/badge.svg", {"ci.yml"}),
        ("…/actions/workflows/pr-required.yaml)", {"pr-required.yaml"}),
        ("…/actions/workflows/a.yml … /actions/workflows/b.yml", {"a.yml", "b.yml"}),
        ("https://github.com/o/r/actions", set()),
        ("see .github/workflows/ci.yml in the tree", set()),
    ],
)
def test_the_badge_pattern_reads_a_url_and_not_prose(text: str, expected: set[str]) -> None:
    """Both directions. The last case is the one that would cause false alarms:
    a prose mention of a workflow path is not a badge and must not be read as one.
    """
    assert _workflows_named_in(text) == expected
