"""Pairs with ``scripts/release/zenodo_deposit.py`` and the ``.zenodo.json`` it reads.

Three invariants live here, and each one exists because the obvious way to check it
does not work:

1. **``.zenodo.json`` must not declare a version.**  ``scripts/release/build_dist.py``
   reconciles four version declarations and refuses a release when they disagree; a
   fifth one in this file would be outside that reconciliation, so it would be the
   one nobody checks.  The deposit script reads the version *through* build_dist
   instead.
2. **The README's DOI badge line must be exactly what the deposit script prints.**
   Two files spelling the same URL is the classic second owner: both look right and
   the divergence surfaces as a badge that renders dead rather than as an error.
   ``zenodo_deposit.doi_badge`` is elected owner of the shape and this module is the
   ratchet.
3. **The repo-id badge form must stay out of the README.**  It is the shape that
   looks filled and is dead here -- probed 2026-09-03, ``/badge/1347566284.svg`` and
   ``/badge/latestdoi/1347566284`` both 404, because that endpoint is registered by
   Zenodo's GitHub integration, which cannot see a private repository and only fires
   on a published release.  A future reader "restoring" it would not find out by
   looking at an HTTP status.

The script is loaded **by file path**, not with ``from scripts.release import ...``:
``tests/unit/scripts`` is a regular package and shadows the root ``scripts``
directory, so the import form resolves to the wrong tree.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "release" / "zenodo_deposit.py"
METADATA_FILE = REPO_ROOT / ".zenodo.json"
CITATION_FILE = REPO_ROOT / "CITATION.cff"
README_FILE = REPO_ROOT / "README.md"
ALLOWLIST = REPO_ROOT / "scripts" / "release" / "public_allowlist.txt"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("zenodo_deposit_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load()


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(METADATA_FILE.read_text())


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load(CITATION_FILE.read_text())


# --------------------------------------------------------------------------- 1


def test_the_shipped_metadata_file_builds(mod, raw):
    built = mod.build_metadata(raw, "1.2.3")
    assert built["version"] == "1.2.3"
    for key in mod.REQUIRED_METADATA:
        assert built[key], f"{key} empty in .zenodo.json"


@pytest.mark.parametrize("missing", ["title", "upload_type", "creators", "description", "license"])
def test_a_missing_required_key_is_rejected(mod, raw, missing):
    """Planted violation: Zenodo accepts an incomplete draft and fails at publish."""
    planted = copy.deepcopy(raw)
    del planted[missing]
    # match= on the FIELD NAME, not on the explanatory sentence: a reworded message
    # must not break the pin, and the field name is the stable part of the contract.
    with pytest.raises(RuntimeError, match=missing):
        mod.build_metadata(planted, "1.2.3")


def test_a_version_key_in_the_metadata_file_is_rejected(mod, raw):
    """Planted violation: a fifth version declaration build_dist.py cannot see."""
    planted = copy.deepcopy(raw)
    planted["version"] = "9.9.9"
    with pytest.raises(RuntimeError, match="version"):
        mod.build_metadata(planted, "1.2.3")


def test_the_shipped_file_does_not_declare_a_version(raw):
    assert "version" not in raw, (
        "`.zenodo.json` declares a version; build_dist.py reconciles four "
        "declarations and would not see this one."
    )


# --------------------------------------------------------------------------- 2


def test_the_readme_carries_exactly_the_generated_badge_line(mod):
    expected = mod.doi_badge(mod.PLACEHOLDER_DOI)
    assert expected in README_FILE.read_text(), (
        f"README.md does not contain the line zenodo_deposit.py prints:\n  {expected}"
    )


def test_a_drifted_badge_shape_is_detected(mod):
    """Planted violation: the two owners disagreeing is exactly what must fail."""
    drifted = mod.doi_badge(mod.PLACEHOLDER_DOI).replace("zenodo.org", "example.org")
    assert drifted not in README_FILE.read_text()


def test_the_concept_doi_wins_over_the_version_doi(mod):
    line = mod.report_badge({"doi": "10.5281/zenodo.99", "conceptdoi": "10.5281/zenodo.98"})
    assert "zenodo.98" in line and "zenodo.99" not in line, (
        "a README badge pinned to a version DOI stops following later releases"
    )


def test_a_draft_without_a_concept_doi_falls_back_and_says_so(mod, capsys):
    line = mod.report_badge({"doi": "10.5281/zenodo.99"})
    assert "zenodo.99" in line
    assert "concept" in capsys.readouterr().err.lower()


def test_no_doi_at_all_reports_rather_than_inventing_one(mod):
    assert mod.report_badge({}) == ""


# --------------------------------------------------------------------------- 3


def test_the_dead_repo_id_badge_form_is_absent_from_the_readme():
    text = README_FILE.read_text()
    # The README *discusses* these endpoints, host-less and in backticks, so that a
    # future reader learns they were probed. What must not appear is either one as a
    # live URL with the host attached -- that is the form a browser would fetch.
    for dead in ("https://zenodo.org/badge/1347566284", "https://zenodo.org/badge/latestdoi/"):
        assert dead not in text, f"{dead} 404s for this repository -- see this module's docstring"


def test_the_readme_still_explains_why_that_form_is_absent():
    """The measurement must stay next to the decision, or it gets 'restored'."""
    text = README_FILE.read_text()
    assert "1347566284" in text, "the probed repo id is the evidence; keep it"
    assert "integration" in text.lower()


# --------------------------------------------------------------------------- agreement


def test_zenodo_metadata_agrees_with_the_citation_file(raw, citation):
    """One release, one set of facts, declared in two formats Zenodo and GitHub read."""
    assert raw["title"] == citation["title"]

    author = citation["authors"][0]
    assert raw["creators"][0]["name"] == f"{author['family-names']}, {author['given-names']}"

    # Zenodo wants the bare identifier; CFF wants the resolvable URL. Normalizing
    # here rather than storing one form twice is the point -- the two schemas
    # genuinely disagree about the representation, not about the person.
    assert raw["creators"][0]["orcid"] == author["orcid"].rsplit("/", 1)[-1]

    assert raw["license"] == citation["license"].lower()
    assert set(raw["keywords"]) == set(citation["keywords"])

    urls = [r["identifier"] for r in raw["related_identifiers"]]
    assert citation["repository-code"] in urls


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("title", lambda d: d.update(title="Something Else")),
        ("license", lambda d: d.update(license="mit")),
        ("orcid", lambda d: d["creators"][0].update(orcid="0000-0000-0000-0000")),
        ("keywords", lambda d: d.update(keywords=["unrelated"])),
    ],
)
def test_a_disagreement_with_the_citation_file_is_detected(raw, citation, field, mutate):
    """Planted violation, one per compared field: the agreement test must be able
    to fail. A test that only ever runs against agreeing inputs proves nothing."""
    planted = copy.deepcopy(raw)
    mutate(planted)
    author = citation["authors"][0]
    agrees = (
        planted["title"] == citation["title"]
        and planted["license"] == citation["license"].lower()
        and planted["creators"][0]["orcid"] == author["orcid"].rsplit("/", 1)[-1]
        and set(planted["keywords"]) == set(citation["keywords"])
    )
    assert not agrees, f"perturbing {field} did not break agreement"


# --------------------------------------------------------------------------- export


@pytest.mark.parametrize("path", [".zenodo.json", "conda/"])
def test_the_new_release_paths_are_named_by_the_export_allowlist(path):
    """The allowlist is fail-closed: it ships nothing it does not name.

    Line-anchored, because a commented-out entry still contains the substring --
    the mistake this test was written with, caught by planting `#` in front of the
    real line and watching nothing go red.
    """
    lines = [ln.split("#")[0].strip() for ln in ALLOWLIST.read_text().splitlines()]
    assert path in lines, f"{path} is not an active allowlist entry"


@pytest.mark.parametrize("path", [".zenodo.json", "conda/meta.yaml"])
def test_the_new_release_paths_are_tracked_by_git(path):
    """An allowlist entry for an untracked path is a DEAD ALLOWANCE that
    `export_public_tree.py --strict` exits 2 on -- and `.zenodo.json` is caught by
    the blanket `*.json` rule in `.gitignore`, so it needs a negation to be
    trackable at all.

    Asked of git rather than of `.gitignore`'s text. The first version of this test
    grepped for the `!.zenodo.json` negation and stayed GREEN when that line was
    commented out, because `#!.zenodo.json` still contains the substring. Git is the
    only thing that actually knows whether a path is tracked, and no amount of
    rewriting the ignore file can fool it.
    """
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{path} is not tracked by git: {proc.stderr.strip()}"
