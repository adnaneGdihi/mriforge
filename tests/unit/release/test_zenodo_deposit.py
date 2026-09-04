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
    expected = mod.doi_badge(mod.CONCEPT_DOI)
    assert expected in README_FILE.read_text(), (
        f"README.md does not contain the line zenodo_deposit.py prints:\n  {expected}"
    )


def test_a_drifted_badge_shape_is_detected(mod):
    """Planted violation: the two owners disagreeing is exactly what must fail."""
    drifted = mod.doi_badge(mod.CONCEPT_DOI).replace("zenodo.org", "example.org")
    assert drifted not in README_FILE.read_text()


# --------------------------------------------------------------------------- 4
#
# The badge's SHAPE has one owner (section 2).  Its NUMBER is a second, separate
# ownership question, and the one with a live failure mode: Zenodo mints a concept DOI
# and a version DOI per deposit, shows the version DOI first, and a README pinned to it
# keeps pointing at v0.1.0 after v0.2.0 ships.  Nothing external catches that --
# ``/badge/DOI/<doi>.svg`` renders whatever string it is given (measured 2026-09-04: a
# real DOI, a nonexistent id and the literal ``not-a-doi-at-all`` all returned HTTP 200
# with a well-formed SVG), so the wrong number produces a badge that looks perfect.
#
# Four files carry the number -- the README badge, the README BibTeX block,
# ``CITATION.cff``'s ``doi:``, and its ``identifiers:`` list.  Each is pinned to
# ``zenodo_deposit.CONCEPT_DOI`` rather than to its neighbours, so there is one owner
# and a release updates one line.


def _doi_offenders(readme: str, citation: dict, mod: ModuleType) -> list[str]:
    """Every way the four declarations can disagree with the elected owner.

    A list rather than an assertion so the same predicate can be run against planted
    text below -- a checker that has only ever been called on a passing tree is not
    known to be able to fail (non-negotiable 15).
    """
    concept, version = mod.CONCEPT_DOI, mod.VERSION_DOI
    bad: list[str] = []
    if mod.doi_badge(concept) not in readme:
        bad.append("README badge does not carry the concept DOI")
    if version in readme:
        bad.append(f"README carries the version DOI {version}, which stops at v0.1.0")
    if f"doi       = {{{concept}}}" not in readme:
        bad.append("README BibTeX does not cite the concept DOI")
    if citation.get("doi") != concept:
        bad.append(f"CITATION.cff doi is {citation.get('doi')!r}, not the concept DOI")
    idents = {i.get("value") for i in citation.get("identifiers", [])}
    if version not in idents:
        bad.append("CITATION.cff does not record the version DOI as an identifier")
    return bad


def test_the_four_doi_declarations_agree_with_the_elected_owner(mod):
    citation = yaml.safe_load(CITATION_FILE.read_text())
    assert _doi_offenders(README_FILE.read_text(), citation, mod) == []


def test_the_concept_and_version_dois_are_distinct(mod):
    """Otherwise every check above is vacuous -- both halves would be satisfied at once."""
    assert mod.CONCEPT_DOI != mod.VERSION_DOI


@pytest.mark.parametrize(
    ("mutate_readme", "mutate_citation", "expect"),
    [
        (lambda r, m: r.replace(m.CONCEPT_DOI, m.VERSION_DOI), None, "version DOI"),
        (lambda r, m: r.replace(m.doi_badge(m.CONCEPT_DOI), ""), None, "README badge"),
        (lambda r, m: r.replace(f"doi       = {{{m.CONCEPT_DOI}}}", ""), None, "BibTeX"),
        (None, lambda c, m: {**c, "doi": m.VERSION_DOI}, "CITATION.cff doi"),
        (None, lambda c, m: {**c, "identifiers": []}, "identifier"),
    ],
)
def test_each_disagreement_is_detected(mod, mutate_readme, mutate_citation, expect):
    """Planted violations: one per way the four can drift, each turning the check red."""
    readme = README_FILE.read_text()
    citation = yaml.safe_load(CITATION_FILE.read_text())
    if mutate_readme is not None:
        readme = mutate_readme(readme, mod)
    if mutate_citation is not None:
        citation = mutate_citation(citation, mod)
    offenders = _doi_offenders(readme, citation, mod)
    assert any(expect in o for o in offenders), (
        f"planted violation went undetected; offenders were {offenders}"
    )


def test_the_concept_check_is_not_duplicated_in_report_badge(mod, capsys):
    """One owner. ``report_badge`` used to print a NOTE when the concept DOI was not
    the recorded one -- *after* ``actions/publish`` had already run, so it named a
    problem at the one moment nothing could be done about it. The invariant now
    belongs to ``assert_concept_unchanged``, which raises while the deposit is still
    a draft (see section 4). Keeping the weaker copy as defence in depth is exactly
    what non-negotiable 17 forbids: neither would then be audited as the sole line.
    """
    mod.report_badge({"doi": "10.5281/zenodo.1", "conceptdoi": "10.5281/zenodo.2"})
    assert "not the one recorded in CONCEPT_DOI" not in capsys.readouterr().err
    assert hasattr(mod, "assert_concept_unchanged"), "the elected owner is gone"


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


# --------------------------------------------------------------------------- 4
#
# A release after the first must add a VERSION to the published record, never mint a
# new one.  ``POST /deposit/depositions`` and ``POST .../actions/newversion`` both
# return a perfectly good draft, and the difference only shows up in the concept DOI
# of the published result -- by which point it is irreversible and the README badge,
# the README BibTeX block and CITATION.cff all point at an archive that no longer
# tracks the project.  So the discrimination is made here, on the call the script
# issues, and the gate that enforces it is exercised in both directions.


class _FakeZenodo:
    """Records every request and answers with the shape Zenodo really returns."""

    def __init__(self, *, conceptrecid: str = "22291316", inherited: tuple[str, ...] = ()):
        self.calls: list[tuple[str, str]] = []
        self.conceptrecid = conceptrecid
        self.inherited = inherited

    def __call__(self, url, token, method="GET", payload=None, data=None):
        self.calls.append((method, url))
        if url.endswith("/actions/newversion"):
            return {"links": {"latest_draft": "https://zenodo.org/api/deposit/depositions/999"}}
        if url.endswith("/actions/publish"):
            return {"doi": "10.5281/zenodo.999", "conceptdoi": "10.5281/zenodo.22291316"}
        if data is not None or method == "DELETE":
            return {}
        return self._draft()

    def _draft(self) -> dict:
        return {
            "id": 999,
            "conceptrecid": self.conceptrecid,
            "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.999"}},
            "links": {
                "bucket": "https://zenodo.org/api/files/bucket",
                "html": "https://zenodo.org/deposit/999",
            },
            "files": [
                {
                    "filename": name,
                    "links": {"self": f"https://zenodo.org/api/deposit/depositions/999/files/{n}"},
                }
                for n, name in enumerate(self.inherited)
            ],
        }

    def methods_for(self, needle: str) -> list[str]:
        return [m for m, u in self.calls if needle in u]


@pytest.fixture
def artefact(tmp_path: Path) -> Path:
    wheel = tmp_path / "spectramr-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"not really a wheel")
    return wheel


def _run(mod, fake, artefact, monkeypatch, **kw):
    monkeypatch.setattr(mod, "_request", fake)
    return mod.deposit(
        {"title": "t"}, [artefact], "tok", mod.LIVE_API, kw.pop("publish", False), **kw
    )


def test_a_release_adds_a_version_rather_than_minting_a_new_record(mod, artefact, monkeypatch):
    fake = _FakeZenodo()
    _run(mod, fake, artefact, monkeypatch)

    assert fake.methods_for("/actions/newversion") == ["POST"], "newversion was not called"
    # The tell: a bare POST to the collection endpoint is what mints a new concept DOI.
    assert ("POST", f"{mod.LIVE_API}/deposit/depositions") not in fake.calls


def test_the_default_parent_is_the_published_record(mod, artefact, monkeypatch):
    fake = _FakeZenodo()
    _run(mod, fake, artefact, monkeypatch)
    assert any(
        f"/depositions/{mod.PARENT_RECORD_ID}/actions/newversion" in u for _, u in fake.calls
    )


def test_inherited_files_are_dropped_before_the_new_ones_are_uploaded(mod, artefact, monkeypatch):
    # newversion seeds the draft with the parent's files; published, they cannot be
    # removed, so v0.1.0's PDF would ship inside the v0.2.0 record.
    fake = _FakeZenodo(inherited=("spectraMR.pdf",))
    _run(mod, fake, artefact, monkeypatch)

    order = [m for m, _ in fake.calls]
    assert "DELETE" in order, "the inherited file was published alongside the new ones"
    assert order.index("DELETE") < order.index("PUT"), "deleted after uploading, not before"


def test_a_first_ever_deposit_can_still_mint_a_record(mod, artefact, monkeypatch):
    fake = _FakeZenodo()
    _run(mod, fake, artefact, monkeypatch, parent_id=None)

    assert ("POST", f"{mod.LIVE_API}/deposit/depositions") in fake.calls
    assert fake.methods_for("/actions/newversion") == []


def test_publish_is_refused_before_a_second_concept_doi_can_be_minted(mod, artefact, monkeypatch):
    # PLANTED: a draft whose parent concept is not the one the README cites.
    fake = _FakeZenodo(conceptrecid="99999999")
    with pytest.raises(RuntimeError, match=r"SECOND concept DOI"):
        _run(mod, fake, artefact, monkeypatch, publish=True)

    # The point of the gate is *when* it fires. A note printed afterwards is not one.
    assert fake.methods_for("/actions/publish") == [], "published anyway -- irreversibly"


def test_publish_proceeds_when_the_concept_is_the_published_one(mod, artefact, monkeypatch):
    fake = _FakeZenodo(conceptrecid=mod.CONCEPT_RECID)
    _run(mod, fake, artefact, monkeypatch, publish=True)
    assert fake.methods_for("/actions/publish") == ["POST"]


def test_the_sandbox_has_its_own_id_space_and_is_exempt(mod):
    # Enforcing a zenodo.org concept id against sandbox.zenodo.org would fail every
    # rehearsal -- a gate that cannot pass stops being run.
    mod.assert_concept_unchanged({"conceptrecid": "1"}, mod.SANDBOX_API)
    with pytest.raises(RuntimeError):
        mod.assert_concept_unchanged({"conceptrecid": "1"}, mod.LIVE_API)


def test_an_absent_conceptrecid_is_refused_rather_than_assumed(mod):
    with pytest.raises(RuntimeError, match="<absent>"):
        mod.assert_concept_unchanged({}, mod.LIVE_API)


def test_the_record_ids_are_derived_from_the_dois_not_written_again(mod):
    # One owner for each number: re-typing them is how the two drift apart.
    assert str(mod.PARENT_RECORD_ID) == mod.VERSION_DOI.rsplit(".", 1)[1]
    assert mod.CONCEPT_DOI.rsplit(".", 1)[1] == mod.CONCEPT_RECID
    assert int(mod.CONCEPT_RECID) != mod.PARENT_RECORD_ID


# ---------------------------------------------------------------------------
# 5. Which inherited files ride the new version
#
# `newversion` seeds the draft with the parent's files, so the choice is forced on
# every release after the first.  Dropping everything is the safe default -- a
# published file cannot be removed, and every distribution artefact is
# version-stamped, so keeping them all makes each record hoard every earlier wheel.
# What must ride the tip is therefore *named*, and a name that is not there raises:
# the silent alternative strands the file on the old version, where nobody looks
# until they resolve the concept DOI and find it missing.
# ---------------------------------------------------------------------------


def test_a_named_inherited_file_is_kept_while_the_rest_are_dropped(mod, artefact, monkeypatch):
    fake = _FakeZenodo(inherited=("spectraMR.pdf", "spectramr-0.0.9-py3-none-any.whl"))
    _run(
        fake=fake,
        mod=mod,
        artefact=artefact,
        monkeypatch=monkeypatch,
        carry_forward=("spectraMR.pdf",),
    )
    deleted = [u for m, u in fake.calls if m == "DELETE"]
    assert len(deleted) == 1, f"expected exactly one deletion, got {deleted}"
    assert deleted[0].endswith("/files/1"), "the wheel is index 1; the PDF must survive"


def test_by_default_every_inherited_file_is_dropped(mod, artefact, monkeypatch):
    """The accumulation guard: without a name, nothing is carried."""
    fake = _FakeZenodo(inherited=("spectraMR.pdf", "spectramr-0.0.9-py3-none-any.whl"))
    _run(fake=fake, mod=mod, artefact=artefact, monkeypatch=monkeypatch)
    assert len([u for m, u in fake.calls if m == "DELETE"]) == 2


def test_carrying_forward_a_file_the_draft_does_not_hold_raises(mod, artefact, monkeypatch):
    fake = _FakeZenodo(inherited=("spectraMR.pdf",))
    with pytest.raises(RuntimeError, match="carry forward"):
        _run(
            fake=fake,
            mod=mod,
            artefact=artefact,
            monkeypatch=monkeypatch,
            carry_forward=("paper.pdf",),
        )
    assert "DELETE" not in [m for m, _ in fake.calls], "must refuse before deleting anything"


def test_the_carry_forward_check_runs_before_any_deletion(mod, artefact, monkeypatch):
    """A typo must not cost the files it was not about."""
    fake = _FakeZenodo(inherited=("a.pdf", "b.whl"))
    with pytest.raises(RuntimeError):
        _run(
            fake=fake,
            mod=mod,
            artefact=artefact,
            monkeypatch=monkeypatch,
            carry_forward=("a.pdf", "typo.pdf"),
        )
    assert [m for m, _ in fake.calls if m == "DELETE"] == []


@pytest.mark.parametrize(
    ("cli", "env", "expected"),
    [
        (None, None, ()),
        (["a.pdf"], None, ("a.pdf",)),
        (None, "a.pdf b.pdf", ("a.pdf", "b.pdf")),
        (None, "a.pdf:b.pdf", ("a.pdf", "b.pdf")),
        (["a.pdf"], "a.pdf", ("a.pdf",)),
    ],
)
def test_the_two_declaration_homes_resolve_in_one_place(mod, cli, env, expected):
    assert mod.resolve_carry_forward(cli, env) == expected


def test_two_homes_that_disagree_raise_rather_than_electing_a_winner(mod):
    with pytest.raises(RuntimeError, match="ZENODO_CARRY_FORWARD"):
        mod.resolve_carry_forward(["a.pdf"], "b.pdf")


def test_the_workflow_default_names_the_paper(mod):
    """The next release must not silently strand the PDF on the old version.

    An empty default would do exactly that, and it would look like a normal run.
    Paired with the raise above, this default is self-auditing: if a future parent
    record no longer holds the file, the dispatch fails loudly instead.
    """
    wf = yaml.safe_load((REPO_ROOT / ".github/workflows/zenodo.yml").read_text())
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert inputs["carry_forward"]["default"] == "spectraMR.pdf"


def test_the_filename_input_reaches_the_script_through_the_environment(mod):
    """`$args` word-splitting is safe for booleans and is the injection seam here."""
    text = (REPO_ROOT / ".github/workflows/zenodo.yml").read_text()
    assert "ZENODO_CARRY_FORWARD: ${{ inputs.carry_forward }}" in text
    assert "inputs.carry_forward }}" not in text.split("run: |")[1], (
        "the input must not be interpolated into the run block"
    )
