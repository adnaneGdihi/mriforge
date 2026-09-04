"""Deposit the built distribution on Zenodo and mint a DOI.

Why this is a separate step from ``release.yml`` rather than a job inside it
--------------------------------------------------------------------------
Publication order here is **public tree -> Zenodo -> tag**: the tag is what
triggers the PyPI publish, and a DOI that is meant to cite the released artefact
has to exist before the release announcement quotes it.  A workflow keyed on
``release: published`` or on the tag push therefore cannot run at the point in
the sequence where it is needed -- by the time either event fires, the tag it was
supposed to precede already exists.  So this runs from a button.

Publishing is irreversible in the same way a PyPI filename is, and worse: a
published Zenodo record cannot be deleted, only a new version issued, and the DOI
it minted resolves forever.  ``--publish`` is therefore opt-in.  Without it the
deposition is created and populated but left as a **draft**, which is reviewable
in the Zenodo UI and deletable, and the reserved DOI is printed so the badge and
the release notes can be prepared before anything is permanent.

The version is not declared here
--------------------------------
``.zenodo.json`` deliberately carries no ``version`` key.  Four files already
state the version independently and ``scripts/release/build_dist.py`` is the sole
comparator of the four; a fifth declaration in a file that comparator does not
read would be a version nothing checks, which is how two of the four came to
disagree in the first place.  The version is read here from the package
``__init__`` -- through ``build_dist``'s own reader, imported rather than
re-spelled, so this script cannot develop a second opinion about where the
version lives (non-negotiable 17).

Usage::

    python scripts/release/zenodo_deposit.py --dry-run          # payload only, no network
    python scripts/release/zenodo_deposit.py                    # create a DRAFT deposition
    python scripts/release/zenodo_deposit.py --publish          # draft + publish (irreversible)
    python scripts/release/zenodo_deposit.py --sandbox --dry-run

``ZENODO_TOKEN`` must carry the ``deposit:write`` scope (and
``deposit:actions`` for ``--publish``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "spectramr"
METADATA_FILE = REPO_ROOT / ".zenodo.json"
BUILD_DIST = REPO_ROOT / "scripts" / "release" / "build_dist.py"

LIVE_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"

#: Zenodo rejects an unknown metadata key with a 400 that names the field, but it
#: accepts a *missing* required one by leaving the draft incomplete -- so the
#: absence is checked here, where it can be reported, rather than at publish time.
REQUIRED_METADATA = ("title", "upload_type", "creators", "description", "license")

#: This module is the single owner of the DOI badge's *shape*, and
#: ``tests/unit/release/test_zenodo_deposit.py`` pins ``README.md`` to what
#: ``doi_badge(CONCEPT_DOI)`` produces, so the URL form cannot drift between the file
#: a reader sees and the line this script prints.  The shape matters: Zenodo's *other*
#: badge endpoint, ``/badge/<github repo id>.svg``, is registered only by its GitHub
#: integration, which cannot see a private repository -- probed 404 for this repo on
#: 2026-09-03.  Depositing over the REST API never registers that mapping, so the
#: repo-id form would render dead here while looking filled.
DOI_BADGE = "[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi})"

#: The published **concept** DOI, and the single owner of that number.  README's badge,
#: README's BibTeX block and ``CITATION.cff`` all carry it, and the tests pin all three
#: here rather than to each other, so a future release updates one line.
#:
#: Concept, not version, is the whole point: Zenodo mints both, and the deposit page
#: shows the *version* DOI first.  ``VERSION_DOI`` below is that one, recorded only so
#: the distinction is stateable and testable -- it must never appear in the badge,
#: because it is frozen on the v0.1.0 deposit and would stop tracking the project at
#: the next release.  Verified 2026-09-04: ``https://doi.org/`` + the concept DOI
#: redirects to record 22291317, the newest version.
CONCEPT_DOI = "10.5281/zenodo.22291316"
VERSION_DOI = "10.5281/zenodo.22291317"

#: Zenodo record ids, DERIVED from the two DOIs above rather than written a third
#: and fourth time.  ``PARENT_RECORD_ID`` is the published record a new release is
#: added *under*; ``CONCEPT_RECID`` is the parent whose DOI the README cites and
#: which every version must keep reporting.
PARENT_RECORD_ID = int(VERSION_DOI.rsplit(".", 1)[1])
CONCEPT_RECID = CONCEPT_DOI.rsplit(".", 1)[1]

#: Neither the status code nor the rendered content of a Zenodo badge can tell a live
#: DOI from a typo: ``/badge/DOI/<doi>.svg`` emits no ``<title>`` (that is a shields.io
#: convention) and does not validate its argument -- measured 2026-09-04, a real DOI, a
#: nonexistent zenodo id and the literal ``not-a-doi-at-all`` all returned HTTP 200 with
#: a well-formed SVG echoing back the string.  Only resolving the DOI discriminates.
DOI_RESOLVER = "https://doi.org/{doi}"
README = REPO_ROOT / "README.md"
CITATION = REPO_ROOT / "CITATION.cff"


def doi_badge(doi: str) -> str:
    """The exact README line for ``doi``.  One owner for the badge URL shape."""
    return DOI_BADGE.format(doi=doi)


def report_badge(record: dict) -> str:
    """Print the finished badge line for a published record, and return it.

    Prefers the **concept** DOI -- the one that always resolves to the newest
    version -- because a README badge pinned to a version DOI silently stops
    tracking the project at the next release.  Zenodo only returns ``conceptdoi``
    once a record is published; a draft has neither, and that is reported rather
    than papered over with the versioned DOI under a concept label.
    """
    concept = record.get("conceptdoi")
    versioned = record.get("doi")
    doi = concept or versioned
    if doi is None:
        print("  no DOI in the response -- cannot build the badge line", file=sys.stderr)
        return ""
    if concept is None:
        print(
            f"  NOTE: no concept DOI in the response; the line below pins version {versioned},"
            " which will not follow later releases.",
            file=sys.stderr,
        )
    line = doi_badge(doi)
    print(f"\n  README badge line for this record:\n    {line}")
    print(
        "  README.md, its BibTeX block and CITATION.cff are pinned to"
        " zenodo_deposit.CONCEPT_DOI; update that constant and the three follow."
    )
    return line


def _load_build_dist() -> ModuleType:
    """Import the release verifier by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("_build_dist", BUILD_DIST)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the version reader from {BUILD_DIST}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_version() -> str:
    """The version, read through the release verifier's own reader."""
    init = REPO_ROOT / "src" / PACKAGE / "__init__.py"
    version = _load_build_dist().declared_version(init.read_text(encoding="utf-8"))
    if not version:
        raise RuntimeError(f"no __version__ found in {init}")
    return version


def build_metadata(raw: dict[str, object], version: str) -> dict[str, object]:
    """``.zenodo.json`` plus the version, validated.

    Pure, so the tests drive it with synthetic metadata rather than with whatever
    ``.zenodo.json`` happens to say today.
    """
    missing = [k for k in REQUIRED_METADATA if not raw.get(k)]
    if missing:
        raise RuntimeError(
            f"{METADATA_FILE.name} is missing required field(s): {', '.join(missing)}. "
            "Zenodo accepts the draft anyway and fails at publish, so it is checked here."
        )
    if "version" in raw:
        raise RuntimeError(
            f"{METADATA_FILE.name} declares a `version`. It must not: the version has "
            "one owner (scripts/release/build_dist.py compares the four files that do "
            "declare it), and a fifth declaration that comparator never reads is a "
            "version nothing checks."
        )
    return {**raw, "version": version}


def distribution_files(dist_dir: Path) -> list[Path]:
    """The artefacts to attach, newest-build-agnostic and sorted for determinism."""
    files = sorted(p for p in dist_dir.iterdir() if p.suffix in {".whl", ".gz"})
    if not files:
        raise RuntimeError(
            f"{dist_dir} holds no .whl or .tar.gz. Run scripts/release/build_dist.py first "
            "-- depositing a record with no files mints a DOI that resolves to nothing."
        )
    return files


def _request(
    url: str, token: str, method: str = "GET", payload: object = None, data: bytes | None = None
):
    body = (
        data
        if data is not None
        else (json.dumps(payload).encode() if payload is not None else None)
    )
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        # Not optional, and NOT the same as sending nothing: urllib defaults an
        # unspecified Content-Type to application/x-www-form-urlencoded whenever a
        # body is present, and Zenodo's bucket API answers that with a 415. The
        # earlier spelling omitted the header for binary bodies believing that meant
        # "no type" -- so every artefact upload this script has ever attempted failed
        # on its first PUT, invisibly, because v0.1.0's only file was deposited by
        # hand through the web UI and no release exercised this line.
        headers["Content-Type"] = "application/octet-stream"
    elif payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:  # surface Zenodo's own field-level message
        raise RuntimeError(f"{method} {url} -> {exc.code}: {exc.read().decode()[:600]}") from exc


def open_new_version(parent_id: int, token: str, api: str) -> dict:
    """A draft that is a NEW VERSION of ``parent_id``, not a new record.

    ``POST /deposit/depositions`` mints a brand-new record with its own **concept**
    DOI -- which is the wrong call for every release after the first, because the
    published badge, the README BibTeX block and ``CITATION.cff`` all cite the
    existing concept and would silently stop pointing at the project's archive.
    Only ``actions/newversion`` adds a version underneath it.
    """
    resp = _request(f"{api}/deposit/depositions/{parent_id}/actions/newversion", token, "POST")
    latest = resp["links"]["latest_draft"]
    return _request(latest, token)


def resolve_carry_forward(cli: list[str] | None, env: str | None) -> tuple[str, ...]:
    """The one place the carry-forward set is interpreted.

    Two declaration homes -- ``--carry-forward`` and ``ZENODO_CARRY_FORWARD`` -- and
    one resolver, because a second resolver does not announce itself: both would
    compute a plausible set and which ran would depend on the call path.  Homes that
    *disagree* raise rather than electing a silent winner; a home that is merely
    absent defers.  Tokens split on the OS path separator or on whitespace, the same
    convention ``SPECTRAMR_PLUGINS`` already uses.
    """
    from_cli = tuple(cli or ())
    from_env = tuple(env.replace(os.pathsep, " ").split()) if env else ()
    if from_cli and from_env and from_cli != from_env:
        raise RuntimeError(
            f"--carry-forward says {list(from_cli)} but ZENODO_CARRY_FORWARD says "
            f"{list(from_env)}. Two homes for one decision must not be reconciled by "
            "picking one silently -- set exactly one of them."
        )
    return from_cli or from_env


def discard_inherited_files(
    draft: dict, token: str, carry_forward: tuple[str, ...] = ()
) -> list[str]:
    """Drop the files Zenodo copied in from the previous version, except named ones.

    ``newversion`` seeds the draft with the parent's files.  Left alone they are
    published alongside the new ones -- and because every distribution artefact is
    version-stamped, *keeping* them all would make each release hoard every earlier
    wheel and sdist.  So the default is to drop, and anything that should ride the
    tip is named explicitly in ``carry_forward``.

    A named file that the draft does not carry **raises**.  That is the whole point
    of naming it: the silent alternative strands the file on the old version, which
    is invisible until someone resolves the concept DOI and finds it missing, and a
    published record cannot be edited afterwards.
    """
    present = {}
    for entry in draft.get("files", []):
        present[entry.get("filename") or entry.get("key", "<unnamed>")] = entry

    missing = [name for name in carry_forward if name not in present]
    if missing:
        raise RuntimeError(
            f"asked to carry forward {missing}, but this draft inherited "
            f"{sorted(present)}. Absent is a state to report, not one to infer: "
            "either the parent record no longer holds that file or the name is a "
            "typo, and dropping it quietly would strand it on the old version."
        )

    dropped = []
    for name, entry in present.items():
        if name in carry_forward:
            print(f"  carried forward inherited file {name}")
            continue
        _request(entry["links"]["self"], token, "DELETE")
        dropped.append(name)
        print(f"  dropped inherited file {name}")
    return dropped


def assert_concept_unchanged(draft: dict, api: str) -> None:
    """Refuse to publish a draft that would mint a second concept DOI.

    This is the gate that used to be a NOTE printed *after* ``actions/publish`` --
    one owner for the invariant, and it now runs while the deposit is still a
    draft.  Skipped on the sandbox, which has its own id space.
    """
    if api != LIVE_API:
        return
    recid = str(draft.get("conceptrecid") or "")
    if recid != CONCEPT_RECID:
        raise RuntimeError(
            f"this draft's concept record is {recid or '<absent>'}, not {CONCEPT_RECID} "
            f"({CONCEPT_DOI}). Publishing it would mint a SECOND concept DOI and orphan "
            "the README badge, the README BibTeX block and CITATION.cff. Deposit a new "
            "version of the existing record instead of a new record."
        )


def deposit(
    metadata: dict[str, object],
    files: list[Path],
    token: str,
    api: str,
    publish: bool,
    parent_id: int | None = PARENT_RECORD_ID,
    carry_forward: tuple[str, ...] = (),
) -> dict:
    """Draft, attach every file, set metadata, optionally publish.

    ``parent_id`` is the published record to add a version to; ``None`` mints a new
    record, which is correct only for a first-ever deposit or on the sandbox.
    """
    if parent_id is None:
        draft = _request(f"{api}/deposit/depositions", token, "POST", payload={})
        print(f"draft deposition {draft['id']} created (NEW record -- new concept DOI)")
    else:
        draft = open_new_version(parent_id, token, api)
        print(f"draft deposition {draft['id']} created (new VERSION of record {parent_id})")
        discard_inherited_files(draft, token, carry_forward)
    dep_id, bucket = draft["id"], draft["links"]["bucket"]

    for path in files:
        _request(f"{bucket}/{path.name}", token, "PUT", data=path.read_bytes())
        print(f"  uploaded {path.name} ({path.stat().st_size} bytes)")

    updated = _request(
        f"{api}/deposit/depositions/{dep_id}", token, "PUT", payload={"metadata": metadata}
    )
    doi = updated["metadata"].get("prereserve_doi", {}).get("doi") or updated.get("doi")
    print(f"  metadata set; reserved DOI: {doi}")

    if publish:
        assert_concept_unchanged(updated, api)
        published = _request(f"{api}/deposit/depositions/{dep_id}/actions/publish", token, "POST")
        print(f"PUBLISHED -- DOI {published['doi']} (irreversible)")
        report_badge(published)
        return published

    print(
        f"left as a DRAFT. Review it at {updated['links']['html']} and publish there, or re-run with --publish."
    )
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deposit the built distribution on Zenodo.")
    parser.add_argument("--dist", default="dist", help="directory holding the built artefacts")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish the draft (IRREVERSIBLE: mints a permanent DOI)",
    )
    parser.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    parser.add_argument(
        "--new-record",
        action="store_true",
        help=(
            "mint a NEW Zenodo record with its own concept DOI, instead of adding a "
            "version to the published one. Correct for a first-ever deposit only -- it "
            "orphans the README badge and CITATION.cff for any later release."
        ),
    )
    parser.add_argument(
        "--carry-forward",
        action="append",
        metavar="FILENAME",
        help=(
            "an inherited file to keep on the new version, by exact name (repeatable). "
            "Everything else the parent record holds is dropped, so that each release "
            "does not hoard every earlier wheel. A name the draft does not carry raises. "
            "Also settable as ZENODO_CARRY_FORWARD."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payload and exit; no network, no token needed",
    )
    args = parser.parse_args(argv)

    metadata = build_metadata(
        json.loads(METADATA_FILE.read_text(encoding="utf-8")), package_version()
    )

    carry_forward = resolve_carry_forward(
        args.carry_forward, os.environ.get("ZENODO_CARRY_FORWARD")
    )

    if args.dry_run:
        print(json.dumps({"metadata": metadata}, indent=2, sort_keys=True))
        for name in carry_forward:
            print(f"would carry forward inherited file: {name}")
        dist = REPO_ROOT / args.dist
        if dist.is_dir():
            for p in distribution_files(dist):
                print(f"would attach: {p.name} ({p.stat().st_size} bytes)")
        else:
            print(f"would attach: (nothing -- {dist} does not exist yet)")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print(
            "ZENODO_TOKEN is not set. Absent is a state to report, not one to infer: "
            "this is a missing credential, not a reason to skip the deposition.",
            file=sys.stderr,
        )
        return 2

    # The sandbox has its own id space, so PARENT_RECORD_ID means nothing there.
    parent = None if (args.new_record or args.sandbox) else PARENT_RECORD_ID
    deposit(
        metadata,
        distribution_files(REPO_ROOT / args.dist),
        token,
        SANDBOX_API if args.sandbox else LIVE_API,
        args.publish,
        parent_id=parent,
        carry_forward=carry_forward,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
