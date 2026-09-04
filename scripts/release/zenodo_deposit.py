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
    if doi != CONCEPT_DOI:
        print(
            f"  NOTE: this record's badge DOI ({doi}) is not the one recorded in"
            f" CONCEPT_DOI ({CONCEPT_DOI}).  A NEW concept DOI means a new Zenodo"
            " record rather than a new version of the existing one -- check that"
            " before updating the constant, since the published badge would then"
            " stop pointing at the project's own archive.",
            file=sys.stderr,
        )
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
    if data is None and payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:  # surface Zenodo's own field-level message
        raise RuntimeError(f"{method} {url} -> {exc.code}: {exc.read().decode()[:600]}") from exc


def deposit(
    metadata: dict[str, object], files: list[Path], token: str, api: str, publish: bool
) -> dict:
    """Create a draft, attach every file, set metadata, optionally publish."""
    draft = _request(f"{api}/deposit/depositions", token, "POST", payload={})
    dep_id, bucket = draft["id"], draft["links"]["bucket"]
    print(f"draft deposition {dep_id} created")

    for path in files:
        _request(f"{bucket}/{path.name}", token, "PUT", data=path.read_bytes())
        print(f"  uploaded {path.name} ({path.stat().st_size} bytes)")

    updated = _request(
        f"{api}/deposit/depositions/{dep_id}", token, "PUT", payload={"metadata": metadata}
    )
    doi = updated["metadata"].get("prereserve_doi", {}).get("doi") or updated.get("doi")
    print(f"  metadata set; reserved DOI: {doi}")

    if publish:
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
        "--dry-run",
        action="store_true",
        help="print the payload and exit; no network, no token needed",
    )
    args = parser.parse_args(argv)

    metadata = build_metadata(
        json.loads(METADATA_FILE.read_text(encoding="utf-8")), package_version()
    )

    if args.dry_run:
        print(json.dumps({"metadata": metadata}, indent=2, sort_keys=True))
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

    deposit(
        metadata,
        distribution_files(REPO_ROOT / args.dist),
        token,
        SANDBOX_API if args.sandbox else LIVE_API,
        args.publish,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
