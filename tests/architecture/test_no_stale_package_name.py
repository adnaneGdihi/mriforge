"""Fitness function: the pre-rename package name does not come back.

The rename to ``mriforge`` was a mechanical rewrite over 7,633 files, and
non-negotiable #19 is explicit that a codemod is locally well-formed everywhere
it is wrong -- ruff cannot see a half-applied rename, and neither can an
import-time check while the *old* package is still installed on the machine (an
editable ``.pth`` makes a stale dotted path resolve silently to the old tree
instead of raising).

So the ratchet is textual and runs over ``git grep``: the tracked set, not an
``rglob``, which would sweep up untracked scratch files and report a defect that
is not in the repository.

Two things this file must never do.

**Spell the old name.** Every needle is assembled from fragments at runtime, and
the prose above says "the pre-rename name" rather than writing it. A literal
would make this file match itself, and the usual fix for that -- skipping the
path that does the scanning -- puts a hole exactly where the scanner lives. The
first draft of this file spelled the name in its docstring; it went undetected
only because the file was still untracked, and ``git grep`` cannot see an
untracked file. Committing it turned the guard red against itself.

**Assume one spelling.** The rewrite covered the plain and hyphenated forms and
was declared clean on that evidence -- while 144 files still carried the name
escaped as ``\\_`` (Sphinx and LaTeX escape underscores in titles, so an
apidoc-generated title kept the old name in a form the plain pattern cannot
match), and one file used a non-ASCII hyphen. NEEDLES below is that lesson: one
entry per *shape* the name can take, not one per name.

**Confuse a module identifier with a directory name.** The same rewrite also
over-reached: it renamed 1,445 occurrences of the old name inside *filesystem
path literals* -- cluster checkout roots like the one 18 shell/sbatch lines
``cd`` into, and the paths quoted inside captured audit logs. Renaming a Python
package does not rename a directory on someone else's disk, and rewriting a
captured log falsifies the record rather than fixing a path. So the rule this
file enforces is narrower than "the old name is gone": *the old name must not
appear as a package, module or distribution identifier.* A path token rooted at
a historical checkout root **that this tree still carries** is an on-disk
location and is correct as written -- it is stripped before the scan. The open
set is derived per tree rather than declared, because the private checkout and
the published export do not carry the same roots, and a declared set is wrong in
whichever of the two it was not written for. Stripping is scoped to the path
**token**, not the line, so a stale dotted import sitting beside such a path is
still caught.
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

REPO_ROOT = Path(__file__).resolve().parents[2]

_G = "gan"
_M = "mri"

#: Every shape the pre-rename name is known to take. Matched as fixed strings,
#: case-insensitively. The separator is what varies: a plain underscore, an
#: ASCII hyphen, a backslash-escaped underscore (Sphinx/LaTeX titles), or one of
#: the Unicode dashes that a prose editor substitutes for a typed hyphen.
_SEPARATORS = [
    "_",
    "-",
    "\\_",
    "‐",  # hyphen
    "‑",  # non-breaking hyphen
    "‒",  # figure dash
    "–",  # en dash
    "—",  # em dash
    "­",  # soft hyphen
]
NEEDLES = [f"{_G}{sep}{_M}" for sep in _SEPARATORS]

#: Checkout roots that named this repository on a real disk *before* the rename.
#: These directories did not move, so the old name is the correct spelling and
#: rewriting it breaks an operational path or falsifies a captured log.
#: Assembled from fragments for the same reason NEEDLES is: a literal here would
#: make this file match itself.
#:
#: This tuple is the *candidate* list -- the historical record of which
#: directories ever carried the old name. It is not itself the hole. The hole is
#: derived from it per tree by ``_active_roots()``, so a root whose last
#: occurrence has gone stops waiving anything on its own, without anyone having
#: to notice it went dead. An unlisted root still fails closed (planted below).
#:
#: Two entries carry a username. They are DATA, in the same class as the SLURM
#: leak-detector needles in ``tests/unit/infrastructure/orchestration/`` -- a
#: record of directories that existed, not a path anything resolves. Placeholding
#: them was tried and reverted: it fails
#: ``test_old_package_name_survives_only_where_documented`` in the private
#: checkout, because those directories are where the tree actually lived and the
#: unwaived occurrences are then reported as stale-name hits. In the published
#: export the same two entries are inert -- ``_active_roots()`` returns the empty
#: tuple there, because none of the files carrying those paths ship. So the
#: string is published; the waiver it would grant is not.
_OLD = _G + "_" + _M
_OLD_HYPHEN = _G + "-" + _M
ALLOWED_ROOTS = (
    "Work/research/" + _OLD,
    "project/johnsson/agdihi/" + _OLD,
    "home/agdihi/project/" + _OLD,
    "Desktop/work/" + _OLD,
    "johnsson/otheruser/" + _OLD,
    "project/someone_else/" + _OLD,
    # Claude project directories slugify a path: every "/" AND "_" becomes "-".
    "-home-adn-Work-research-" + _OLD_HYPHEN,
)

#: Forge URLs that name the **repository**, which was not renamed with the
#: package. A repository name is not a module identifier: rewriting one does not
#: fix an import, it turns a working link into a 404. At the time of writing the
#: target name is occupied by a *different*, empty repository, so the rewritten
#: spelling resolves to the wrong project rather than redirecting.
#:
#: This allowance is direction-safe and decides nothing about whether the
#: repository is eventually renamed: the old spelling stays valid either way,
#: because a forge issues a permanent redirect from the old name after a rename.
ALLOWED_URLS = ("github.com/adnaneGdihi/" + _OLD,)

#: Both tuples are **historical identifiers that are correct as written** -- an
#: on-disk location that did not move, or a repository that did not rename.
#:
#: This is the single declared vocabulary, and both consumers are derived from
#: it rather than from each other (non-negotiable #17). They do NOT share one
#: compiled pattern, because they are asking different questions: the codemod
#: masks every literal here unconditionally, since it is offered a rewrite
#: against the text in front of it and must not consult a census; the predicate
#: masks only the subset this tree still carries, so a hole closes by
#: construction once nothing needs it. The relation is directional and checked
#: below -- the predicate's set is always a subset of this one, so it can only
#: ever be stricter than the codemod, never more permissive.
ALLOWED_LITERALS = ALLOWED_ROOTS + ALLOWED_URLS

#: An allowed token is the prefix plus the characters a path or URL may continue
#: with. It deliberately stops at whitespace, a quote and a colon, so stripping
#: consumes one token and never the rest of the line.
def _build_allowed_path(literals: tuple[str, ...]) -> re.Pattern[str]:
    """Compile the strip pattern for exactly ``literals``.

    Pure and total, so the tests below drive it with synthetic sets rather than
    with whatever the surrounding tree happens to contain.

    The zero-literal case is spelled out rather than left to the join: an empty
    alternation compiles to ``(?:)``, which matches the empty string at every
    position, so such a pattern would strip nothing *and* never fail. That is the
    vacuous shape non-negotiable #15 exists to keep out of a detector, and here it
    is reachable rather than theoretical -- the export ships none of the files
    that carry a historical checkout root.
    """
    if not literals:
        return re.compile(r"(?!)")  # matches nowhere, at any position
    return re.compile(
        "(?:" + "|".join(re.escape(r) for r in literals) + r")[A-Za-z0-9_./<>-]*"
    )


#: The codemod's contract: EVERY historical identifier, masked unconditionally.
#: ``scripts/migrations/rename_package_identifier.py`` imports this by name, and
#: it must not depend on what the surrounding tree happens to contain -- a
#: rewrite is offered against the text in front of it, not against a census.
_ALLOWED_PATH = _build_allowed_path(ALLOWED_LITERALS)


def _active_roots(candidates: tuple[str, ...] = ALLOWED_LITERALS) -> tuple[str, ...]:
    """The candidate literals that still occur in *this* tree.

    A hole in a scan is only safe while something needs it. Deriving the open
    holes from actual occurrence closes every unneeded one **by construction**, in
    whichever tree the suite runs in -- which a fixed tuple plus a "still occurs"
    ratchet cannot do. Those two trees disagree: the private checkout carries six
    of these literals in paths that are correct as written, while the published
    tree ships none of the files carrying them. A ratchet asserting they all still
    occur is therefore right in one tree and wrong in the other, and it fails in
    the published one precisely *because* sanitization worked.

    This is a DIFFERENT question from the one ``_ALLOWED_PATH`` answers, not a
    second owner of the same one (non-negotiable #17): that pattern says what the
    codemod may mask, this says which holes this tree still needs open.
    """
    return tuple(r for r in candidates if _git_grep_files([r]))


@functools.lru_cache(maxsize=1)
def _allowed_path() -> re.Pattern[str]:
    """The live pattern for this tree. Deferred: it shells out to git."""
    return _build_allowed_path(_active_roots())


def _strip_allowed_paths(text: str, pattern: re.Pattern[str] | None = None) -> str:
    """Remove historical identifiers, leaving everything else intact."""
    return (pattern if pattern is not None else _allowed_path()).sub("", text)


#: A space is deliberately NOT a separator: "multi-organ MRI" contains
#: "gan MRI", and rewriting it would corrupt correct prose.
_FALSE_POSITIVE = "multi-or" + _G + " MRI"

#: The one place the old name is still correct. ``universal_dataset`` repairs
#: paths recorded in EXISTING manifests, so this literal names a historical
#: on-disk location rather than this package. Rewriting it would not raise --
#: it would silently stop repairing those paths.
EXCLUDED = "src/mriforge/data/datasets/universal_dataset.py"
EXCLUDED_LITERAL = "/" + _G + "_" + _M + "/databases/m4raw/databases/"


def _git_grep_files(needles: list[str]) -> list[str]:
    """Tracked text files matching any needle (``-I`` drops binaries)."""
    args = ["git", "-C", str(REPO_ROOT), "grep", "-lIFi"]
    for n in needles:
        args += ["-e", n]
    args += ["--", "."]
    out = subprocess.run(args, capture_output=True, text=True, check=False)
    # git grep exits 1 when nothing matches -- the clean state, not an error.
    if out.returncode not in (0, 1):
        pytest.fail(f"git grep failed ({out.returncode}): {out.stderr[:400]}")
    return [ln for ln in out.stdout.splitlines() if ln]


def _offending_files() -> list[str]:
    """git grep finds candidates; the predicate decides which are offences."""
    offenders = []
    for rel in set(_git_grep_files(NEEDLES)) - {EXCLUDED}:
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            offenders.append(rel)  # unreadable: report, never silently pass
            continue
        if _is_offending_text(text):
            offenders.append(rel)
    return sorted(offenders)


def test_old_package_name_survives_only_where_documented() -> None:
    offenders = _offending_files()
    assert not offenders, (
        f"{len(offenders)} tracked file(s) still carry the pre-rename package "
        "name in one of its spellings.\nA stale dotted path does not raise on a "
        "machine where the old package is still installed -- it resolves to the "
        "old tree. Rewrite them.\n"
        + "\n".join(f"  {p}" for p in offenders[:40])
    )


def _is_offending_text(text: str, pattern: re.Pattern[str] | None = None) -> bool:
    """Does this text carry the old name as an *identifier*?

    Sole owner of the rule (non-negotiable #17). ``git grep`` cannot strip, so
    it is demoted to a candidate finder and this predicate decides; the corpus
    test below pins the prefilter to be non-vacuous rather than pinning a second
    implementation to agree with this one.
    """
    lowered = _strip_allowed_paths(text, pattern).lower()
    return any(n.lower() in lowered for n in NEEDLES)


def test_every_needle_shape_is_actually_caught() -> None:
    """The guard is only a guard for a shape it has been watched to catch.

    A typo in one NEEDLES entry would otherwise drop that whole shape from
    coverage with every run still green (non-negotiable #15). Each separator is
    fed a synthetic offender built independently of NEEDLES, so an entry that
    stopped spelling the name would fail here rather than go quiet.
    """
    assert len(set(NEEDLES)) == len(_SEPARATORS), "duplicate separator in NEEDLES"

    for sep in _SEPARATORS:
        planted = f"from {_G}{sep}{_M}.models import Thing"
        assert _is_offending_text(planted), (
            f"separator {sep!r} (U+{ord(sep[-1]):04X}) is in _SEPARATORS but the "
            f"needle set does not match {planted!r}. That shape is uncovered."
        )

    # Upper-case and title-case must be caught too: the corpus scan passes -i.
    for variant in (_G.upper(), _G.capitalize()):
        planted = f"{variant}_{_M.upper()}_HOME"
        assert _is_offending_text(planted), f"case variant missed: {planted!r}"


def test_the_guard_does_not_match_correct_prose() -> None:
    """"multi-organ MRI" must survive: it contains the stem by coincidence.

    A case-insensitive sweep that treated a space as a separator would rewrite
    real prose -- the phrase is in a shipped model docstring. Prove the needle
    set discriminates rather than trusting that it does.
    """
    for benign in (
        _FALSE_POSITIVE,
        f"multi-or{_G} {_M.upper()} reconstruction",
        "a GAN trained on MRI volumes",
        "organ segmentation, MRI-derived",
    ):
        assert not _is_offending_text(benign), (
            f"a needle matches {benign!r}, which is correct English. Narrow the "
            "separator set -- a space must never be a separator."
        )


def test_the_prefilter_is_not_vacuous() -> None:
    """The predicate is the only owner, so the prefilter must still find things.

    Demoting ``git grep`` to a candidate finder creates a new failure mode it did
    not have as a co-owner: a prefilter that silently matched *nothing* would
    make the corpus test pass with the predicate never consulted. Pin it on the
    one file known to carry the name.
    """
    assert EXCLUDED in _git_grep_files(NEEDLES), (
        "the prefilter no longer finds the one file that legitimately carries "
        "the old name, so it may be matching nothing at all and the corpus scan "
        "would pass vacuously."
    )
    text = (REPO_ROOT / EXCLUDED).read_text(encoding="utf-8")
    assert _is_offending_text(text), "predicate no longer flags the excluded file"


def test_a_historical_checkout_root_is_not_an_offence() -> None:
    """Each allowed root, on its own, must read as clean.

    One planted line per root: an entry that stopped matching -- a typo, a
    changed separator -- would otherwise sit in the tuple doing nothing.

    Driven through a pattern built from that one root, never through the live
    one. The live pattern holds only the roots this tree still carries, so
    iterating it would report every root the export legitimately dropped as a
    misspelling. What is asserted here is a property of the rule, not of the
    tree, and it therefore reads the same in both.
    """
    assert ALLOWED_ROOTS, (
        "ALLOWED_ROOTS is empty, so the loop below asserts nothing. An empty "
        "allowlist must be a deliberate, visible decision -- delete this test "
        "with it rather than leaving a green test that checks no roots."
    )
    for root in ALLOWED_ROOTS:
        planted = f'DATA_ROOT="/{root}/databases/m4raw"'
        assert not _is_offending_text(planted, _build_allowed_path((root,))), (
            f"allowed root {root!r} does not strip {planted!r}: that entry is "
            "misspelt, so it waives nothing and protects nothing."
        )


def test_the_active_set_follows_occurrence_in_both_directions() -> None:
    """The hole opens for a root something carries, and for no other.

    Planted both ways (non-negotiable #15). A derivation that answered "active"
    for everything and one that answered "inactive" for everything would each
    satisfy a one-directional check while destroying the property in opposite
    directions -- the first reopens every hole, the second closes the ones that
    are load-bearing and reports correct paths as offences.
    """
    absent = "no-such-checkout-root-" + _OLD
    assert _git_grep_files([absent]) == [], (
        f"{absent!r} was chosen because nothing carries it; something now does, "
        "so the plant below can no longer tell a working derivation from a "
        "broken one. Pick another string."
    )
    assert _active_roots((absent,)) == (), "a root nothing carries was made active"

    present = "mriforge"  # this package's own directory: tracked in every tree
    assert _active_roots((present,)) == (present,), (
        "a root the tree plainly carries was dropped from the active set"
    )
    assert set(_active_roots()) <= set(ALLOWED_LITERALS), (
        "the active set is a SUBSET of the declared literals -- derivation may "
        "close a hole, never invent one. Bounded by ALLOWED_LITERALS rather than "
        "ALLOWED_ROOTS because a URL is a literal too, and bounding by the roots "
        "alone would read a correctly-derived URL as an invented hole."
    )


def test_an_inactive_root_opens_no_hole() -> None:
    """What the retired "still occurs" ratchet could only nudge toward.

    That ratchet asserted every listed root still occurred, and asked a human to
    delete the ones that did not. The property it was protecting -- that a stale
    identifier arriving under a no-longer-used root is caught rather than waived
    -- is asserted directly here, and holds without anyone acting on a message.

    The candidate list is passed in, so the root under test is *listed and not
    occurring* -- the state the export puts every one of these roots into. An
    earlier draft planted an unlisted root instead and passed for the wrong
    reason: unlisted roots are already covered below, and a derivation that
    declared every candidate active left that draft green.
    """
    absent = "no-such-checkout-root-" + _OLD
    pattern = _build_allowed_path(_active_roots((absent, ALLOWED_ROOTS[0])))
    planted = f'DATA_ROOT="/{absent}/databases/m4raw"'
    assert _is_offending_text(planted, pattern), (
        "a listed root that occurs nowhere still stripped a path under it, so "
        "its hole outlived the last thing that needed it"
    )


def test_a_zero_root_pattern_strips_nothing() -> None:
    """The empty-alternation footgun, planted.

    ``"(?:" + "|".join(()) + ...`` compiles to ``(?:)``, which matches the empty
    string at every position: built that way, a zero-root pattern strips nothing
    and also never fails, so nothing downstream could tell it apart from a
    working one. This is the configuration the published tree actually runs --
    it ships no historical checkout root at all -- so it is planted rather than
    reasoned about.
    """
    text = f'DATA_ROOT="/Desktop/work/{_OLD}/databases/m4raw"'
    assert _build_allowed_path(()).sub("", text) == text
    assert _is_offending_text(text, _build_allowed_path(()))


def test_an_unlisted_checkout_root_is_still_an_offence() -> None:
    """The allowlist fails CLOSED: only the enumerated roots are waived.

    A path-shaped context is not itself a licence. A new cluster root has to be
    added deliberately, with someone confirming the directory really exists
    under the old name, rather than being inferred from the surrounding slashes.
    """
    for planted in (
        f"DATA_ROOT=/project/foo/{_OLD}/databases",
        f"cd /scratch/newcluster/{_OLD}",
        f"REPO_ROOT=/home/someone/{_OLD_HYPHEN}",
    ):
        assert _is_offending_text(planted), (
            f"{planted!r} was waved through. The allowlist must fail closed: an "
            "unlisted root is an offence until someone lists it."
        )


def test_stripping_a_path_does_not_mask_an_identifier_beside_it() -> None:
    """Stripping is scoped to the path token, never to the line.

    This is the shape that makes an allowlist dangerous: an audit line quotes a
    historical path *and* a dotted import, and a line-scoped strip would hide
    the second behind the first. The most likely place for a stale identifier to
    survive is exactly beside a legitimately-old path.
    """
    root = ALLOWED_ROOTS[0]
    pattern = _build_allowed_path((root,))  # not the live one: it may be dead here
    for planted in (
        f"evidence: /home/<user>/{root}/x.py:12 -- `from {_OLD}.models import Thing`",
        f"see /{root}/README then import {_OLD}.core",
        f'"/{root}/a.py": "{_OLD_HYPHEN} audit --probe"',
    ):
        assert _is_offending_text(planted, pattern), (
            f"{planted!r} passed. The path token was stripped along with the "
            "identifier beside it -- narrow the strip back to the token."
        )


def test_a_repository_url_is_not_an_offence() -> None:
    """Each allowed repository URL, on its own, must read as clean.

    The repository did not rename with the package, so a link spelled the old
    way is the *working* one. An entry that stopped matching would sit in the
    tuple doing nothing while the corpus test went red on a correct link -- and
    the obvious "fix" would be to rewrite the link into a 404.
    """
    assert ALLOWED_URLS, (
        "ALLOWED_URLS is empty, so the loop below asserts nothing. If the "
        "repository really was renamed, delete this test with the tuple "
        "rather than leaving a green test that checks no urls."
    )
    for url in ALLOWED_URLS:
        planted = f'"issue": "https://{url}/issues/1585"'
        assert not _is_offending_text(planted), (
            f"allowed url {url!r} is in ALLOWED_URLS but does not strip "
            f"{planted!r}. That allowance is dead."
        )


def test_an_unlisted_repository_url_is_still_an_offence() -> None:
    """The URL allowance fails CLOSED, exactly as the root allowance does.

    A URL-shaped context is not a licence either. Only this repository under
    this owner is waived; a different owner, a different forge, and above all a
    *distribution* index are all still the old name used as an identifier.
    """
    for planted in (
        f"https://github.com/someoneelse/{_OLD}/issues/1",
        f"https://gitlab.com/adnaneGdihi/{_OLD}/-/issues/1",
        f"https://pypi.org/project/{_OLD_HYPHEN}/",
        f"pip install {_OLD_HYPHEN}",
    ):
        assert _is_offending_text(planted), (
            f"{planted!r} was waved through. Only the enumerated repository is "
            "waived -- a distribution or a foreign forge is still an offence."
        )


def test_stripping_a_repo_url_does_not_mask_an_identifier_beside_it() -> None:
    """Masking is scoped to the URL token, never to the line.

    Same hazard as the path case, and likelier here: a changelog entry cites the
    pull request that did the rename *and* the import it changed, on one line.
    """
    url = ALLOWED_URLS[0]
    for planted in (
        f"see https://{url}/pull/1536 -- it rewrote `from {_OLD}.core import x`",
        f'"found_by": "https://{url}/pull/1584", "sym": "{_OLD}.metrics"',
        f"https://{url}/issues/1 then run `{_OLD_HYPHEN} audit`",
    ):
        assert _is_offending_text(planted), (
            f"{planted!r} passed. The URL token was masked along with the "
            "identifier beside it -- narrow the mask back to the token."
        )


def test_a_stale_package_directory_outside_a_root_is_an_offence() -> None:
    """``src/<old>/...`` is a package directory, and it did move.

    The revert deliberately restored only the *checkout root*; everything below
    it is inside the repository and was renamed. A ``src/<old>`` that is not
    under an allowed root is a stale reference to the package tree.
    """
    for planted in (
        f"src/{_OLD}/models/registry.py",
        f'packages = ["src/{_OLD}"]',
    ):
        assert _is_offending_text(planted), (
            f"{planted!r} passed, but the package directory did rename."
        )


def test_the_documented_exclusion_is_exact() -> None:
    """The allowance is one occurrence, not one file.

    Excluding the whole file would leave a hole in precisely the place a stale
    reference is most likely to reappear: a planted second mention inside this
    file passed the corpus guard until this test counted occurrences instead of
    skipping the path. Assert the literal is still present *and* that it is the
    only one, so the allowance can neither be deleted nor widened.
    """
    text = (REPO_ROOT / EXCLUDED).read_text(encoding="utf-8")

    assert EXCLUDED_LITERAL in text, (
        f"{EXCLUDED} no longer contains the historical manifest prefix "
        f"{EXCLUDED_LITERAL!r}. It is not a leftover: it matches paths already "
        "written into existing manifests, so removing it does not raise -- it "
        "silently stops repairing them. If it went deliberately, drop the "
        "EXCLUDED entry here too so the ratchet stays exact."
    )

    lowered = text.lower()
    count = sum(lowered.count(n.lower()) for n in NEEDLES)
    assert count == 1, (
        f"{EXCLUDED} mentions the pre-rename package name {count} times across "
        "all spellings; exactly one is allowed (the manifest prefix). The extra "
        "mention(s) are stale and invisible to the corpus guard, which skips "
        "this path."
    )
