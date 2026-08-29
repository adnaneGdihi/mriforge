"""Anti-drift guard for the counts stated in ``scripts/sim2rank/`` docstrings.

This module started at #1516, where ``sim2rank.py``'s module docstring
advertised ``13 axes`` and ``137 per-image metrics``. They failed differently,
and the difference is still the core of the guard: ``137`` was simply **stale**
(``PER_IMAGE_SPECS`` holds 139, and 137 matched no constant at all), while
``13`` was **arithmetically correct and still wrong** -- ``len(AXIS_CONFIGS)``
really is 13, but that is the ``legacy`` bank and ``--axis-bank`` defaults to
``registry``, so it described a sweep that does not run.

#1521 then found the same rot in ``consensus_figures.py`` -- and, with it, that
this guard could never have caught it. It was blind in **four** independent
dimensions at once, which is why widening any one of them alone fixes nothing:

1. ``SIM2RANK`` was a *constant* scan root naming one file, so no violation
   planted in a sibling module could turn the guard red.
2. Only the **module** docstring was read, while two of #1521's three sites are
   *function* docstrings.
3. ``_METRIC_COUNT`` required the literal phrase ``per-image``, which none of
   the three sites use -- they say plain ``137 metrics``.
4. There was no category detector at all, so ``~13 categories`` was unseen.

#1521 also settles *why* the fix is "name the SSOT" rather than "correct the
number": **no literal is right.** The scored set is flag-dependent -- 139
metrics in 12 categories by default, 149 in 13 under
``--include-summary-metrics`` (``SUMMARY_SPECS``' lone ``distributional``
category is disjoint from the per-image 12). So the corpus sweep below checks
**qualification, not equality**: a count is legitimate when the surrounding
text names the collection it counts, and a defect when it reads as a property
of "a sim2rank run" in general.

The checks are pure functions over docstring text, which lets the planted
violations be committed as tests rather than run once by hand (non-negotiable
15) -- and lets each detector be shown to *discriminate*, not merely to fire.
Discovery takes a ``root`` argument for the same reason: a scanner whose root
is a constant cannot be planted against at all.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2RANK_DIR = REPO_ROOT / "scripts" / "sim2rank"
SIM2RANK = SIM2RANK_DIR / "sim2rank.py"

#: Counts the sweep below has seen but not adjudicated -- see the test's own
#: docstring for why this exists and how it must (not) be regenerated.
ALLOWLIST_PATH = Path(__file__).resolve().parent / "_known_sim2rank_docstring_counts.json"

#: ``N per-image metric(s)`` -- a count of the evaluated battery.
_METRIC_COUNT = re.compile(r"(\d+)\s+per-image\s+metric", re.IGNORECASE)

#: ``N axes`` / ``N-axis`` -- a count of the degradation sweep.
_AXIS_COUNT = re.compile(r"(\d+)[\s-]+ax(?:is|es)\b", re.IGNORECASE)

#: ``~N metrics`` in *any* phrasing -- the shape #1521's sites actually used.
_ANY_METRIC_COUNT = re.compile(r"(?:~\s*)?\d+\s+(?:[\w-]+\s+)?metrics?\b", re.IGNORECASE)

#: ``~N categories`` -- derived from the metric set, so it drifts with it.
_CATEGORY_COUNT = re.compile(r"(?:~\s*)?\d+\s+(?:[\w-]+\s+)?categor(?:y|ies)\b", re.IGNORECASE)

#: How far after a count we look for the qualifier that makes it legitimate.
_QUALIFIER_WINDOW = 120

#: Naming any of these makes a count a statement about a known collection
#: rather than about "a sim2rank run" in general.
_SSOT_NAMES = (
    "PER_IMAGE_SPECS",
    "SUMMARY_SPECS",
    "DOMAIN_SPECS",
    "METRIC_SPECS",
    "AXIS_CONFIGS",
    "DEGRADATION_REGISTRY",
    "CATEGORY_DISPLAY_NAMES",
    "legacy",
)


def iter_docstrings(root: Path) -> Iterator[tuple[Path, str, str]]:
    """Yield ``(path, qualname, docstring)`` for every docstring under ``root``.

    Modules, classes and functions alike -- blindness (2) above was reading
    only the module node. ``root`` is a parameter, not a constant, so the
    discovery mechanism itself can be planted against (blindness (1)).
    """
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            doc = ast.get_docstring(node)
            if doc:
                yield path, getattr(node, "name", "<module>"), doc


def module_docstring(path: Path = SIM2RANK) -> str:
    """The module docstring, read via AST so importing torch is not required."""
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


def stale_metric_counts(doc: str, live: int) -> list[int]:
    """Per-image metric counts stated in ``doc`` that disagree with ``live``."""
    return [int(n) for n in _METRIC_COUNT.findall(doc) if int(n) != live]


def unqualified_counts(
    doc: str,
    pattern: re.Pattern[str],
    qualifiers: Sequence[str] = _SSOT_NAMES,
    window: int = _QUALIFIER_WINDOW,
) -> list[str]:
    """Matched count phrases in ``doc`` that name no collection near by.

    Returns the matched **text**, not the number: the text is what an
    allowlist entry must pin, so that editing a waived claim un-waives it.
    """
    found: list[str] = []
    for match in pattern.finditer(doc):
        near = doc[match.start() : match.end() + window]
        if not any(q in near for q in qualifiers):
            found.append(match.group(0).strip())
    return found


def unqualified_axis_counts(doc: str) -> list[int]:
    """Axis counts not tied to the ``legacy`` bank within the trailing window.

    A number is fine when the text says *which* bank it counts. It is a defect
    when it reads as a property of the sweep in general, because the default
    bank is ``registry`` and the only fixed axis count is ``legacy``'s.
    """
    found: list[int] = []
    for match in _AXIS_COUNT.finditer(doc):
        window = doc[match.start() : match.end() + _QUALIFIER_WINDOW]
        if "AXIS_CONFIGS" not in window and "legacy" not in window.lower():
            found.append(int(match.group(1)))
    return found


def survey(root: Path) -> list[tuple[str, str, str]]:
    """Every unqualified count under ``root``, as ``(relpath, qualname, text)``."""
    findings: list[tuple[str, str, str]] = []
    for path, qualname, doc in iter_docstrings(root):
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:  # a planted tree outside the repo
            rel = path.name
        for pattern in (_ANY_METRIC_COUNT, _CATEGORY_COUNT, _AXIS_COUNT):
            findings.extend((rel, qualname, text) for text in unqualified_counts(doc, pattern))
    return findings


def load_allowlist() -> set[tuple[str, str, str]]:
    """The seeded waivers, keyed by ``(relpath, qualname, exact matched text)``."""
    raw = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    return {tuple(entry) for entry in raw["waived"]}


def _require_subject() -> None:
    """Decline visibly when ``scripts/sim2rank/`` is not part of this tree.

    Two absences, answered differently -- the same split
    ``tests/utils/corpus.py`` draws for the experiment corpus:

    * **the directory is missing entirely** -- the public export does not ship
      ``scripts/sim2rank/`` (the in-package half under
      ``mriforge.core.metrics.meta_evaluation`` does, and stands alone). There
      is no subject to sweep, so the honest answer is to decline and say so.
    * **the directory is there and something inside it moved** -- a real
      defect, and every assertion below still runs to catch it. This helper
      checks the *directory*, so a relocated ``sim2rank.py`` inside a present
      package keeps failing ``test_the_docstring_is_actually_readable`` as it
      always did.

    A skip is not the vacuity this module's docstring forbids. The trap named
    there is a sweep that reports **green** having read zero files; that is
    exactly what ``test_no_new_unqualified_counts_across_sim2rank`` did in the
    export before this guard existed -- measured PASSED against an absent
    ``scripts/sim2rank/``, beside four loud failures that made it easy to miss.
    A skip declines to answer and is counted separately.

    ``test_the_scan_root_is_populated`` remains the **sole** owner of the
    present-but-empty case (non-negotiable 17); do not re-assert a scanned
    count in the individual sweeps.
    """
    if not SIM2RANK_DIR.is_dir():
        pytest.skip(
            f"{SIM2RANK_DIR} is not present in this tree. scripts/sim2rank/ is "
            f"deliberately not shipped in the public export, so there is no "
            f"docstring corpus to sweep here. The detector tests below do not "
            f"need it and still run."
        )


# --------------------------------------------------------------------------
# The guard, against the real docstrings
# --------------------------------------------------------------------------


def test_the_docstring_is_actually_readable() -> None:
    """Fail loudly if the file moves, instead of checking an empty string.

    Every assertion below is vacuously true against ``""``, so a relocated
    script would turn this whole module green while checking nothing.
    """
    _require_subject()
    assert SIM2RANK.is_file(), f"{SIM2RANK} not found -- fix the path, not this test"
    doc = module_docstring()
    assert doc.strip(), "module docstring is empty"
    assert "Sim2Rank" in doc


def test_the_scan_root_is_populated() -> None:
    """Same vacuity trap, one level up: an empty root passes every sweep."""
    _require_subject()
    assert SIM2RANK_DIR.is_dir(), f"{SIM2RANK_DIR} not found -- fix the path"
    assert len(list(iter_docstrings(SIM2RANK_DIR))) > 100


def test_no_stale_per_image_metric_count() -> None:
    """Any stated per-image count must equal ``len(PER_IMAGE_SPECS)``."""
    pytest.importorskip("torch")
    # The driver half of sim2rank is not shipped in the public export, so this
    # import raises ModuleNotFoundError there -- which reads as a broken install
    # rather than as the deliberate scope decision it is. Guarding here keeps the
    # other 20 tests in this file, which need only the docstrings.
    pytest.importorskip("scripts.sim2rank.metrics_list")
    from scripts.sim2rank.metrics_list import PER_IMAGE_SPECS

    stale = stale_metric_counts(module_docstring(), len(PER_IMAGE_SPECS))
    assert not stale, (
        f"docstring states {stale} per-image metrics; live PER_IMAGE_SPECS has "
        f"{len(PER_IMAGE_SPECS)}. Name the SSOT instead of restating the count."
    )


def test_axis_counts_are_bank_qualified() -> None:
    """A bare axis count describes a sweep that only the legacy bank runs."""
    _require_subject()
    unqualified = unqualified_axis_counts(module_docstring())
    assert not unqualified, (
        f"docstring states {unqualified} axes without naming a bank. "
        "--axis-bank defaults to 'registry'; only 'legacy'/AXIS_CONFIGS is a "
        "fixed count."
    )


def test_docstring_names_the_bank_selector() -> None:
    """The reader must learn which bank actually runs -- that was the defect."""
    _require_subject()
    assert "--axis-bank" in module_docstring()


def test_no_new_unqualified_counts_across_sim2rank() -> None:
    """Ratchet over the whole package: no count may state a bare number.

    The allowlist holds the claims this sweep found on first run and #1521 did
    **not** adjudicate -- they are tracked as a body of work, not blessed. It
    is keyed by exact matched text, so editing a waived claim un-waives it
    (a file-level key would waive that file forever). Regenerating it to make
    this test green is the one thing it must never be used for.
    """
    _require_subject()
    findings = sorted(set(survey(SIM2RANK_DIR)) - load_allowlist())
    assert not findings, "\n".join(
        [f"{len(findings)} unqualified count(s) in sim2rank docstrings:"]
        + [f"  {rel}::{name}: {text!r}" for rel, name, text in findings]
    )


def test_every_waiver_still_describes_something_real() -> None:
    """A waiver that matches nothing is stale -- the claim was fixed or moved."""
    _require_subject()
    dead = sorted(load_allowlist() - set(survey(SIM2RANK_DIR)))
    assert not dead, f"{len(dead)} allowlist entries match no live docstring: {dead}"


# --------------------------------------------------------------------------
# The scope guard itself, planted both ways (non-negotiable 15)
#
# A guard that declines is one edit away from a guard that always declines, and
# that failure is silent: the sweeps above would report "skipped" forever while
# the corpus rotted. Both directions are pinned here, and both run in ANY tree
# -- they drive `_require_subject` through a monkeypatched root rather than
# through the real one, so neither is inert in the export.
# --------------------------------------------------------------------------


def test_the_scope_guard_declines_when_the_package_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export case: no subject, so decline visibly rather than sweep zero files."""
    monkeypatch.setattr(sys.modules[__name__], "SIM2RANK_DIR", tmp_path / "definitely_not_here")
    with pytest.raises(pytest.skip.Exception):
        _require_subject()


def test_the_scope_guard_is_inert_when_the_package_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev case: the guard must not swallow a real corpus regression.

    Paired with the test above on purpose. Dropping the ``is_dir()`` condition
    turns that one red; making the guard unconditional turns this one red.

    **The skip is caught rather than propagated, and that is the whole test.**
    Letting it escape would report this case as *skipped*, and a skip is not a
    failure -- an unconditional ``pytest.skip`` in the helper then silences the
    six sweeps above AND this plant in one edit, for a run that reads ``16
    passed, 7 skipped`` with nothing red. That was measured on the first draft
    of this file, which is the same always-green-by-declining shape the commit
    adding the guard exists to remove, reproduced one level up.
    """
    present = tmp_path / "sim2rank"
    present.mkdir()
    monkeypatch.setattr(sys.modules[__name__], "SIM2RANK_DIR", present)
    try:
        _require_subject()
    except pytest.skip.Exception as exc:  # pragma: no cover - the planted case
        pytest.fail(
            f"_require_subject() declined on a PRESENT package at {present}: {exc}. "
            f"The guard must be inert whenever the corpus is there, or every sweep "
            f"in this module silently stops running."
        )


# --------------------------------------------------------------------------
# Planted violations (non-negotiable 15) -- every detector, every shape.
# A detector that only ever fires is as useless as one that never does, so
# each plant is paired with the real post-fix text it must accept.
# --------------------------------------------------------------------------

_PRE_FIX_METRICS = "3. Evaluate 137 per-image metrics across all timesteps"
_POST_FIX_METRICS = "3. Evaluate every metric in ``PER_IMAGE_SPECS`` across all"
_PRE_FIX_AXES = "2. Progressive Digital Twin degradation sweep (13 axes)"
_POST_FIX_AXES = "default ``registry``; ``legacy`` = the 13-axis ``AXIS_CONFIGS``"

#: The real #1521 text, in the phrasing that defeated the per-image regex.
_PRE_FIX_BARE = "A sim2rank run scores ~137 metrics. Until run_all_generations"
_POST_FIX_BARE = "A sim2rank run scores every metric in ``PER_IMAGE_SPECS`` -- and"
_PRE_FIX_CATEGORIES = "137 metrics is not a legible axis; ~13 categories is."
_POST_FIX_CATEGORIES = "The full metric axis is too long to read; the category axis"


def test_metric_detector_fires_on_the_real_pre_fix_text() -> None:
    assert stale_metric_counts(_PRE_FIX_METRICS, 139) == [137]


def test_metric_detector_accepts_a_matching_count() -> None:
    """Discrimination: a correct restated count is not a finding."""
    assert stale_metric_counts("Evaluate 139 per-image metrics", 139) == []


def test_metric_detector_accepts_the_real_post_fix_text() -> None:
    assert stale_metric_counts(_POST_FIX_METRICS, 139) == []


def test_axis_detector_fires_on_the_real_pre_fix_text() -> None:
    assert unqualified_axis_counts(_PRE_FIX_AXES) == [13]


def test_axis_detector_accepts_the_real_post_fix_text() -> None:
    """Discrimination: the same 13, qualified by its bank, is legitimate."""
    assert unqualified_axis_counts(_POST_FIX_AXES) == []


def test_axis_detector_catches_a_registry_count_too() -> None:
    """The trap is the missing bank, not the specific number."""
    assert unqualified_axis_counts("a sweep over 30 axes") == [30]


def test_bare_metric_phrasing_is_caught_where_per_image_regex_was_blind() -> None:
    """Blindness (3): the old detector needed the words ``per-image``."""
    assert stale_metric_counts(_PRE_FIX_BARE, 139) == []
    assert unqualified_counts(_PRE_FIX_BARE, _ANY_METRIC_COUNT) == ["~137 metrics"]


def test_bare_metric_detector_accepts_the_real_post_fix_text() -> None:
    assert unqualified_counts(_POST_FIX_BARE, _ANY_METRIC_COUNT) == []


def test_category_detector_fires_on_the_real_pre_fix_text() -> None:
    """Blindness (4): there was no category detector at all."""
    assert unqualified_counts(_PRE_FIX_CATEGORIES, _CATEGORY_COUNT) == ["~13 categories"]


def test_category_detector_accepts_the_real_post_fix_text() -> None:
    assert unqualified_counts(_POST_FIX_CATEGORIES, _CATEGORY_COUNT) == []


def test_category_detector_accepts_a_qualified_count() -> None:
    """Discrimination: naming the collection makes the number checkable."""
    assert unqualified_counts("18 categories across ``METRIC_SPECS``", _CATEGORY_COUNT) == []


# --- the discovery mechanism itself, planted against (blindness (1) and (2)) ---


def _plant(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_discovery_finds_a_planted_sibling_module(tmp_path: Path) -> None:
    """A constant scan root could not have seen this file at all."""
    root = _plant(tmp_path, "sibling.py", '"""Scores 86 metrics in one pass."""\n')
    assert survey(root) == [("sibling.py", "<module>", "86 metrics")]


def test_discovery_reads_function_and_class_docstrings(tmp_path: Path) -> None:
    """Two of #1521's three sites were function docstrings, not module ones."""
    root = _plant(
        tmp_path,
        "nested.py",
        '"""Fine -- names ``PER_IMAGE_SPECS``."""\n\n\n'
        'class Roller:\n    """Rolls up ~13 categories."""\n\n'
        '    def run(self):\n        """Collapses 173 metrics."""\n',
    )
    assert sorted(survey(root)) == [
        ("nested.py", "Roller", "~13 categories"),
        ("nested.py", "run", "173 metrics"),
    ]


def test_discovery_accepts_a_planted_qualified_count(tmp_path: Path) -> None:
    """Discrimination for the sweep: the plant is the bareness, not the digit."""
    root = _plant(tmp_path, "ok.py", '"""Scores 139 metrics -- all of ``PER_IMAGE_SPECS``."""\n')
    assert survey(root) == []
