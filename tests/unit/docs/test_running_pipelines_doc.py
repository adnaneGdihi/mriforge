"""Doc-contract guard for ``docs/running_pipelines.rst`` (the run-modes guide).

A how-to guide that names CLI verbs and cross-references other pages rots the
moment a verb is renamed or a page moved. This test parses the guide as text
(importing ``spectramr.cli.app`` is torch-free) and pins three invariants:

* every ``spectramr <verb>`` / ``python -m spectramr.cli <verb>`` example names a verb
  the parser actually registers,
* every ``:doc:`target``` cross-reference resolves to a real ``docs/<target>.rst``,
* the guide is wired into the ``index.rst`` toctree.

So the guide cannot drift out of sync with ``build_parser`` or the doc tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[3] / "docs"
GUIDE = DOCS / "running_pipelines.rst"

# ``spectramr <verb>`` and ``python -m spectramr.cli <verb>``. Only these two forms
# introduce a CLI verb; a Python import of the package (``from spectramr import
# fit``) must NOT match. A literal ``<verb>`` placeholder or a ``--flag`` is
# excluded by requiring the captured token to start with a letter.
#
# The two leading lookbehinds are load-bearing, and were NOT needed before the
# 2026-08 rename. Until then the console script and the import package were
# spelled differently (one hyphenated, one underscored), so a pattern anchored
# on the console-script spelling could not match an import line at all -- the
# discrimination was a free consequence of the naming. The rename collapsed
# both onto the single token ``spectramr`` and destroyed that channel:
# ``from spectramr import fit`` began parsing as the CLI verb ``import``. The
# lookbehinds now state explicitly what the old spelling encoded implicitly.
# ``from `` and ``import `` are fixed width, so both are legal lookbehinds.
_VERB_RE = re.compile(
    r"(?<!from )(?<!import )\b(?:spectramr|spectramr\.cli)\s+([a-z][a-z0-9_-]*)"
)
_DOCREF_RE = re.compile(r":doc:`([^`]+)`")


def _registered_subcommands() -> set[str]:
    from spectramr.cli.app import build_parser

    parser = build_parser()
    subact = next(
        a
        for a in parser._actions
        if getattr(a, "choices", None) and "train" in a.choices
    )
    return set(subact.choices)


def test_guide_exists():
    assert GUIDE.is_file(), f"missing run-modes guide: {GUIDE}"


def test_every_referenced_verb_is_registered():
    text = GUIDE.read_text()
    referenced = set(_VERB_RE.findall(text))
    # ``torchrun`` / ``python`` are command prefixes, not spectramr verbs; the regex
    # only matches the spectramr/spectramr.cli prefixes, so referenced holds verbs.
    registered = _registered_subcommands()
    unknown = referenced - registered
    assert not unknown, (
        f"running_pipelines.rst references CLI verbs that build_parser does not "
        f"register: {sorted(unknown)} (registered: {sorted(registered)})"
    )
    # Sanity: the guide must actually exercise the core lifecycle verbs.
    for verb in ("doctor", "audit", "train", "sanity_check", "infer", "report"):
        assert verb in referenced, f"guide does not document the '{verb}' mode"


def test_doc_cross_references_resolve():
    text = GUIDE.read_text()
    for target in set(_DOCREF_RE.findall(text)):
        # ``:doc:`name``` -> docs/name.rst (targets here are all top-level stems).
        assert (DOCS / f"{target}.rst").is_file(), (
            f"running_pipelines.rst :doc:`{target}` has no docs/{target}.rst"
        )


def test_guide_is_in_toctree():
    index = (DOCS / "index.rst").read_text()
    assert "running_pipelines" in index, (
        "running_pipelines is not wired into the index.rst toctree"
    )


# --- Discrimination probes for _VERB_RE -------------------------------------
#
# _VERB_RE decides what the three invariants above are even applied to, so a
# regex that silently stops matching turns every one of them green and blind.
# Before the 2026-08 rename the CLI/module discrimination came free from the
# two spellings being lexically different; it is now carried by two
# lookbehinds, i.e. by code that can be broken. These pin both directions: the shapes that MUST
# yield a verb, and the shapes that MUST NOT.
#
# The last case is the one that matters most. A regex that matched nothing
# would pass every other probe here and pass the three invariants too, because
# an empty ``referenced`` set is trivially a subset of ``registered``. Pinning
# that an *unregistered* verb is still captured is what makes this guard a
# guard rather than a decoration.

_VERB_RE_MUST_MATCH = [
    ("spectramr train --config a.yaml", "train"),
    ("python -m spectramr.cli audit arm.yaml", "audit"),
    ("$ spectramr infer-dataset x", "infer-dataset"),
    # An unregistered verb must still be captured, or the guard cannot fail.
    ("spectramr frobnicate --x", "frobnicate"),
]

_VERB_RE_MUST_NOT_MATCH = [
    "from spectramr import fit, make_model",
    "from spectramr.cli import app",
    "import spectramr",
]


@pytest.mark.parametrize(("line", "expected"), _VERB_RE_MUST_MATCH)
def test_verb_regex_captures_cli_invocations(line: str, expected: str) -> None:
    assert _VERB_RE.findall(line) == [expected]


@pytest.mark.parametrize("line", _VERB_RE_MUST_NOT_MATCH)
def test_verb_regex_ignores_python_imports_of_the_package(line: str) -> None:
    """``from spectramr import fit`` is not the CLI verb ``import``.

    This is the exact regression the rename introduced: the pre-rename spelling
    made the two cases lexically distinct, so no lookbehind was needed.
    """
    assert _VERB_RE.findall(line) == []
