"""``docs/known_limitations.rst`` states schema facts; pin them to the live schema.

This page ships. Its numbers are the ones a reader trusts instead of measuring, so
a stale figure here is worse than no figure -- and the page cannot be checked by
re-running the command it publishes, because that command was itself the source of
the error it would be confirming: it unwrapped a fixed number of ``typing.get_args``
levels and therefore could not see a discriminated union's members, which is where
three of the eight ``extra="allow"`` classes live.

So these tests compare the page against a **fixed-point** walk of the live schema
rather than against a frozen constant. When the schema drifts the failure names the
number to update, instead of the page quietly going stale.

**The walk has TWO roots, and that is the whole point.** Seeding it from
``TrainingSettings`` alone -- which is what this module and the page's published
snippet both did until they were corrected together -- reports 8 open classes
where 20 are reachable. The 12 it cannot see are the per-paradigm top-level
schemas in ``_MODE_DISPATCH``: a YAML selects one by ``training_mode``, not by
being a *field* of anything, so no amount of field-walking from
``TrainingSettings`` will ever arrive at ``TrainingConfigSE3Navigator``.

That is the same defect as the discriminated-union one recorded below, reached
by a third route, and it went unseen for the reason non-negotiable 15 gives: the
page and its guard were not independent. The guard re-derived the page's own
choice of root, so the two agreed with each other and both were wrong. A test
that shares its subject's blind spot is a fidelity check, not a truth check.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from spectramr.config.schemas import training as training_schemas
from spectramr.config.settings import TrainingSettings

DOC = Path(__file__).resolve().parents[3] / "docs" / "known_limitations.rst"


def _models_in(annotation: object) -> set[type[BaseModel]]:
    """Every ``BaseModel`` in an annotation, unwrapping ``get_args`` to a fixed point.

    A fixed depth is not enough: a discriminated union is annotated
    ``Optional[Annotated[A | B | ..., FieldInfo(discriminator=...)]]``, so its members
    first appear three levels down.
    """
    found: set[type[BaseModel]] = set()
    stack = [annotation]
    while stack:
        item = stack.pop()
        if isinstance(item, type) and issubclass(item, BaseModel):
            found.add(item)
        stack.extend(typing.get_args(item))
    return found


def _roots() -> list[type[BaseModel]]:
    """Every schema a YAML can select, from both directions it can select one.

    ``TrainingSettings`` covers everything reachable as a *field*.
    ``_MODE_DISPATCH`` covers the per-paradigm top-level schemas, which
    ``create_training_config`` picks by ``training_mode`` -- a mode with no
    entry falls through to the default schema, so the table's values are the
    complete second root set. It is private and imported anyway because it is
    the SSOT for that dispatch and has no public alias; a hand-written list here
    would be a second owner that goes stale on the next paradigm.
    """
    roots: list[type[BaseModel]] = [TrainingSettings]
    for name in sorted(set(training_schemas._MODE_DISPATCH.values())):
        roots.append(getattr(training_schemas, name))
    return roots


def _walk() -> set[type[BaseModel]]:
    seen: set[type[BaseModel]] = set()

    def visit(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        for field in model.model_fields.values():
            for sub in _models_in(field.annotation):
                visit(sub)

    for root in _roots():
        visit(root)
    return seen


@pytest.fixture(scope="module")
def reachable() -> set[type[BaseModel]]:
    return _walk()


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _policy(model: type[BaseModel]) -> str:
    return model.model_config.get("extra") or "ignore"


def test_the_allow_class_list_is_the_live_set(reachable, doc_text):
    """The bulleted list of open schemas must be exactly the live ``allow`` set.

    An omission is the harmful direction: the page tells the reader which blocks
    silently carry a misspelled key, so a class missing from it is a class the
    reader has been told is safe.
    """
    section = doc_text.split('are ``extra="allow"``:', 1)[1].split(".. [#du]", 1)[0]
    documented = sorted(re.findall(r"\* ``(\w+)``", section))
    live = sorted(m.__name__ for m in reachable if _policy(m) == "allow")
    assert documented == live


def test_the_class_counts_match_the_live_schema(reachable, doc_text):
    total = len(reachable)
    forbid = sum(1 for m in reachable if _policy(m) == "forbid")
    ignore = sum(1 for m in reachable if _policy(m) == "ignore")

    # The noun is prose ("model"/"schema" classes) and has already been reworded
    # once; the count and the sentence frame are what this test is about, so the
    # regex must not break on a wording change. A miss says so explicitly rather
    # than surfacing as AttributeError on None.
    m = re.search(r"of the (\d+) \w+ classes reachable", doc_text)
    assert m, "the doc no longer states a reachable-class total in the expected frame"
    stated_total = int(m.group(1))
    stated_forbid = int(re.search(r"\*\*(\d+) are\*\* ``extra=\"forbid\"``", doc_text).group(1))
    stated_ignore = int(re.search(r"\*\*(\d+) are\*\* ``extra=\"ignore\"``", doc_text).group(1))

    assert (stated_total, stated_forbid, stated_ignore) == (total, forbid, ignore)


_COUNT_WORDS = {
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    20: "Twenty",
    21: "Twenty-one",
    22: "Twenty-two",
}

# ``[\w-]``, not ``\w``: the table above spells 21 and 22 with a hyphen, which
# ``\w`` does not match -- see the pairing test at the foot of this module.
_HEADING_RE = r"^([\w-]+) schema classes accept \*any\* key$"


def test_the_heading_count_word_matches_the_list_length(doc_text):
    """The heading says a number in words; a list edit must not leave it behind."""
    section = doc_text.split('are ``extra="allow"``:', 1)[1].split(".. [#du]", 1)[0]
    n = len(re.findall(r"\* ``(\w+)``", section))
    m = re.search(_HEADING_RE, doc_text, re.M)
    assert m, "the doc no longer states the open-class count as a word in its heading"
    assert m.group(1) == _COUNT_WORDS[n]


def test_every_spelling_in_the_table_is_matchable_by_the_heading_regex():
    r"""The blindness this pair had: a table entry the regex cannot match.

    ``_HEADING_RE`` was ``^(\w+) schema classes ...`` while the table already
    offered ``Twenty-one`` and ``Twenty-two`` -- and ``\w`` excludes ``-``. So the
    day the live count reached 21 the test above began failing with
    ``AttributeError`` on *any* page content, correct spelling included, and the
    page could not be made green without editing this file. The regex had only
    ever been exercised on the un-hyphenated half of its own vocabulary.
    """
    for n, word in _COUNT_WORDS.items():
        probe = f"{word} schema classes accept *any* key"
        m = re.search(_HEADING_RE, probe, re.M)
        assert m and m.group(1) == word, f"{word!r} (n={n}) is unmatchable by _HEADING_RE"


def test_the_published_walk_reaches_a_discriminated_union_member(reachable):
    """The regression this page's own re-measure command used to have.

    ``ColdParams`` is reachable only through the discriminated union at
    ``TrainingStrategyConfigSchema.diffusion``. Any walk that cannot see it will
    under-report the open-schema list, which is how the page came to say five.
    """
    assert "ColdParams" in {m.__name__ for m in reachable}


def test_the_walk_reaches_a_mode_dispatch_only_schema(reachable):
    """The second root, pinned the way the union member above is pinned.

    ``TrainingConfigSE3Navigator`` is a top-level schema selected by
    ``training_mode: se3_equivariant_navigator``. It is a field of nothing, so a
    walk seeded only from ``TrainingSettings`` never reaches it -- and it is
    ``extra="allow"``, so under-reporting it tells the reader a block that
    silently carries misspelled keys is safe.
    """
    assert "TrainingConfigSE3Navigator" in {m.__name__ for m in reachable}


def test_both_roots_contribute(reachable):
    """Neither root may become redundant without someone noticing.

    If ``_MODE_DISPATCH``'s schemas ever become reachable as fields, this test
    says so rather than leaving a second root that looks load-bearing and is
    not. It fails in the useful direction too: a dispatch table that stops
    resolving leaves the field-walk alone and this goes red.
    """
    field_only: set[type[BaseModel]] = set()

    def visit(model):
        if model in field_only:
            return
        field_only.add(model)
        for field in model.model_fields.values():
            for sub in _models_in(field.annotation):
                visit(sub)

    visit(TrainingSettings)
    assert len(reachable) > len(field_only), (
        "the _MODE_DISPATCH root added nothing -- either it is now redundant "
        "(delete it and say so) or it stopped resolving (fix it)"
    )
