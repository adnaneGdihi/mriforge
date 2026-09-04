"""The both-spellings corpus reader must not paper over a wrong lookup."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.renames import RENAMES
from tests.utils.corpus_keys import legacy_spellings, read_key, spelling_used

# A record that is genuinely `fold` today. Resolved from the table rather than
# written down, so this file cannot outlive the posture it assumes.
_FOLDING = next(
    (r for r in RENAMES.values() if r.posture == "fold"),
    None,
)


class TestLegacySpellings:
    def test_only_folding_records_are_offered(self) -> None:
        """A `raise` record is retired: an arm declaring it does not load.

        Accepting one would let a guard call a config healthy that cannot
        start -- the opposite of what these guards exist for.
        """
        retired = [r for r in RENAMES.values() if r.posture == "raise"]
        if not retired:
            pytest.skip("no retired records on this tree")
        for record in retired:
            assert record.legacy not in legacy_spellings(record.canonical)

    def test_a_folding_record_is_offered(self) -> None:
        """Anti-vacuity for the test above: the filter must not reject all."""
        assert _FOLDING is not None, "no folding records left — drain complete?"
        assert _FOLDING.legacy in legacy_spellings(_FOLDING.canonical)


class TestReadKey:
    def test_reads_the_canonical_path(self) -> None:
        assert _FOLDING is not None
        doc = _nest(_FOLDING.canonical, "canonical-value")
        assert read_key(doc, _FOLDING.canonical) == "canonical-value"

    def test_falls_back_to_the_legacy_path(self) -> None:
        assert _FOLDING is not None
        doc = _nest(_FOLDING.legacy, "legacy-value")
        assert read_key(doc, _FOLDING.canonical) == "legacy-value"

    def test_canonical_wins_when_both_are_present(self) -> None:
        """Matches the loader: the fold only fills a destination left empty."""
        assert _FOLDING is not None
        doc = _nest(_FOLDING.canonical, "canonical-value")
        _merge(doc, _nest(_FOLDING.legacy, "legacy-value"))
        assert read_key(doc, _FOLDING.canonical) == "canonical-value"

    def test_absent_is_the_default_not_a_crash(self) -> None:
        assert _FOLDING is not None
        assert read_key({}, _FOLDING.canonical) is None
        assert read_key({}, _FOLDING.canonical, default=8) == 8

    def test_a_scalar_midway_does_not_raise(self) -> None:
        """Corpus YAML is not always the shape a guard expects."""
        assert _FOLDING is not None
        head = _FOLDING.canonical.split(".")[0]
        assert read_key({head: "not-a-mapping"}, _FOLDING.canonical) is None

    def test_a_falsy_declared_value_is_not_treated_as_absent(self) -> None:
        """`0` and `False` are declarations. A truthiness test here would send
        the reader on to the legacy path and return the wrong branch's value."""
        assert _FOLDING is not None
        doc = _nest(_FOLDING.canonical, 0)
        _merge(doc, _nest(_FOLDING.legacy, 99))
        assert read_key(doc, _FOLDING.canonical) == 0


class TestSpellingUsed:
    def test_reports_which_branch_answered(self) -> None:
        assert _FOLDING is not None
        assert spelling_used(_nest(_FOLDING.canonical, 1), _FOLDING.canonical) == _FOLDING.canonical
        assert spelling_used(_nest(_FOLDING.legacy, 1), _FOLDING.canonical) == _FOLDING.legacy
        assert spelling_used({}, _FOLDING.canonical) is None


def _nest(path: str, value: object) -> dict:
    parts = path.split(".")
    doc: dict = {}
    node = doc
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return doc


def _merge(into: dict, other: dict) -> None:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value
