"""``split_index_three_way`` — disjointness is the whole point.

``data.test_split`` is declared by 468 experiment YAMLs and read by nothing
(#665): there is no held-out test set, and ``make_dataset(split="test")``
returns the VALIDATION loader. This is the splitter that makes a real one
possible.

The properties below are the ones that fail SILENTLY if got wrong -- an
overlapping val/test still trains, still reports numbers, and the numbers are
simply wrong.
"""

from __future__ import annotations

import pytest

from spectramr.data.split_utils import split_index, split_index_three_way


@pytest.mark.parametrize("n", [3, 4, 10, 97])
def test_the_three_splits_are_pairwise_disjoint_and_cover_everything(n):
    """The leak this exists to prevent: tuning and reporting on the same items."""
    items = list(range(n))
    train, val, test = split_index_three_way(items, 0.2, 0.2)

    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set(), (
        "validation and test overlap — the set you tune on is inside the set you "
        "report, which is exactly what a held-out split exists to prevent"
    )
    assert sorted(train + val + test) == items


@pytest.mark.parametrize("n", [2, 5, 40])
def test_no_test_split_is_byte_identical_to_the_two_way_split(n):
    """``test_split: 0`` must leave existing arms untouched.

    468 YAMLs set ``test_split``, but the ones that set it to 0 -- and every arm
    that adds the key later -- must keep the split they already had.
    """
    items = list(range(n))
    train3, val3, test3 = split_index_three_way(items, 0.25, 0.0)
    train2, val2 = split_index(items, 0.25)

    assert test3 == []
    assert train3 == train2
    assert val3 == val2


def test_validation_fraction_applies_to_the_post_test_remainder():
    """Adding a test set must not silently shrink the validation set as well.

    The fractions are applied in sequence, not independently against the whole
    index -- independent application is what makes them overlap.
    """
    items = list(range(100))
    _, val, test = split_index_three_way(items, 0.1, 0.2)

    assert len(test) == 20
    # 10% of the 80 that remain, not 10% of 100.
    assert len(val) == 8


def test_test_items_come_from_the_tail_and_are_contiguous():
    """Deterministic and inspectable: the same index yields the same test set."""
    items = list(range(10))
    _, _, test = split_index_three_way(items, 0.2, 0.2)
    assert test == [8, 9]

    # Deterministic across calls — no hidden shuffling.
    assert split_index_three_way(items, 0.2, 0.2)[2] == test


def test_too_few_items_raises_rather_than_returning_an_empty_test_set():
    """An empty test set would be reported from as though it were real."""
    with pytest.raises(ValueError, match="at least 3"):
        split_index_three_way([1, 2], 0.2, 0.2)


def test_two_items_suffice_when_no_validation_set_is_requested():
    train, val, test = split_index_three_way([1, 2], 0.0, 0.5)
    assert val == []
    assert len(test) == 1 and len(train) == 1


def test_empty_index_is_empty_everywhere():
    assert split_index_three_way([], 0.2, 0.2) == ([], [], [])
