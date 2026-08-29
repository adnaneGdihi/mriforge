"""Tests for VolumeBlockedSliceSampler — the epoch order that makes all_slices affordable.

The sampler exists for an IO property, so the tests assert that property directly
(decodes per epoch under a simulated LRU) rather than only that it emits indices. A
sampler that yielded a correct permutation in a pathological order would satisfy every
coverage assertion and still reinstate the starvation it was written to prevent.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict

import pytest

from mriforge.data.samplers import VolumeBlockedSliceSampler


def _corpus(groups: int = 3, fields: int = 5, slices: int = 20):
    """all_pairs over `groups` subject-contrast groups: the real container topology.

    Combinatorial pairing is what gives the sampler its leverage — a 5-field group is
    20 ordered pairs drawn from 5 volumes — so the fixture reproduces that rather than
    a flat list of independent items.
    """
    container_volumes: list[frozenset[str]] = []
    for g in range(groups):
        vols = [f"g{g}_f{i}" for i in range(fields)]
        container_volumes.extend(frozenset(p) for p in itertools.permutations(vols, 2))
    index_map = [(c, s) for c in range(len(container_volumes)) for s in range(slices)]
    return index_map, container_volumes


def _decodes(order, index_map, container_volumes, budget: int) -> int:
    """Volume decodes an LRU of `budget` would perform for this visit order."""
    lru: OrderedDict[str, int] = OrderedDict()
    misses = 0
    for dataset_idx in order:
        container_idx, _ = index_map[dataset_idx]
        for volume in container_volumes[container_idx]:
            if volume in lru:
                lru.move_to_end(volume)
            else:
                misses += 1
                lru[volume] = 1
                if len(lru) > budget:
                    lru.popitem(last=False)
    return misses


class TestCoverage:
    def test_yields_every_sample_exactly_once(self) -> None:
        index_map, container_volumes = _corpus()
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6
        )
        order = list(sampler)
        assert sorted(order) == list(range(len(index_map)))
        assert len(sampler) == len(index_map)

    def test_shuffled_and_unshuffled_cover_the_same_set(self) -> None:
        index_map, container_volumes = _corpus()
        kwargs = {"max_resident_volumes": 6}
        a = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, shuffle=True, **kwargs
            )
        )
        b = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, shuffle=False, **kwargs
            )
        )
        assert sorted(a) == sorted(b)
        assert a != b, "shuffle=True produced the deterministic order"


class TestTheIOPropertyItExistsFor:
    def test_blocked_order_needs_far_fewer_decodes_than_a_plain_shuffle(self) -> None:
        """THE reason the sampler exists. Measured, not asserted qualitatively.

        A plain shuffled epoch makes nearly every sample a fresh ~226 MB decode against
        a working set no budget can hold; the blocked order amortises one decode over
        every slice of every container that volume takes part in.
        """
        import random

        index_map, container_volumes = _corpus()
        budget = 6

        naive = list(range(len(index_map)))
        random.Random(0).shuffle(naive)
        blocked = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=budget
            )
        )

        naive_decodes = _decodes(naive, index_map, container_volumes, budget)
        blocked_decodes = _decodes(blocked, index_map, container_volumes, budget)

        # An order-of-magnitude bound, not the exact ratio: the point is the regime
        # change, and pinning the precise number would make the test brittle against
        # a harmless packing tweak.
        assert blocked_decodes * 10 < naive_decodes

    def test_decodes_approach_the_number_of_distinct_volumes(self) -> None:
        """The floor: with a budget covering a group, each volume is decoded ~once."""
        index_map, container_volumes = _corpus(groups=1, fields=5, slices=20)
        distinct = len({v for f in container_volumes for v in f})
        order = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=6
            )
        )
        assert _decodes(order, index_map, container_volumes, 6) <= distinct * 2

    def test_describe_reports_the_amortisation(self) -> None:
        index_map, container_volumes = _corpus(groups=1)
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6
        )
        text = sampler.describe()
        assert "samples per decode" in text
        assert "max_resident_volumes=6" in text


class TestPreconditionsFailLoud:
    def test_a_budget_below_the_widest_container_raises(self) -> None:
        """No silent thrash: below the container arity every item evicts the next one's
        volume, so the mode would 'work' at a hit rate of zero (pitfall #9)."""
        index_map, container_volumes = _corpus()
        with pytest.raises(ValueError, match="smaller than the widest container"):
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=1
            )

    def test_no_container_volumes_raises(self) -> None:
        with pytest.raises(ValueError, match="needs per-container volume paths"):
            VolumeBlockedSliceSampler([], [], max_resident_volumes=6)


class TestDeterminism:
    def test_same_seed_and_epoch_gives_the_same_order(self) -> None:
        index_map, container_volumes = _corpus()
        kwargs = {"max_resident_volumes": 6, "seed": 7}
        a = list(VolumeBlockedSliceSampler(index_map, container_volumes, **kwargs))
        b = list(VolumeBlockedSliceSampler(index_map, container_volumes, **kwargs))
        assert a == b

    def test_set_epoch_changes_the_order_and_keeps_coverage(self) -> None:
        index_map, container_volumes = _corpus()
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6, seed=7
        )
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        assert first != second
        assert sorted(first) == sorted(second)
