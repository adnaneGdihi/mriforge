"""Unit tests for :mod:`spectramr.infrastructure.training.utils.mask_table_cache`.

The cache exists to remove a blocking device-to-host copy from the validation
mask path, so the tests that matter are the ones that would go red if it ever
served a mask the uncached resolver would not have produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.utils.mask_table_cache import MaskTableCache


def _accelerator(seed: int | None = 42, enforce_nested: bool = False) -> SimpleNamespace:
    """A stand-in shaped like ``ColdDiffusionAccelerator``: wrapper + inner."""
    return SimpleNamespace(
        accelerator=SimpleNamespace(seed=seed),
        enforce_nested=enforce_nested,
        seed="WRAPPER-SENTINEL",
    )


def _key(**kw):
    base = {
        "acceleration_type": "variable_density",
        "image_shape": (8, 8),
        "acceleration_factor": 4,
        "device": torch.device("cpu"),
        "accelerator": _accelerator(),
    }
    base.update(kw)
    return MaskTableCache.build_key(**base)


class TestBuildKey:
    """The key must move whenever mask content could move."""

    def test_reads_seed_from_inner_accelerator_not_wrapper(self) -> None:
        """The wrapper's ``seed`` is a read-only sentinel; the inner one is real.

        Keying on the wrapper would collapse every seed onto one entry and serve
        a stale cascade after a seed change, so this asserts the wrapper value
        is absent and the inner value present.
        """
        key = _key(accelerator=_accelerator(seed=1234))
        assert 1234 in key
        assert "WRAPPER-SENTINEL" not in key

    @pytest.mark.parametrize(
        ("field", "changed"),
        [
            ("acceleration_type", "radial"),
            ("image_shape", (16, 8)),
            ("acceleration_factor", 8),
            ("device", torch.device("meta")),
        ],
    )
    def test_key_discriminates_each_field(self, field: str, changed: object) -> None:
        assert _key() != _key(**{field: changed})

    def test_key_discriminates_inner_seed(self) -> None:
        assert _key() != _key(accelerator=_accelerator(seed=99))

    def test_key_discriminates_enforce_nested(self) -> None:
        """``_generate_batch_masks_dynamic`` toggles this; it changes content."""
        assert _key() != _key(accelerator=_accelerator(enforce_nested=True))

    def test_identical_inputs_collide_deliberately(self) -> None:
        assert _key() == _key()

    def test_accelerator_without_inner_falls_back_to_its_own_seed(self) -> None:
        bare = SimpleNamespace(seed=7, enforce_nested=False)
        assert 7 in MaskTableCache.build_key(
            "variable_density", (8, 8), 4, torch.device("cpu"), bare
        )


class TestTableFor:
    """Build-once semantics and faithful stacking."""

    def test_builds_once_then_serves_from_cache(self) -> None:
        cache = MaskTableCache()
        calls: list[int] = []

        def build_one(t: int) -> torch.Tensor:
            calls.append(t)
            return torch.full((1, 4, 4), float(t))

        key = _key()
        first = cache.table_for(key, 5, build_one)
        assert calls == [0, 1, 2, 3, 4]
        second = cache.table_for(key, 5, build_one)
        assert calls == [0, 1, 2, 3, 4], "second request must not rebuild"
        assert first is second

    def test_distinct_keys_build_distinct_tables(self) -> None:
        cache = MaskTableCache()
        build = lambda t: torch.full((1, 4, 4), float(t))  # noqa: E731
        cache.table_for(_key(), 3, build)
        cache.table_for(_key(acceleration_factor=8), 3, build)
        assert len(cache) == 2

    def test_stacks_timesteps_along_dim0_preserving_dtype(self) -> None:
        cache = MaskTableCache()
        table = cache.table_for(_key(), 6, lambda t: torch.zeros((1, 4, 4), dtype=torch.bool))
        assert table.shape == (6, 1, 4, 4)
        assert table.dtype is torch.bool

    def test_row_t_is_the_mask_for_timestep_t(self) -> None:
        """Ordering is the contract: ``index_select`` relies on it."""
        cache = MaskTableCache()
        table = cache.table_for(_key(), 4, lambda t: torch.full((1, 2, 2), float(t)))
        for t in range(4):
            assert torch.equal(table[t], torch.full((1, 2, 2), float(t)))

    def test_clear_drops_entries_and_forces_rebuild(self) -> None:
        cache = MaskTableCache()
        calls: list[int] = []
        build = lambda t: (calls.append(t), torch.zeros((1, 2, 2)))[1]  # noqa: E731
        cache.table_for(_key(), 2, build)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0
        cache.table_for(_key(), 2, build)
        assert calls == [0, 1, 0, 1]
