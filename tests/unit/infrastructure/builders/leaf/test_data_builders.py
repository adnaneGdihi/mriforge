"""Tests for the leaf data builders (WS-D BuilderContext migration).

Covers the Phase-0 back-compat migration of the ``(config)``-only data
builder -- :class:`DatasetBuilder` -- to the
canonical ``def __init__(self, ctx: BuilderContext)`` convention behind the
:func:`accepts_builder_context` shim. The shim must keep the legacy
``Builder(config)`` call site working while also accepting an explicit
``BuilderContext``; both forms must produce equivalent state.

``DataLoaderBuilder`` is intentionally NOT migrated (its ``__init__`` takes
``(config, dataset)``), so it is not exercised here.
"""

from __future__ import annotations

import warnings

import pytest

from tests.utils.data_config_stub import DataConfigStub

from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.builders.leaf.data_builders import (
    DatasetBuilder,
)


class _StubConfig:
    """Minimal stand-in for ``TrainingSettings``.

    Both migrated ``__init__`` bodies only stash ``config`` on ``self._config``
    (config attributes are not read until ``validate``/``build``), so a plain
    sentinel object is sufficient to exercise both construction shapes without
    pulling in the heavy real schema.
    """


# The builder migrated in this file; the default state assertions below
# only depend on the shared ``(config)``-only __init__ shape.
MIGRATED_BUILDERS = [DatasetBuilder]


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_builder_accepts_legacy_config(builder_cls) -> None:
    """Legacy ``Builder(config)`` call still works via the shim."""
    config = _StubConfig()
    builder = builder_cls(config)  # type: ignore[arg-type]
    assert builder._config is config


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_builder_accepts_builder_context(builder_cls) -> None:
    """Canonical ``Builder(BuilderContext(config=config))`` works."""
    config = _StubConfig()
    builder = builder_cls(BuilderContext(config=config))
    assert builder._config is config


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_builder_both_forms_equivalent(builder_cls) -> None:
    """Both construction shapes thread the same config SSOT through."""
    config = _StubConfig()

    legacy = builder_cls(config)  # type: ignore[arg-type]
    ctx_form = builder_cls(BuilderContext(config=config))

    assert legacy._config is ctx_form._config is config


def test_dataset_builder_state_equivalent() -> None:
    """DatasetBuilder default state matches across both construction shapes."""
    config = _StubConfig()

    legacy = DatasetBuilder(config)  # type: ignore[arg-type]
    ctx_form = DatasetBuilder(BuilderContext(config=config))

    assert legacy._dataset_type == ctx_form._dataset_type
    assert legacy._split == ctx_form._split
    assert legacy._data_root == ctx_form._data_root
    assert legacy._transform == ctx_form._transform
    assert legacy._motion_artifacts == ctx_form._motion_artifacts
    assert legacy._noise_level == ctx_form._noise_level
    assert legacy._kwargs == ctx_form._kwargs


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_builder_legacy_is_silent(builder_cls) -> None:
    """Legacy construction must not emit a DeprecationWarning by default."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder_cls(_StubConfig())  # type: ignore[arg-type]  # must not raise


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_builder_init_carries_marker(builder_cls) -> None:
    """The shim records its marker on the wrapped __init__."""
    assert (
        getattr(builder_cls.__init__, "__accepts_builder_context__", False) is True
    )


# --- DataLoaderBuilder worker RNG seeding (reliability) --------------------


class _CollationStub:
    strategy = "robust"  # → no extra kwargs, no tio.Subject collation needed
    log_strategy_selection = False


# The shared stub, built once. `patch_size` and `batch_size` are flat legacy
# spellings — they folded to `data.sampling.patch_size` and
# `data.loader.batch_size` — so a hand-written class carrying them flat leaves
# the reader without the sub-block ("'_DataStub' object has no attribute
# 'sampling'"). DataConfigStub routes them from RENAMES; `collation` and
# `enable_slab_mode` are not rename records and are set directly, which is
# exactly what the stub does with an unrecognised kwarg.
def _data_stub():
    return DataConfigStub(
        dataset_type="synthetic",
        patch_size=(8, 8, 1),
        enable_slab_mode=False,
        batch_size=2,
        collation=_CollationStub(),
    )


class _LoaderConfigStub:
    data = _data_stub()


def _build_loader(num_workers: int):
    import torch
    from torch.utils.data import TensorDataset

    from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

    ds = TensorDataset(torch.zeros(6, 1, 8, 8))
    return (
        DataLoaderBuilder(_LoaderConfigStub(), dataset=ds)
        .with_batch_size(2)
        .with_num_workers(num_workers)
        .with_pin_memory(False)
        .build()
    )


def test_dataloader_sets_worker_init_fn() -> None:
    """The built DataLoader must carry a worker seeding hook.

    Without ``worker_init_fn`` the NumPy/``random`` RNG is duplicated across
    workers (PyTorch only auto-seeds the *torch* RNG per worker), so any
    numpy-based augmentation produces identical "random" draws in every worker.
    """
    loader = _build_loader(num_workers=2)
    assert loader.worker_init_fn is not None
    assert callable(loader.worker_init_fn)


def test_persistent_workers_and_prefetch_propagate_to_dataloader() -> None:
    """The director wires ``persistent_workers`` / ``prefetch_factor`` into the
    leaf builder (2026-07-02); the built loader must reflect them when workers
    are enabled — previously these knobs were silently ignored on the val +
    site-A train loaders, so workers respawned every epoch."""
    import torch
    from torch.utils.data import TensorDataset

    from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

    ds = TensorDataset(torch.zeros(6, 1, 8, 8))
    loader = (
        DataLoaderBuilder(_LoaderConfigStub(), dataset=ds)
        .with_batch_size(2)
        .with_num_workers(2)
        .with_pin_memory(False)
        .with_persistent_workers(True)
        .with_prefetch_factor(4)
        .build()
    )
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 4


def test_persistent_workers_noop_when_no_workers() -> None:
    """``persistent_workers`` must self-disable at ``num_workers=0`` (torch
    rejects persistent workers without workers)."""
    import torch
    from torch.utils.data import TensorDataset

    from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

    ds = TensorDataset(torch.zeros(6, 1, 8, 8))
    loader = (
        DataLoaderBuilder(_LoaderConfigStub(), dataset=ds)
        .with_batch_size(2)
        .with_num_workers(0)
        .with_pin_memory(False)
        .with_persistent_workers(True)
        .build()
    )
    assert loader.persistent_workers is False


def test_seed_worker_seeds_numpy_and_random_from_torch_seed() -> None:
    """``seed_worker`` derives NumPy + ``random`` seeds from torch's seed.

    Re-seeding torch to the same base and invoking the hook must reproduce the
    same NumPy/``random`` draws — distinct per worker (torch varies the base),
    yet reproducible across runs. Post-WS5 the hook is the canonical
    ``core.worker_seeding.seed_worker`` re-exported by this builder module.
    """
    import random

    import numpy as np
    import torch

    from spectramr.infrastructure.builders.leaf.data_builders import seed_worker

    torch.manual_seed(1234)
    seed_worker(0)
    first = (np.random.rand(), random.random())

    torch.manual_seed(1234)
    seed_worker(0)
    second = (np.random.rand(), random.random())

    assert first == second


class TestDataLoaderBuilderSampler:
    """``with_sampler`` — the seam VolumeBlockedSliceSampler reaches the loader through.

    Without it the mrixfields all_slices path silently falls back to ``shuffle=True``,
    which is the pathological order the sampler exists to replace: an ordering fix that
    never reaches the DataLoader is a facade (pitfall #16).
    """

    def _dataset(self):
        import torch
        from torch.utils.data import TensorDataset

        return TensorDataset(torch.arange(8).float().unsqueeze(1))

    def test_sampler_reaches_the_dataloader(self) -> None:
        from torch.utils.data import SequentialSampler

        from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

        dataset = self._dataset()
        sampler = SequentialSampler(dataset)
        loader = (
            DataLoaderBuilder(_LoaderConfigStub(), dataset=dataset)
            .with_batch_size(2)
            .with_sampler(sampler)
            .build()
        )
        assert loader.sampler is sampler

    def test_setting_a_sampler_clears_shuffle(self) -> None:
        """DataLoader raises when both are set; the builder resolves it at the call site
        rather than letting it surface as a constructor error two layers away."""
        from torch.utils.data import SequentialSampler

        from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

        dataset = self._dataset()
        builder = (
            DataLoaderBuilder(_LoaderConfigStub(), dataset=dataset)
            .with_batch_size(2)
            .with_shuffle(True)
            .with_sampler(SequentialSampler(dataset))
        )
        assert builder._shuffle is False
        builder.build()  # must not raise

    def test_no_sampler_keeps_shuffle_behaviour(self) -> None:
        from torch.utils.data import RandomSampler

        from spectramr.infrastructure.builders.leaf.data_builders import DataLoaderBuilder

        loader = (
            DataLoaderBuilder(_LoaderConfigStub(), dataset=self._dataset())
            .with_batch_size(2)
            .with_shuffle(True)
            .build()
        )
        assert isinstance(loader.sampler, RandomSampler)
