"""Tests for ``DeterministicSeedHarness`` and ``set_global_seed``.

Targets ``spectramr.shared.utils.deterministic_seed``. The harness derives a
deterministic seed from a configuration dict via SHA-256, applies it
across ``random``, ``numpy``, and ``torch``, and persists the seed for
later verification.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from spectramr.shared.utils.deterministic_seed import (
    DEFAULT_SEED,
    DeterministicSeedHarness,
    create_deterministic_harness,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# set_global_seed — module-level helper
# ---------------------------------------------------------------------------


def test_set_global_seed_makes_random_reproducible() -> None:
    """Same seed → same ``random.random`` output."""
    set_global_seed(123)
    a = random.random()
    set_global_seed(123)
    b = random.random()
    assert a == b


def test_set_global_seed_makes_numpy_reproducible() -> None:
    """Same seed → same ``np.random`` output."""
    set_global_seed(7)
    a = np.random.rand(3)
    set_global_seed(7)
    b = np.random.rand(3)
    assert (a == b).all()


def test_set_global_seed_makes_torch_reproducible() -> None:
    """Same seed → same ``torch.randn`` output."""
    set_global_seed(99)
    a = torch.randn(4)
    set_global_seed(99)
    b = torch.randn(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_config_is_deterministic_across_key_order() -> None:
    """Reordering keys does not change the hash."""
    harness = DeterministicSeedHarness()
    cfg_a = {"a": 1, "b": 2, "c": 3}
    cfg_b = {"c": 3, "a": 1, "b": 2}
    assert harness.hash_config(cfg_a) == harness.hash_config(cfg_b)


def test_hash_config_changes_when_value_changes() -> None:
    """Any value change → different hash."""
    harness = DeterministicSeedHarness()
    h_a = harness.hash_config({"a": 1})
    h_b = harness.hash_config({"a": 2})
    assert h_a != h_b


def test_hash_config_recurses_into_nested_dicts() -> None:
    """Nested dict reordering also stable."""
    harness = DeterministicSeedHarness()
    cfg_a = {"outer": {"x": 1, "y": 2}}
    cfg_b = {"outer": {"y": 2, "x": 1}}
    assert harness.hash_config(cfg_a) == harness.hash_config(cfg_b)


# ---------------------------------------------------------------------------
# Config → seed
# ---------------------------------------------------------------------------


def test_config_to_seed_returns_int() -> None:
    """Seed is a non-negative 31-bit integer."""
    harness = DeterministicSeedHarness()
    seed = harness.config_to_seed({"learning_rate": 1e-3})
    assert isinstance(seed, int)
    assert 0 <= seed < (2**31 - 1)


def test_config_to_seed_deterministic() -> None:
    """Same config → same seed across instances."""
    h1 = DeterministicSeedHarness()
    h2 = DeterministicSeedHarness()
    cfg = {"foo": [1, 2, 3]}
    assert h1.config_to_seed(cfg) == h2.config_to_seed(cfg)


# ---------------------------------------------------------------------------
# apply_deterministic_seed
# ---------------------------------------------------------------------------


def test_apply_deterministic_seed_no_config_uses_base_seed() -> None:
    """When config is None, the base seed is applied directly."""
    harness = DeterministicSeedHarness(base_seed=17)
    assert harness.apply_deterministic_seed(None) == 17
    assert harness.get_current_seed() == 17


def test_apply_deterministic_seed_with_config_returns_derived_seed() -> None:
    """Config-derived seed is stored and returned."""
    harness = DeterministicSeedHarness()
    cfg = {"epochs": 100, "lr": 0.001}
    seed = harness.apply_deterministic_seed(cfg)
    assert seed == harness.config_to_seed(cfg)
    assert harness.get_current_config_hash() == harness.hash_config(cfg)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_seed_info_round_trip(tmp_path: Path) -> None:
    """``save_seed_info → load_seed_info`` recovers the seed and hash."""
    harness = DeterministicSeedHarness(base_seed=42)
    cfg = {"model": "unet"}
    harness.apply_deterministic_seed(cfg)

    out_file = tmp_path / "seed.json"
    harness.save_seed_info(str(out_file))

    new_harness = DeterministicSeedHarness()
    info = new_harness.load_seed_info(str(out_file))
    assert info["seed"] == harness.get_current_seed()
    assert info["config_hash"] == harness.get_current_config_hash()


def test_save_seed_info_creates_parent_dir(tmp_path: Path) -> None:
    """Missing parent directories are created."""
    harness = DeterministicSeedHarness()
    harness.apply_deterministic_seed(None)
    nested = tmp_path / "a" / "b" / "c" / "seed.json"
    harness.save_seed_info(str(nested))
    assert nested.exists()


# ---------------------------------------------------------------------------
# Reproducibility verification
# ---------------------------------------------------------------------------


def test_verify_reproducibility_returns_true_after_apply() -> None:
    """Just-applied config verifies as reproducible."""
    harness = DeterministicSeedHarness()
    cfg = {"a": 1}
    harness.apply_deterministic_seed(cfg)
    assert harness.verify_reproducibility(cfg) is True


def test_verify_reproducibility_false_for_unrelated_config() -> None:
    """Different config → False."""
    harness = DeterministicSeedHarness()
    harness.apply_deterministic_seed({"a": 1})
    assert harness.verify_reproducibility({"a": 2}) is False


def test_verify_reproducibility_warns_when_no_seed_applied() -> None:
    """No seed applied → False with warning."""
    harness = DeterministicSeedHarness()
    assert harness.verify_reproducibility({"a": 1}) is False


# ---------------------------------------------------------------------------
# Factory + defaults
# ---------------------------------------------------------------------------


def test_create_deterministic_harness_factory() -> None:
    """Factory returns a ``DeterministicSeedHarness`` with the given base seed."""
    h = create_deterministic_harness(base_seed=99)
    assert isinstance(h, DeterministicSeedHarness)
    assert h.base_seed == 99


def test_default_seed_is_documented_constant() -> None:
    """``DEFAULT_SEED`` is the documented default (42)."""
    assert DEFAULT_SEED == 42
