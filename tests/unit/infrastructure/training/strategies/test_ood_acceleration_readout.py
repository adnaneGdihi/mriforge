"""``ood_acceleration_readout``: the OOD rungs' one owner (VF review 2026-09-03). Planted violations first."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
    OOD_COUNT_KEY,
    ood_acceleration_readout,
    ood_accelerations,
    ood_metric_key,
)


class _Twin:
    """Records the rate every score() call saw and whether it was restored."""

    def __init__(self) -> None:
        self.acceleration = 4.0
        self.enable_undersampling = True
        self.entered: list[float] = []

    @contextmanager
    def at_acceleration(self, rate: float):
        previous = self.acceleration
        self.acceleration = rate
        self.entered.append(rate)
        try:
            yield
        finally:
            self.acceleration = previous


def _config(rng):
    return SimpleNamespace(
        physics=SimpleNamespace(
            digital_twin=SimpleNamespace(ood_acceleration_range=rng, enable_undersampling=True)
        )
    )


def test_two_rungs_give_two_prefixed_sets_and_the_count() -> None:
    twin = _Twin()
    out = ood_acceleration_readout(
        twin, (16.0, 32.0), lambda: {"val_psnr": twin.acceleration, "hfen": 1.0}
    )
    assert out == {
        "val_ood_16x_psnr": 16.0,
        "val_ood_16x_hfen": 1.0,
        "val_ood_32x_psnr": 32.0,
        "val_ood_32x_hfen": 1.0,
        OOD_COUNT_KEY: 2.0,
    }
    assert twin.entered == [16.0, 32.0] and twin.acceleration == 4.0


def test_no_rungs_writes_only_a_zero_count_and_never_scores() -> None:
    calls: list[int] = []

    def score():
        calls.append(1)
        return {"val_psnr": 1.0}

    assert ood_acceleration_readout(_Twin(), (), score) == {OOD_COUNT_KEY: 0.0}
    assert calls == []


def test_the_twin_is_restored_when_a_rung_raises() -> None:
    """Planted violation: a failing rung must not leave the twin at the OOD rate."""
    twin = _Twin()

    def score():
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        ood_acceleration_readout(twin, (16.0,), score)
    assert twin.acceleration == 4.0


def test_metric_key_strips_a_val_prefix_once() -> None:
    assert ood_metric_key(16.0, "val_psnr") == "val_ood_16x_psnr"
    assert ood_metric_key(16.0, "psnr") == "val_ood_16x_psnr"
    assert ood_metric_key(2.5, "val_ssim") == "val_ood_2.5x_ssim"


def test_declared_rungs_are_read_as_floats() -> None:
    assert ood_accelerations(_config([16, 32])) == (16.0, 32.0)


def test_an_absent_range_is_no_rungs() -> None:
    assert ood_accelerations(_config(None)) == ()
    assert ood_accelerations(SimpleNamespace(physics=None)) == ()
    assert ood_accelerations(SimpleNamespace()) == ()


def test_an_absent_twin_block_is_no_rungs() -> None:
    assert ood_accelerations(SimpleNamespace(physics=SimpleNamespace(digital_twin=None))) == ()
