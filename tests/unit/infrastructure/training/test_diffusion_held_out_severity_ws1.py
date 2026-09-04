"""WS1-core-03: held_out_severity_eval — the H4 robustness eval reads the
held-out severity grid from model.model_kwargs.digital_twin_kwargs, gated on the
validation flag, so non-opted-in arms are unaffected."""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _stub(enabled, grid):
    mk = (
        {"digital_twin_kwargs": {"held_out_severity_grid": grid}}
        if grid is not None
        else {}
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            validation=block_stub(
            "validation",
            held_out_severity_eval=enabled),
            model=SimpleNamespace(model_kwargs=mk),
        )
    )


def test_disabled_returns_empty() -> None:
    assert DiffusionTrainingStrategy._held_out_severity_points(_stub(False, [64, 128])) == []


def test_enabled_returns_grid_as_floats() -> None:
    assert DiffusionTrainingStrategy._held_out_severity_points(
        _stub(True, [64, 128])
    ) == [64.0, 128.0]


def test_enabled_but_no_grid_returns_empty() -> None:
    assert DiffusionTrainingStrategy._held_out_severity_points(_stub(True, None)) == []


def test_enabled_empty_grid_returns_empty() -> None:
    assert DiffusionTrainingStrategy._held_out_severity_points(_stub(True, [])) == []
