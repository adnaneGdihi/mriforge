"""Unit tests for the SPECTRA backend adapter and executor.

Exercises the export -> lower -> dispatch contract in isolation with a mocked
runtime (no real cosim / RTL). Mirrors the framework's "mock the runtime"
test-layering discipline from the implementation plan Workstream C.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.schemas.spectra import BackendAccelerationConfigSchema
from spectramr.infrastructure.accelerator.spectra import (
    SpectraBackendAdapter,
    SpectraExecutor,
    SpectraLoweringError,
    get_spectra_executor,
)
from spectramr.infrastructure.accelerator.spectra.spectra_backend import (
    LOWERABLE_OPS,
    ROCC_INSTRUCTION_CONTRACT,
    SPECTRA_BYOC_COMPILER_PATH,
)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Conv2d(1, 1, 3, padding=1)


def _example_inputs() -> list[torch.Tensor]:
    return [torch.randn(1, 1, 8, 8)]


def test_export_graph_captures_traceable_model() -> None:
    """A simple conv model traces without error."""
    adapter = SpectraBackendAdapter(BackendAccelerationConfigSchema())
    graph = adapter.export_graph(_tiny_model(), _example_inputs())
    assert graph is not None


def test_export_graph_raises_typed_error_on_untraceable() -> None:
    """A model whose forward cannot be traced raises SpectraLoweringError."""

    class Untraceable(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("complex / data-consistency layer not traceable")

    adapter = SpectraBackendAdapter(BackendAccelerationConfigSchema())
    with pytest.raises(SpectraLoweringError):
        adapter.export_graph(Untraceable(), _example_inputs())


def test_lower_emits_contract_level_plan() -> None:
    """lower() returns a plan carrying the mesh / precision / RoCC contract."""
    cfg = BackendAccelerationConfigSchema(mesh_rows=8, mesh_cols=8, pe_mode_policy="bf16")
    adapter = SpectraBackendAdapter(cfg)
    graph = adapter.export_graph(_tiny_model(), _example_inputs())
    plan = adapter.lower(graph)

    assert plan["backend"] == "spectra"
    assert plan["mesh"] == (8, 8)
    assert plan["precision"] == "bf16"
    assert plan["rocc_contract"] == ROCC_INSTRUCTION_CONTRACT
    assert plan["compiler"] == SPECTRA_BYOC_COMPILER_PATH
    # The line above is self-referential -- it passes for ANY value, including
    # the absolute developer home path this constant used to hold. Pin the
    # property that actually matters instead: the contract placeholder must not
    # name a machine-specific location a reader could mistake for a real target.
    assert not SPECTRA_BYOC_COMPILER_PATH.startswith("/"), (
        "SPECTRA_BYOC_COMPILER_PATH is a documented placeholder, not a path; "
        "an absolute value here is a hardcoded developer location"
    )
    assert "<" in SPECTRA_BYOC_COMPILER_PATH, (
        "the placeholder must be visibly unresolvable, e.g. '<SPECTRA_REPO>/...'"
    )
    assert set(plan["lowerable_ops"]) <= LOWERABLE_OPS
    # Honest: no real codegen until TVM lands.
    assert plan["lowered"] is False


def test_lower_rejects_none_graph() -> None:
    adapter = SpectraBackendAdapter(BackendAccelerationConfigSchema())
    with pytest.raises(SpectraLoweringError):
        adapter.lower(None)


def test_dispatch_uses_injected_runtime() -> None:
    """A mocked runtime receives the plan + inputs and its result is returned."""
    seen: dict[str, object] = {}

    def runtime_fn(plan: dict, inputs: object) -> torch.Tensor:
        seen["plan"] = plan
        seen["inputs"] = inputs
        return torch.ones(1, 1, 8, 8)

    cfg = BackendAccelerationConfigSchema()
    adapter = SpectraBackendAdapter(cfg, runtime_fn=runtime_fn)
    out = adapter.compile_and_run(_tiny_model(), _example_inputs())

    assert out.shape == (1, 1, 8, 8)
    assert seen["plan"]["backend"] == "spectra"  # type: ignore[index]


def test_dispatch_unwired_raises_not_implemented() -> None:
    """Without a runtime_fn the real cosim/rocc path is honestly unimplemented."""
    adapter = SpectraBackendAdapter(BackendAccelerationConfigSchema())
    graph = adapter.export_graph(_tiny_model(), _example_inputs())
    plan = adapter.lower(graph)
    with pytest.raises(NotImplementedError):
        adapter.dispatch(plan, _example_inputs())


def test_executor_run_full_pipeline_with_mock() -> None:
    """SpectraExecutor.run drives capture -> lower -> dispatch end to end."""
    captured: dict[str, object] = {}

    def runtime_fn(plan: dict, inputs: object) -> torch.Tensor:
        captured["plan"] = plan
        return torch.zeros(1, 1, 8, 8)

    cfg = BackendAccelerationConfigSchema()
    executor = SpectraExecutor(cfg, runtime_fn=runtime_fn)
    out = executor.run(_tiny_model(), _example_inputs())

    assert out.shape == (1, 1, 8, 8)
    assert captured["plan"]["mesh"] == (4, 4)  # type: ignore[index]


def test_get_spectra_executor_selection() -> None:
    """The selection function returns an executor only for the spectra backend."""
    # No block -> None (caller falls through to default backend).
    assert get_spectra_executor(types.SimpleNamespace(backend_acceleration=None)) is None

    # Block present but a different backend -> None.
    other = BackendAccelerationConfigSchema(backend="nvdla")
    assert get_spectra_executor(types.SimpleNamespace(backend_acceleration=other)) is None

    # Spectra block -> a real executor.
    cfg = BackendAccelerationConfigSchema()
    executor = get_spectra_executor(types.SimpleNamespace(backend_acceleration=cfg))
    assert isinstance(executor, SpectraExecutor)


def test_get_spectra_executor_forwards_runtime_fn() -> None:
    """The selection function forwards runtime_fn so dispatch is mockable."""

    def runtime_fn(plan: dict, inputs: object) -> torch.Tensor:
        return torch.full((1, 1, 8, 8), 7.0)

    cfg = BackendAccelerationConfigSchema()
    executor = get_spectra_executor(
        types.SimpleNamespace(backend_acceleration=cfg), runtime_fn=runtime_fn
    )
    assert executor is not None
    out = executor.run(_tiny_model(), _example_inputs())
    assert torch.allclose(out, torch.full((1, 1, 8, 8), 7.0))
