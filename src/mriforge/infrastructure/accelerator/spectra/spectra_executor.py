"""SPECTRA execution provider: graph capture -> BYOC lowering -> dispatch.

:class:`SpectraExecutor` is the *execution provider* the inference layer selects
when a config opts into the SPECTRA hardware backend. It is intentionally NOT a
training strategy and is NOT registered in the training ``STRATEGY_CLASS_PATHS``
dispatcher — inference-on-SPECTRA is an execution backend, per the
implementation plan Workstream C "New execution hook".

Selection seam
--------------
:func:`get_spectra_executor` is the one clean selection function. The inference
factory (``mriforge.infrastructure.inference.inference_factory``) can engage the
SPECTRA backend with a single guard at the top of ``create``::

    from mriforge.infrastructure.accelerator.spectra import get_spectra_executor
    spectra = get_spectra_executor(config)
    if spectra is not None:
        return spectra            # backend selected -> hardware execution provider

That one-line hook is deliberately left to the caller rather than rewriting the
factory's model-type-substring dispatch, to keep the integration light and
reversible (the plan's "prefer the lightest correct integration" guidance).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from mriforge.infrastructure.accelerator.spectra.spectra_backend import (
    SpectraBackendAdapter,
)

if TYPE_CHECKING:
    import torch

    from mriforge.config.schemas.spectra import BackendAccelerationConfigSchema

logger = logging.getLogger(__name__)


class SpectraExecutor:
    """Run inference for a torch model on the SPECTRA backend.

    Thin orchestrator over :class:`SpectraBackendAdapter`: it owns the
    capture -> lowering -> dispatch flow and is the object the inference layer
    holds. State is minimal (the validated config + a lazily built adapter) so
    it composes cleanly with the existing inference seam.
    """

    def __init__(
        self,
        config: BackendAccelerationConfigSchema,
        runtime_fn: Callable[[dict[str, Any], Sequence[Any]], Any] | None = None,
    ) -> None:
        self.config = config
        self.adapter = SpectraBackendAdapter(config, runtime_fn=runtime_fn)

    def run(
        self,
        model: torch.nn.Module,
        example_inputs: Sequence[torch.Tensor],
    ) -> Any:
        """Execute ``model`` on ``example_inputs`` via the SPECTRA backend.

        Returns the runtime's result tensors. With no injected ``runtime_fn``
        the dispatch raises ``NotImplementedError`` (the real cosim/rocc paths
        are not wired here) — capture and lowering still run, so an export
        failure is surfaced as :class:`SpectraLoweringError` before dispatch.
        """
        logger.info(
            "SpectraExecutor.run: backend=%s runtime=%s mesh=%dx%d",
            self.config.backend,
            self.config.runtime.value,
            self.config.mesh_rows,
            self.config.mesh_cols,
        )
        graph = self.adapter.export_graph(model, example_inputs)
        plan = self.adapter.lower(graph)
        return self.adapter.dispatch(plan, example_inputs)


def get_spectra_executor(
    config: Any,
    runtime_fn: Callable[[dict[str, Any], Sequence[Any]], Any] | None = None,
) -> SpectraExecutor | None:
    """Return a :class:`SpectraExecutor` iff the config selects the SPECTRA backend.

    The single, clean selection function for the inference seam. Returns
    ``None`` when no SPECTRA backend is configured (so the caller falls through
    to its default backend); returns an executor when
    ``config.backend_acceleration.backend == "spectra"``.

    Args:
        config: A ``TrainingSettings`` (or any object exposing
            ``backend_acceleration``).
        runtime_fn: Optional dispatch callable forwarded to the executor (test /
            future-wiring seam).
    """
    block = getattr(config, "backend_acceleration", None)
    if block is None:
        return None
    if getattr(block, "backend", None) != "spectra":
        return None
    return SpectraExecutor(block, runtime_fn=runtime_fn)


__all__ = ["SpectraExecutor", "get_spectra_executor"]
