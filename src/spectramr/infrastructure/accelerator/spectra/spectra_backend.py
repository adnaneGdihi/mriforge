"""SPECTRA backend adapter: torch graph -> BYOC lowering -> runtime dispatch.

This is the framework-facing seam onto the SPECTRA chip. Given a ``torch.nn``
model and a :class:`BackendAccelerationConfigSchema`, it:

1. **Exports the graph.** Captures the model as a traceable graph via
   ``torch.jit.trace`` (ONNX export is used when available as a richer IR).
   Complex-valued and data-consistency layers are the known friction point
   (implementation plan Workstream C risk): if tracing raises, the failure is
   surfaced as :class:`SpectraLoweringError` rather than silently swallowed.
2. **Invokes the BYOC compiler (contract-level).** The lowering target is the
   TVM-BYOC backend at ``compiler/spectra_byoc`` in the SPECTRA repo
   (``patterns.py`` / ``partition.py`` / ``codegen_rocc.py``). TVM is **not**
   installed in this environment, so :meth:`lower` documents and validates the
   contract (offloadable op set, the RoCC ``CFG``/``FIRE``/``POLL``/``FLUSH``
   instruction-stream shape) without emitting real code.
3. **Dispatches to the runtime.** ``runtime == cosim`` drives the Verilator MMIO
   bridge; ``runtime == rocc`` targets silicon. Both are wired through a single
   ``runtime_fn`` injection point so unit tests can mock the dispatch.

Nothing here executes on real RTL yet. The honest, testable surface is: the
graph capture is real; the lowering and dispatch are contract-level stubs with
explicit, mockable seams.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from spectramr.config.schemas.spectra import BackendAccelerationConfigSchema

logger = logging.getLogger(__name__)

# Path to the SPECTRA repo's BYOC compiler. Import-by-path / contract-level
# only: TVM is not installed, so this is documented, not loaded. Kept as a
# module constant so the contract is greppable and a future wiring step has one
# place to point at.
#
# The value is a PLACEHOLDER, not a location. It used to be one developer's
# absolute home path, which read as a real, resolvable target on any machine
# that grepped for it. Nothing opens this string -- it is copied verbatim into
# the lowering plan below and never handed to the filesystem -- so a
# placeholder loses no capability and gains an obviously-unresolvable value.
# The wiring step that makes it real owns the knob that supplies it; do not add
# a config knob before there is a reader (non-negotiable 8).
SPECTRA_BYOC_COMPILER_PATH = "<SPECTRA_REPO>/compiler/spectra_byoc"

# Operators the SPECTRA BYOC backend is contracted to lower (implementation
# plan Workstream B3 coverage matrix). Anything outside this set runs on the
# host; the Tier-1 ``check_spectra_*`` guards and (eventually) ``spectra_lowerable``
# flag a gap rather than allowing a silent host fallback.
LOWERABLE_OPS: frozenset[str] = frozenset(
    {
        "conv",
        "gemm",
        "matmul",
        "linear",
        "pixel_shuffle",
        "depth_to_space",
        "fft_shift",
        "transpose",
        "gridding",
        "instance_norm",
        "group_norm",
        "spectral_norm",
        "softmax",
        "complex_conv",
    }
)

# The RoCC opcode contract emitted per offloaded subgraph (byte-for-byte the
# contract proven against the SystemVerilog oracle in the SPECTRA repo).
ROCC_INSTRUCTION_CONTRACT: tuple[str, ...] = ("CFG", "FIRE", "POLL", "FLUSH")

# ---------------------------------------------------------------------------
# SPECTRA unified-top routing contract (chip repo rtl/spectra/spectra_top.sv).
# ---------------------------------------------------------------------------
# The chip exposes a single MMIO control surface (0x0A00-0x0FFF, one block per
# dataflow stage) and a read-only discovery register, SPECTRA_CAPS @0x0FFC:
#
#     [9:0]  feature bitmap  WCSR|CPLX|SPARSE24|SCALE|NORM|FLASH|WINO|DEP|GRID|TILE
#     [19:16] mesh_rows   [23:20] mesh_cols   [27:24] lanes   [31:28] version
#
# Each ``backend_acceleration`` config field maps to the register block that
# realizes it. mesh dims are build-time fixed and *advertised* via SPECTRA_CAPS
# (the config is validated against the read-back caps, not written); the rest
# drive their stage's control register. Precision policy spans Path-A PE_MODE
# (0x0300) and the spectra SCALE/CPLX control blocks. The
# ``test_spectra_caps_routing`` unit test asserts every feature field is routed.
SPECTRA_CAPS_ADDR: int = 0x0FFC
SPECTRA_REGISTER_MAP: dict[str, str | list[str]] = {
    "mesh_rows": "SPECTRA_CAPS",  # advertised; config validated against caps
    "mesh_cols": "SPECTRA_CAPS",
    "lanes": "SPECTRA_CAPS",
    "pe_mode_policy": ["PE_MODE", "SCALE_CTRL", "CPLX_CTRL"],
    "enable_nmae": "GRID_CTRL",
    "gridding_kernel_width": "GRID_CTRL",  # KW sizing validated vs synthesized engine
    "oversampling": "GRID_CTRL",  # sigma sizing
    "per_channel_scaling": "SCALE_CTRL",
}


class SpectraLoweringError(RuntimeError):
    """Raised when a model cannot be exported/lowered for the SPECTRA backend.

    The common cause is the known Workstream-C risk: ``torch`` cannot trace a
    complex-valued or data-consistency layer into an exportable graph.
    """


class SpectraBackendAdapter:
    """Exports a torch model and dispatches it to the SPECTRA chip.

    Args:
        config: The validated SPECTRA backend block.
        runtime_fn: Optional dispatch callable ``(plan, inputs) -> outputs``.
            Defaults to a contract-level stub that raises
            :class:`NotImplementedError`; tests inject a mock here.
    """

    def __init__(
        self,
        config: BackendAccelerationConfigSchema,
        runtime_fn: Callable[[dict[str, Any], Sequence[Any]], Any] | None = None,
    ) -> None:
        self.config = config
        self._runtime_fn = runtime_fn

    # ---- 1. graph capture -------------------------------------------------

    def export_graph(
        self,
        model: torch.nn.Module,
        example_inputs: Sequence[torch.Tensor],
    ) -> Any:
        """Capture ``model`` as a traceable graph.

        Uses ``torch.jit.trace`` (the lowest-common-denominator capture that is
        always available with torch). ONNX is preferred when installed, but is
        not a hard dependency here.

        Raises:
            SpectraLoweringError: if tracing fails (the complex/DC-layer risk).
        """
        import torch

        model.eval()
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, tuple(example_inputs), strict=False)
        except Exception as exc:
            raise SpectraLoweringError(
                "torch.jit.trace failed to capture the model graph for the "
                "SPECTRA backend. This is the known Workstream-C risk: "
                "complex-valued / data-consistency layers may not be traceable. "
                f"Underlying error: {exc!r}"
            ) from exc
        return traced

    # ---- 2. BYOC lowering (contract-level) --------------------------------

    def lower(self, graph: Any) -> dict[str, Any]:
        """Lower a captured graph to a SPECTRA dispatch plan (contract-level).

        TVM is not installed, so this does not emit real RoCC code. It returns a
        *plan* dict describing what the ``compiler/spectra_byoc`` codegen would
        produce: the offload target, the mesh geometry, the precision policy,
        the RoCC instruction contract, and the lowerable-op set. This is the
        stable contract the executor and the audit checks reason about.
        """
        if graph is None:
            raise SpectraLoweringError("Cannot lower a None graph.")

        cfg = self.config
        plan: dict[str, Any] = {
            "backend": cfg.backend,
            "compiler": SPECTRA_BYOC_COMPILER_PATH,
            "mesh": (cfg.mesh_rows, cfg.mesh_cols),
            "lanes": cfg.lanes,
            "precision": cfg.pe_mode_policy.value,
            "enable_nmae": cfg.enable_nmae,
            "gridding": {
                "kernel_width": cfg.gridding_kernel_width,
                "oversampling": cfg.oversampling,
            },
            "per_channel_scaling": cfg.per_channel_scaling,
            "rocc_contract": ROCC_INSTRUCTION_CONTRACT,
            "lowerable_ops": tuple(sorted(LOWERABLE_OPS)),
            "lowered": False,  # honest: no real codegen until TVM lands
        }
        logger.info(
            "SPECTRA BYOC lowering is contract-level (TVM not installed); "
            "emitted plan targets mesh=%s precision=%s runtime=%s.",
            plan["mesh"],
            plan["precision"],
            cfg.runtime.value,
        )
        return plan

    # ---- 3. runtime dispatch ----------------------------------------------

    def dispatch(self, plan: dict[str, Any], inputs: Sequence[Any]) -> Any:
        """Dispatch a lowered plan to the configured runtime.

        Uses the injected ``runtime_fn`` when present (the test / future-wiring
        seam). Otherwise raises :class:`NotImplementedError` for the real
        cosim/rocc paths, which are not yet wired in this environment.
        """
        if self._runtime_fn is not None:
            return self._runtime_fn(plan, inputs)

        runtime = self.config.runtime.value
        raise NotImplementedError(
            f"SPECTRA '{runtime}' runtime dispatch is not wired in this "
            "environment. The cosim path needs the Verilator MMIO TCP bridge "
            "(set backend_acceleration.cosim_bridge_addr); the rocc path needs "
            "a Chipyard-elaborated RISC-V target. Inject runtime_fn to mock."
        )

    # ---- convenience: full pipeline ---------------------------------------

    def compile_and_run(
        self,
        model: torch.nn.Module,
        example_inputs: Sequence[torch.Tensor],
    ) -> Any:
        """Run the full export -> lower -> dispatch pipeline."""
        graph = self.export_graph(model, example_inputs)
        plan = self.lower(graph)
        return self.dispatch(plan, example_inputs)


__all__ = [
    "LOWERABLE_OPS",
    "ROCC_INSTRUCTION_CONTRACT",
    "SPECTRA_BYOC_COMPILER_PATH",
    "SpectraBackendAdapter",
    "SpectraLoweringError",
]
