"""Runtime dimension-contract guard (Layer 2 of the dimension-contract plan).

A single ``assert_input_contract`` installed as a ``register_forward_pre_hook`` on
the generator (one call site, all paradigms — see
``BaseTrainingStrategy._ensure_input_contract_guard``). It raises
``DimensionContractError`` at *iteration 1* when the input tensor violates the
model's *declared* ``ModelCapabilities``, instead of a cryptic conv/matmul error
(or a silent NaN) hours into training.

False-positive firewall — this guard asserts ONLY declared + universally-safe
invariants, so it never blocks a legitimate config:

* **Unannotated model** (``capabilities is None``): the hook is not even installed
  (see the wiring site), so legacy models carry zero risk.
* **Spatial rank** vs declared ``spatial_dims``: a 2D-declared conv net fed a 5D
  tensor is unambiguously broken (the 3D-into-2D crash class).
* **Even channels** vs ``expects_real_imag_interleaved``: interleaved real/imag
  needs an even channel count; odd is unambiguously broken.
* It does **NOT** assert ``channels == in_channels`` — conditioning-concat
  strategies (S-map doubling, etc.) legitimately change the channel count, so
  that assertion would false-positive everywhere.
* Internal reshapes (coil flatten, complex stacking) happen *inside* ``forward``,
  downstream of this input hook, so reshape-heavy models are untouched.
* Partially-declared capabilities → only the declared, non-``None`` fields are
  checked.
"""

from __future__ import annotations

import os
from typing import Any

_VALID_MODES = ("off", "observe", "enforce")


def resolve_contract_mode() -> str:
    """Resolve the runtime guard mode from ``SPECTRAMR_DIMENSION_CONTRACT``.

    * ``off``     — guard not installed.
    * ``observe`` — log a warning on a violation but do NOT raise (rollout
      default; never breaks an in-flight run).
    * ``enforce`` — raise :class:`DimensionContractError` at iteration 1.

    Raises ``ValueError`` on an unknown value (pitfall #15: an advertised knob is
    read AND validated). Defaults to ``observe`` so landing the guard is safe.
    """
    raw = os.environ.get("SPECTRAMR_DIMENSION_CONTRACT", "observe").strip().lower()
    if raw not in _VALID_MODES:
        raise ValueError(
            f"SPECTRAMR_DIMENSION_CONTRACT={raw!r} is invalid; expected one of {_VALID_MODES}."
        )
    return raw


class DimensionContractError(RuntimeError):
    """Raised when a forward input violates the model's declared contract."""

    def __init__(self, *, where: str, got: str, expected: str) -> None:
        self.where = where
        self.got = got
        self.expected = expected
        super().__init__(
            f"[{where}] dimension contract violated: got {got}; expected {expected}. "
            "Either the dataloader is delivering the wrong shape, or the model's "
            "ModelCapabilities declaration is wrong."
        )


def first_tensor(inputs: Any) -> Any | None:
    """Return the first tensor-like positional argument, or None.

    A forward pre-hook receives the positional ``inputs`` tuple. Some paradigms
    pass a non-tensor first (a coords list for an INR, an int timestep) — skip
    those and find the first object that quacks like a tensor.
    """
    seq = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
    for x in seq:
        if hasattr(x, "ndim") and hasattr(x, "shape") and hasattr(x, "dtype"):
            return x
    return None


def assert_input_contract(
    tensor: Any,
    capabilities: Any,
    *,
    where: str,
    data_rank: int | None = None,
) -> None:
    """Assert the input tensor satisfies the model's *declared* contract.

    Args:
        tensor: the first forward input tensor (or None — then a no-op).
        capabilities: the model's ``ModelCapabilities`` (or None — then a no-op).
        where: human label for the error (usually the model_type).
        data_rank: the data's declared spatial rank, advisory (error context only).

    Raises:
        DimensionContractError: on a declared-and-unambiguous violation.
    """
    caps = capabilities
    if caps is None or tensor is None:
        return
    ndim = getattr(tensor, "ndim", None)
    if not isinstance(ndim, int):
        return

    # 1. Spatial rank vs declared spatial_dims. Layout is [B, C, *spatial], so the
    #    spatial rank is ndim - 2. Only checked for tensors that have a channel +
    #    at least one spatial axis (ndim >= 3).
    spatial_dims = getattr(caps, "spatial_dims", None)
    if spatial_dims is not None and ndim >= 3:
        spatial_rank = ndim - 2
        if spatial_rank not in tuple(spatial_dims):
            expected_ndims = tuple(d + 2 for d in spatial_dims)
            raise DimensionContractError(
                where=where,
                got=f"input ndim={ndim} (spatial rank {spatial_rank}"
                + (f", data declares rank {data_rank}" if data_rank is not None else "")
                + ")",
                expected=f"spatial_dims {tuple(spatial_dims)} (input ndim one of {expected_ndims})",
            )

    # 2. Interleaved real/imag requires an even channel count.
    if getattr(caps, "expects_real_imag_interleaved", None) is True and ndim >= 2:
        channels = int(tensor.shape[1])
        if channels % 2 != 0:
            raise DimensionContractError(
                where=where,
                got=f"in_channels={channels} (odd)",
                expected="even channel count (real/imag interleaved pairs)",
            )
