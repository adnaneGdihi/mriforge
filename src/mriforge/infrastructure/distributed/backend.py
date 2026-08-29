"""Process-group backend resolution.

``config.parallel.backend`` existed, was documented as the backend selector, and
was **read by nothing**. ``setup_distributed`` took its value from the
``train-distributed --backend`` CLI flag instead -- and because that flag carried
``default="nccl"``, "the user asked for nccl" and "argparse filled it in" were
indistinguishable. There was no way for the YAML value to win, which is why it
never did (#621 F3).

Precedence, highest first:

1. ``--backend`` **when the user actually typed it** (the flag's default is now
   ``None``, which is what makes this distinguishable at all).
2. ``config.parallel.backend`` — the SSOT.
3. The schema default.

Returns the source alongside the value so provenance can record *why* a run used
what it used, rather than only what.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["resolve_distributed_backend"]

#: Backends that require CUDA. NCCL on a CPU device is not a slow-but-working
#: configuration -- it fails at collective time, well after startup.
_CUDA_ONLY = frozenset({"nccl"})


def resolve_distributed_backend(
    config: Any,
    cli_backend: str | None = None,
    *,
    device: Any = None,
) -> tuple[str, str]:
    """Resolve the process-group backend.

    Args:
        config: ``TrainingSettings``.
        cli_backend: ``--backend`` as typed, or ``None`` when not passed.
        device: Resolved device, checked for the nccl/CPU contradiction.

    Returns:
        ``(backend, source)`` with ``source`` in ``{"cli", "config", "default"}``.
    """
    parallel = getattr(config, "parallel", None)

    if cli_backend:
        backend, source = cli_backend, "cli"
    elif parallel is not None and "backend" in getattr(parallel, "model_fields_set", ()):
        backend, source = parallel.backend, "config"
    elif parallel is not None:
        backend, source = parallel.backend, "default"
    else:
        backend, source = "nccl", "default"

    device_type = getattr(device, "type", None) or (device if isinstance(device, str) else None)
    if backend in _CUDA_ONLY and device_type is not None and "cuda" not in str(device_type):
        raise ValueError(
            f"parallel.backend={backend!r} (from {source}) requires CUDA, but the "
            f"resolved device is {device_type!r}. Use backend: 'gloo' for CPU "
            "collectives. NCCL on CPU does not degrade -- it fails at the first "
            "collective, long after the run reports a successful start."
        )

    logger.info("[Parallel] process-group backend=%s (source=%s)", backend, source)
    return backend, source
