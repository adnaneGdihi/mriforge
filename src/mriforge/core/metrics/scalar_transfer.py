"""One device->host sync for a batch of scalar tensors.

Three sites had independently grown the same fused-transfer idiom
(``torch.stack(...).cpu().tolist()``) because the naive spelling -- ``.item()``
per entry -- costs one GPU sync *each*. A sync is not merely slow: it drains the
queue, so N syncs in a loop also destroy the overlap between one metric's kernels
and the next one's launch.

This module owns the mechanism and nothing else. **Reduction policy stays with
the caller**, deliberately: ``mean()`` over a non-scalar is the right answer for a
per-sample loss and a *defect* for a metric that is supposed to return a scalar.
Folding both into one helper is how two different questions end up sharing one
answer (pitfall #13b), so :func:`fuse_to_host` refuses anything but scalars and
lets each caller reduce first.

Lives in ``core/`` because ``core.metrics.computer`` is its main consumer and
``core`` may not import from ``infrastructure`` (non-negotiable #5) -- the
pre-existing copy of this idiom sits in a strategy mixin, which the computer
cannot reach.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["fuse_to_host"]


def fuse_to_host(scalars: Sequence[torch.Tensor]) -> list[float]:
    """Transfer already-scalar tensors to the host in a SINGLE sync.

    Args:
        scalars: Single-element, real-valued tensors, in the order the caller
            wants them back.

    Returns:
        One Python float per input, in input order. Empty in, empty out.

    Raises:
        ValueError: If any entry is not a single-element tensor, or is complex.
            Both are caller-policy questions (how to reduce, which part to take)
            and guessing here would silently answer them.

    Note:
        Values are widened to ``float64`` before stacking so a mixed fp16/fp32
        batch under AMP has a uniform dtype. Widening is lossless, so this is
        not a precision change -- it is the reason a naive ``torch.stack`` of
        autocast outputs raises.
    """
    if not scalars:
        return []

    prepared: list[torch.Tensor] = []
    for i, t in enumerate(scalars):
        if not isinstance(t, torch.Tensor):
            raise ValueError(
                f"fuse_to_host[{i}]: expected a torch.Tensor, got "
                f"{type(t).__name__}. Convert non-tensors before calling."
            )
        if t.numel() != 1:
            raise ValueError(
                f"fuse_to_host[{i}]: expected a single-element tensor, got "
                f"shape {tuple(t.shape)}. Reduce it first -- mean() is correct "
                "for a loss and wrong for a metric, so this helper will not pick."
            )
        if torch.is_complex(t):
            raise ValueError(
                f"fuse_to_host[{i}]: complex tensor. Take .real or .abs() first; "
                "which one is meaningful depends on the quantity."
            )
        prepared.append(t.detach().reshape(()).double())

    try:
        # THE point of this module: one sync for the whole batch.
        return [float(v) for v in torch.stack(prepared).cpu().tolist()]
    except RuntimeError:
        # Entries on different devices defeat the stack. Correctness outranks
        # the fusion, so fall back to per-tensor transfers rather than raising.
        return [float(t.cpu().item()) for t in prepared]
