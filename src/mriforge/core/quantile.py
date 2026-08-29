"""``torch.quantile`` that survives tensors above torch's hard element cap.

``torch.quantile`` raises once the reduced dimension exceeds **2**24** elements
(~16.7M). That is not a theoretical bound for this codebase: a single 3-D
multi-coil volume reaches it, and the call sites that hit it are robust-scale
computations on whole volumes — the ones that decide what intensity the model
sees.

Hoisted here rather than copied a third time. The implementation already
existed in ``infrastructure/physics/digital_twin_simulator.py``; the data layer
had five raw ``torch.quantile`` calls with no guard at all, and
``core/metrics/tissue_segmentation.py`` carried a second, stricter variant.
``core/`` is the shared home every layer may import inward from — the data layer
must not reach into ``infrastructure/`` (non-negotiable #5).

Two separate concerns live here.

**Above the cap** — decimation is a deterministic even stride, never
``randperm``: the quantiles these callers need are statistically unchanged under
uniform subsampling, and a random draw here would perturb
``initialize_accelerator`` seeding and break run-to-run reproducibility.

**Below the cap** — an exact selection fast path. ``torch.quantile`` sorts the
whole tensor (``O(n log n)``) to read at most two order statistics; two
``torch.kthvalue`` selections (``O(n)``) read the same two, and interpolating
between them reproduces torch's answer **bit for bit** — but only when the rank
``q * (n - 1)`` is evaluated in the *tensor's own* dtype, which is what torch
does internally. Evaluating that rank in Python ``float`` (float64) is what
makes the two disagree.

That distinction matters for reading issue #1537, which reported the symptom
(divergence appearing above ~2**18 elements) and attributed it to ``kthvalue``
being inexact at scale. Measured, that attribution is wrong: with the rank in
the tensor dtype the route is bit-exact at 262,144 and 1,048,576 — the sizes the
issue flagged as unsafe — and a control using a full ``sort`` under the *same*
float64 rank arithmetic diverges identically. The selection method was never the
cause; the rank dtype was. Size was a proxy for "the two roundings happen to
differ here".

The fast path is **guarded, not approximate**. Every precondition it cannot
honour bit-exactly delegates to ``torch.quantile`` instead of returning a near
answer (non-negotiable 3) — see :func:`_fast_select_quantile` for the four that
bite and why.
"""

from __future__ import annotations

import torch

#: torch.quantile's hard cap on the reduced dimension.
QUANTILE_MAX_ELEMS = 1 << 24

#: Element count below which the selection fast path is not worth its guards.
#:
#: The guards are two ``O(n)`` reductions against an ``O(n log n)`` sort, so the
#: crossover is a constant-factor question, not a complexity one, and only
#: measurement settles it. Measured (CPU, 1 thread, gradient-magnitude-shaped
#: data, ``q=0.9``, min-of-9 trials over 3 seeds): the selection route *loses*
#: below this size — 0.78x at n=1024, 0.82x at n=2048 — and wins from 4096 up
#: (1.44x at 4096, 1.58x at 16,384, 2.40x at 102,400). 4096 is the first size
#: where it wins on both the sparse gradient-magnitude and uniform
#: distributions, so it is the crossover rather than a round number near it.
#:
#: Below the threshold the declined call costs nothing measurable: 0.96-1.02x
#: against this function's previous body, i.e. inside noise.
_FAST_SELECT_MIN_ELEMS = 1 << 12

#: Dtypes the fast path is verified bit-exact on. Anything else delegates, so a
#: dtype ``kthvalue`` accepts but ``torch.quantile`` rejects (e.g. float16)
#: keeps raising through the one owner rather than being silently widened.
_FAST_SELECT_DTYPES = frozenset({torch.float32, torch.float64})

__all__ = ["QUANTILE_MAX_ELEMS", "robust_quantile"]


def _fast_select_quantile(flat: torch.Tensor, q: float) -> torch.Tensor | None:
    """Exact ``q``-quantile of 1-D *flat* via two ``kthvalue`` selections.

    Returns ``None`` when any precondition for bit-identity does not hold; the
    caller then delegates to ``torch.quantile``. ``None`` never means "an
    approximation was returned" — this helper is exact or absent, because a
    close-enough number here is exactly the silent-fallback shape
    non-negotiable 3 forbids.

    The five preconditions that actually bite, all measured rather than assumed:

    * **NaN.** ``kthvalue`` does not propagate it. On ``rand(1000)`` with one
      element set to NaN, ``torch.quantile(x, 0.9)`` is NaN while the selection
      route returns ``0.9038815498352051`` — a plausible, wrong number.
    * **±inf.** Interpolating between two infinities is NaN, where sorting is
      not. Folded into the same ``isfinite`` scan as NaN.
    * **Negative zero.** ``sort`` and ``kthvalue`` order tied ``-0.0``/``+0.0``
      differently, so one can return ``-0.0`` where the other returns ``0.0``.
      Numerically equal; not bit-identical, and bit-identity is the contract.
    * **Non-CPU / other dtypes.** Verified on CPU float32/float64 only. On CUDA
      the ``int(pos.floor())`` below is also a host sync.
    * **Above ``QUANTILE_MAX_ELEMS``.** ``kthvalue`` carries no such cap, so it
      answers where ``torch.quantile`` raises ``quantile() input tensor is too
      large``. Measured at ``n = 2**24 + 1``: torch raises, the selection route
      returns ``0.9000321626663208``. Unreachable with the default ``max_elems``
      (decimation guarantees ``n <= max_elems`` before the call) but reachable
      through the public signature as ``robust_quantile(x, q, max_elems=2**30)``.
      Above the cap there is no reference value to be identical *to*, so the
      fast path declines and the canonical raise stands.
    """
    n = flat.numel()
    if (
        n < _FAST_SELECT_MIN_ELEMS
        or n > QUANTILE_MAX_ELEMS
        or flat.device.type != "cpu"
        or flat.dtype not in _FAST_SELECT_DTYPES
        or not 0.0 <= q <= 1.0
    ):
        return None
    if not bool(torch.isfinite(flat).all()):
        return None
    if bool(((flat == 0) & torch.signbit(flat)).any()):
        return None

    # The rank MUST be computed in the tensor's dtype -- this single line is
    # what the whole bit-identity claim rests on (see the module docstring).
    pos = torch.tensor(q, dtype=flat.dtype) * (n - 1)
    lo_i = int(pos.floor())
    hi_i = int(pos.ceil())
    w = pos - lo_i
    lo = torch.kthvalue(flat, lo_i + 1).values
    if hi_i == lo_i:
        return torch.lerp(lo, lo, w)
    hi = torch.kthvalue(flat, hi_i + 1).values
    return torch.lerp(lo, hi, w)


def robust_quantile(
    x: torch.Tensor,
    q: float,
    dim: int | None = None,
    *,
    max_elems: int = QUANTILE_MAX_ELEMS,
) -> torch.Tensor:
    """``torch.quantile``, decimated above *max_elems* and selected below it.

    Cannot change an existing number. Above the cap that is because decimation
    is the only alternative to torch raising; below it, because the selection
    fast path is bit-identical where it engages and delegates everywhere else.

    ``dim`` is deliberately excluded from the fast path: only one production
    call site passes it, and per-slice NaN semantics would make this module a
    second owner of logic ``torch.quantile`` already owns (non-negotiable 17).

    Args:
        x: Input tensor.
        q: Quantile in ``[0, 1]``.
        dim: Reduction dimension; ``None`` flattens.
        max_elems: Decimation threshold. Defaults to torch's own cap.

    Returns:
        The quantile, same shape semantics as ``torch.quantile``.
    """
    if dim is None:
        flat = x.reshape(-1)
        n = flat.numel()
        if n > max_elems:
            step = (n + max_elems - 1) // max_elems
            flat = flat[::step]
        fast = _fast_select_quantile(flat, q)
        if fast is not None:
            return fast
        return torch.quantile(flat, q)

    n = x.shape[dim]
    if n > max_elems:
        step = (n + max_elems - 1) // max_elems
        idx = torch.arange(0, n, step, device=x.device)
        x = x.index_select(dim, idx)
    return torch.quantile(x, q, dim=dim)
