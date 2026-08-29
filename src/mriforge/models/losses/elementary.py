"""Registry-routed resolver for elementary per-step reconstruction losses.

Cached-cascade WS-6 (plan finding #6): several diffusion ``p_losses``
methods and ``vq_reconstruction_loss`` each carried a private ``if/elif``
chain mapping ``loss_type`` strings onto ``torch.nn.functional`` calls.
This module is the single dispatch surface that replaces those chains:
names resolve through :class:`~mriforge.models.losses.registry.LossRegistry`
(the loss SSOT), and instances are constructed once and shared — the
routed losses are stateless (no parameters, no buffers, no train/eval
behaviour), so per-step callers never pay module construction.

Numerical equivalence with the replaced functionals:

- ``l1``  → :class:`~mriforge.models.losses.standard_losses.L1Loss`
  (``nn.L1Loss`` subclass) ≡ ``F.l1_loss``.
- ``l2``  → :class:`~mriforge.models.losses.standard_losses.MSELoss`
  (``nn.MSELoss`` subclass) ≡ ``F.mse_loss``.
- ``huber`` → :class:`~mriforge.models.losses.diffusion_losses.HuberLoss`,
  i.e. ``F.huber_loss(delta=1.0)`` ≡ ``F.smooth_l1_loss(beta=1.0)`` exactly.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from torch import nn

from mriforge.models.losses.registry import create_loss

# ``loss_type`` string as exposed by the p_losses call surfaces → registry name.
_ELEMENTARY_REGISTRY_NAMES: dict[str, str] = {
    "l1": "l1",
    "l2": "l2",
    "huber": "huber",
}


@cache
def shared_loss(name: str, **kwargs: Any) -> nn.Module:
    """Return a process-wide shared instance of a registered loss.

    Only safe for stateless losses (no parameters/buffers, no train/eval
    behaviour) — which holds for every name routed through this module.
    ``kwargs`` values must be hashable (they form the cache key).
    """
    return create_loss(name, **kwargs)


def resolve_elementary_loss(loss_type: str) -> nn.Module:
    """Resolve an elementary reconstruction ``loss_type`` via the registry.

    Args:
        loss_type: One of ``"l1"``, ``"l2"``, ``"huber"``.

    Returns:
        The shared, cached loss module (call as ``loss(prediction, target)``).

    Raises:
        ValueError: On a ``loss_type`` outside the supported set — never
            degrades to a default (pitfall #9).
    """
    try:
        registry_name = _ELEMENTARY_REGISTRY_NAMES[loss_type]
    except KeyError:
        raise ValueError(
            f"Unsupported loss type: {loss_type!r}. Supported: {sorted(_ELEMENTARY_REGISTRY_NAMES)}"
        ) from None
    return shared_loss(registry_name)


__all__ = ["resolve_elementary_loss", "shared_loss"]
