"""8-bit optimizers from ``bitsandbytes`` (Dettmers et al.).

**Registered unconditionally, construction raises when the extra is absent.**

Skipping registration when the dependency is missing would be the tempting
shortcut, and it is wrong: the name would vanish from ``list_available()``, and
the user would get ``Unknown optimizer: 'adamw8bit'`` — a message that
misdiagnoses a missing extra as a typo, and sends them to fix their YAML instead
of their environment.

Falling back to fp32 AdamW would be worse still. The 8-bit optimizer state *is*
the arm's memory claim; substituting a different optimizer would leave the run
reporting a memory and convergence profile that belongs to something else
(CLAUDE.md non-negotiable #3).

``params`` is declared explicitly on each registration rather than introspected,
because ``_Base`` differs by environment — the resolver's accepted-kwarg set must
not depend on whether the extra happens to be installed on the machine doing the
resolving.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.optimizer_registry import register_optimizer

__all__ = ["Adam8bit", "AdamW8bit"]

try:  # pragma: no cover - environment dependent
    import bitsandbytes as _bnb
except Exception as _exc:
    # Deliberately broad. A bitsandbytes wheel built for the wrong CUDA arch
    # raises RuntimeError/OSError at import, not ImportError; letting that escape
    # would take down the whole optimizer registry at package-import time and
    # present as "spectramr is broken" rather than "this extra does not fit".
    _bnb = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None


def _install_hint(name: str) -> str:
    detail = f" (import failed: {_IMPORT_ERROR!r})" if _IMPORT_ERROR else ""
    return (
        f"optimizer_type '{name}' requires the [bnb] extra. Install with:\n"
        '    pip install -e ".[bnb]"\n'
        "Refusing to fall back to a 32-bit optimizer: the 8-bit state is the "
        "point of the arm, so substituting one would invalidate its memory and "
        f"convergence numbers.{detail}"
    )


_ADAM_PARAMS = frozenset({"lr", "betas", "eps", "weight_decay", "amsgrad"})


@register_optimizer("adam8bit", aliases=["adam_8bit", "bnb_adam"], params=_ADAM_PARAMS)
class Adam8bit(_bnb.optim.Adam8bit if _bnb is not None else torch.optim.Adam):  # type: ignore[misc]
    """``bitsandbytes.optim.Adam8bit``; raises without the extra."""

    def __init__(self, params, **kwargs) -> None:
        if _bnb is None:
            raise ImportError(_install_hint("adam8bit"))
        super().__init__(params, **kwargs)


@register_optimizer("adamw8bit", aliases=["adamw_8bit", "bnb_adamw"], params=_ADAM_PARAMS)
class AdamW8bit(_bnb.optim.AdamW8bit if _bnb is not None else torch.optim.AdamW):  # type: ignore[misc]
    """``bitsandbytes.optim.AdamW8bit``; raises without the extra."""

    def __init__(self, params, **kwargs) -> None:
        if _bnb is None:
            raise ImportError(_install_hint("adamw8bit"))
        super().__init__(params, **kwargs)
