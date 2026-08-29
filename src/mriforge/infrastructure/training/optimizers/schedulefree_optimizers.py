"""Schedule-free optimizers (Defazio et al., 2024).

Same guard contract as the 8-bit wrappers: registered unconditionally, raising at
construction when the extra is absent, so a missing dependency never presents as
an unknown-name typo and never silently degrades to a different optimizer.

**These have a train/eval mode, and getting it wrong evaluates the wrong iterate.**
Schedule-free methods maintain an averaged sequence separate from the iterate the
gradient is taken at; ``optimizer.eval()`` swaps in the averaged weights and
``optimizer.train()`` swaps them back. Validating without that call measures the
wrong point in weight space — which looks like a merely-worse arm, not a bug. The
training loop calls both at the validation boundary via a ``hasattr(opt, "train")``
hook rather than leaving it to the user.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.training.optimizer_registry import register_optimizer

__all__ = ["ScheduleFreeAdamW", "ScheduleFreeSGD", "supports_schedule_free_modes"]

try:  # pragma: no cover - environment dependent
    import schedulefree as _sf
except Exception as _exc:
    _sf = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None


def _install_hint(name: str) -> str:
    detail = f" (import failed: {_IMPORT_ERROR!r})" if _IMPORT_ERROR else ""
    return (
        f"optimizer_type '{name}' requires the [schedulefree] extra. Install with:\n"
        '    pip install -e ".[schedulefree]"\n'
        "Refusing to fall back to a scheduled optimizer: the whole claim of a "
        "schedule-free arm is that it needs no LR schedule, so substituting one "
        f"would answer a different question.{detail}"
    )


def supports_schedule_free_modes(optimizer: object) -> bool:
    """True when ``optimizer`` needs ``train()``/``eval()`` around validation.

    Duck-typed rather than ``isinstance``: the check has to work when the extra
    is absent (nothing to compare against) and when a user supplies their own
    schedule-free implementation.

    ``step`` is part of the predicate because ``train``/``eval`` alone do not
    identify an optimizer -- every ``nn.Module`` has both. Without it, a model
    that ever reached the ``pipeline.optimizers`` mapping would be toggled as
    though it were a schedule-free optimizer.

    This is the SSOT for the question. ``training_loop._set_optimizer_eval_mode``
    used to inline its own variant of it, and the two had already drifted: the
    inline one tested only ONE of the two methods (whichever direction it was
    toggling), so an object exposing ``train`` but not ``eval`` was switched
    into train mode and never back out.
    """
    return (
        callable(getattr(optimizer, "train", None))
        and callable(getattr(optimizer, "eval", None))
        and callable(getattr(optimizer, "step", None))
    )


_ADAMW_PARAMS = frozenset(
    {"lr", "betas", "eps", "weight_decay", "warmup_steps", "r", "weight_lr_power"}
)
_SGD_PARAMS = frozenset({"lr", "momentum", "weight_decay", "warmup_steps", "r", "weight_lr_power"})


@register_optimizer("schedulefree_adamw", aliases=["schedule_free_adamw"], params=_ADAMW_PARAMS)
class ScheduleFreeAdamW(
    _sf.AdamWScheduleFree if _sf is not None else torch.optim.AdamW  # type: ignore[misc]
):
    """``schedulefree.AdamWScheduleFree``; raises without the extra."""

    def __init__(self, params, **kwargs) -> None:
        if _sf is None:
            raise ImportError(_install_hint("schedulefree_adamw"))
        super().__init__(params, **kwargs)


@register_optimizer("schedulefree_sgd", aliases=["schedule_free_sgd"], params=_SGD_PARAMS)
class ScheduleFreeSGD(
    _sf.SGDScheduleFree if _sf is not None else torch.optim.SGD  # type: ignore[misc]
):
    """``schedulefree.SGDScheduleFree``; raises without the extra."""

    def __init__(self, params, **kwargs) -> None:
        if _sf is None:
            raise ImportError(_install_hint("schedulefree_sgd"))
        super().__init__(params, **kwargs)
