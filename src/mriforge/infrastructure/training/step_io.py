"""Canonical training-step I/O dataclasses + a non-breaking compat adapter.

Cached-cascade WS-D. The recon census found ``train_step`` / ``validation_step``
signatures drift across 40+ strategies (the GAN family adds
``iteration``/``batch_data``/``scaler``; the calibration family swaps to
``(batch, epoch=0, step=0)``; ``disentangled`` drops ``input_batch``/``target_batch``;
``pma_varnet`` uses Lightning-style ``(val_batch, batch_idx)``). This module
freezes the canonical container so component workstreams (WS-6/WS-7, Phase B)
can migrate behind the :func:`accepts_step_io` adapter without a big-bang.

Design notes
------------
* ``TrainingStepInput`` is the *union* of every parameter the surveyed step
  signatures consume, so any strategy's needs project onto a subset of its
  fields.
* :func:`accepts_step_io` lets a method accept EITHER a ``TrainingStepInput``
  positional OR the legacy explicit kwargs. It is **silent by default**: the
  repo promotes ``DeprecationWarning`` from ``mriforge.*`` to a test error
  (``filterwarnings``), so warning on the legacy path would break every
  not-yet-migrated caller. The deprecation escalation (``warn=True``) is a
  *flagged follow-up* for when the legacy path is actually removed -- not this
  pass.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TrainingStepInput",
    "TrainingStepOutput",
    "accepts_step_io",
]

# The canonical parameter names a step method may consume (the union over the
# surveyed signatures). ``accepts_step_io`` explodes a TrainingStepInput onto
# exactly the subset a given method declares.
_STEP_FIELDS: tuple[str, ...] = (
    "batch",
    "epoch",
    "input_batch",
    "target_batch",
    "iteration",
    "step",
    "scaler",
    "batch_data",
)


@dataclass(frozen=True)
class TrainingStepInput:
    """Canonical, frozen input to a training/validation step.

    A strategy migrated to the canonical signature takes a single
    ``TrainingStepInput``; legacy strategies keep their explicit kwargs and are
    bridged by :func:`accepts_step_io`.
    """

    batch: Any = None
    epoch: int = 0
    input_batch: Any = None
    target_batch: Any = None
    iteration: int = 0
    step: int = 0
    scaler: Any = None
    batch_data: Any = None
    #: overflow for paradigm-specific extras (kept out of the positional union)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def as_kwargs(self, accepted: set[str] | frozenset[str]) -> dict[str, Any]:
        """Project this input onto the parameter names a method ``accepted``.

        Only canonical fields present in ``accepted`` are returned; ``extras``
        keys in ``accepted`` are merged in last.
        """
        out: dict[str, Any] = {
            name: getattr(self, name) for name in _STEP_FIELDS if name in accepted
        }
        for key, value in self.extras.items():
            if key in accepted:
                out[key] = value
        return out


@dataclass(frozen=True)
class TrainingStepOutput:
    """Canonical, frozen output of a training/validation step.

    Accommodates both shapes the codebase uses today: the training path returns
    a list of per-substep loss/metric dicts (``records``); the validation path
    returns a flat ``dict[str, float]`` (``metrics``).
    """

    records: tuple[dict[str, Any], ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    total_loss: Any = None

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> TrainingStepOutput:
        return cls(records=tuple(records))

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, float]) -> TrainingStepOutput:
        return cls(metrics=dict(metrics))

    def to_records(self) -> list[dict[str, Any]]:
        """Back-compat view: the training-loop expects ``list[dict]``."""
        return list(self.records)


def accepts_step_io(
    func: Callable[..., Any] | None = None, *, warn: bool = False
) -> Callable[..., Any]:
    """Decorator: accept either a ``TrainingStepInput`` or legacy kwargs.

    Usage::

        @accepts_step_io
        def train_step(self, batch, epoch, input_batch=None, target_batch=None, **kw):
            ...

    Called the new way (``strategy.train_step(TrainingStepInput(...))``) the
    decorator explodes the dataclass onto the method's declared parameters.
    Called the legacy way (explicit kwargs) it passes through unchanged -- and is
    SILENT (see module docstring) unless ``warn=True`` is set for a future
    removal pass.
    """

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        accepted = frozenset(sig.parameters) - {"self"}
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], TrainingStepInput):
                sio = args[0]
                # If the method accepts **kwargs, give it the full union;
                # otherwise project onto the declared names only.
                allowed = set(_STEP_FIELDS) if has_var_kw else set(accepted)
                exploded = sio.as_kwargs(allowed)
                exploded.update(kwargs)  # explicit kwargs win over the dataclass
                return fn(self, **exploded)
            if warn:
                warnings.warn(
                    f"{fn.__qualname__} called with legacy kwargs; pass a "
                    "TrainingStepInput instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return fn(self, *args, **kwargs)

        # marker for the signature-drift fitness check (setattr: function objects
        # carry arbitrary attrs; direct assignment trips the type-checker)
        setattr(wrapper, "__accepts_step_io__", True)  # noqa: B010
        return wrapper

    if func is not None:  # bare @accepts_step_io
        return _decorate(func)
    return _decorate  # @accepts_step_io(warn=True)
