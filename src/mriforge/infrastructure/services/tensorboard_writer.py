"""The single TensorBoard writer.

There used to be two. ``pipelines/train.py`` built one at a hardcoded
``<run_dir>/tensorboard`` and received every scalar and image the run produced;
``ComprehensiveLoggingService.__init__`` built another from
``logging.tracking.tensorboard_dir`` and never ran, because ``bootstrap.py``
resolves ``ILoggingService`` through ``LoggingServiceFactory.create``, which
returns a *base* ``LoggingService``. So the knob the config documented steered
the writer nobody had, and the writer everybody used ignored the knob (#928).

This module is the one writer. It owns the ``SummaryWriter``, resolves the
directory, and carries the feature surface so each capability has exactly one
call site and can be unit-tested without running a training loop.

Design constraints this file exists to honour
---------------------------------------------

* **Disabled is a first-class state, not an error.** ``enabled`` is False when
  the config says so, when the rank is not zero, or when ``tensorboard`` is not
  installed. Every method is then a cheap no-op, so call sites do not need
  ``if writer:`` around each one.
* **No silent fallback on an unknown backend** (non-negotiable #3). The backend
  is a closed :class:`TrackingService`; the schema refuses anything else at load
  time, so there is no string comparison here to fall off.
* **No GPU sync in the training loop** (non-negotiable #9). ``add_histogram``
  moves every parameter to host memory. :meth:`histograms` is therefore gated on
  ``logging.intervals.histogram`` (default 1000) and returns immediately on a
  step that is not due, *before* touching a tensor.
* **DDP-safe.** A non-zero rank gets a disabled writer. Nothing here is
  collective, so a disabled rank never waits on an enabled one.
"""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mriforge.config.schemas.enums import TrackingService

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from mriforge.config.schemas.logging import LoggingConfigSchema

try:  # pragma: no cover - import shape depends on the environment
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

__all__ = ["TensorBoardWriter", "resolve_event_dir"]


def resolve_event_dir(run_dir: Path | str, configured: str | None) -> Path:
    """Where the event files go.

    Resolved **relative to the run directory**, which is what makes the knob
    safe to wire at all. 21 committed arms declare it and every one is relative
    (``./tensorboard`` x20, ``runs/`` x1); honouring those against the process
    CWD would put every run's events in one shared directory and interleave
    them. ``Path.__truediv__`` discards the left operand when the right is
    absolute, so an absolute declaration still overrides -- one expression
    covers all three cases.
    """
    return Path(run_dir) / (configured or "tensorboard")


class TensorBoardWriter:
    """Owns the run's ``SummaryWriter`` and every TensorBoard feature used."""

    def __init__(
        self,
        run_dir: Path | str,
        logging_config: LoggingConfigSchema,
        *,
        is_rank_zero: bool = True,
        start_iteration: int = 0,
    ) -> None:
        tracking = logging_config.tracking
        self._interval = logging_config.intervals.histogram
        self._requested = (
            tracking.enabled
            and tracking.enable_tensorboard
            and tracking.service is TrackingService.TENSORBOARD
        )
        self.event_dir = resolve_event_dir(run_dir, tracking.tensorboard_dir)
        self._writer: Any = None
        self._hparam_metrics: dict[str, float] = {}

        if not (self._requested and is_rank_zero):
            return
        if SummaryWriter is None:
            # `tensorboard` is in the `viz` EXTRA, not core, while `service`
            # defaults to `tensorboard`. So "missing" splits in two, and only
            # one half is an error:
            #
            #   declared  -> the arm ASKED for tracking and cannot have it. Raise.
            #   defaulted -> the arm never mentioned tracking. Disabling it is
            #                not a silent fallback, because nothing was requested
            #                -- and raising here would reject the library default
            #                on every arm in an install without the extra.
            #
            # `model_fields_set` is what separates them: pydantic records which
            # keys the YAML actually provided, so this cannot be inferred wrong
            # from a value that merely equals the default.
            if tracking.model_fields_set:
                raise RuntimeError(
                    "logging.tracking declares "
                    f"{sorted(tracking.model_fields_set)} but "
                    "torch.utils.tensorboard is unavailable. Install the extra "
                    "(`pip install -e '.[viz]'`) or declare "
                    "`logging.tracking.service: none`. Degrading silently here "
                    "is how a run reports success having recorded nothing "
                    "(pitfall #9)."
                )
            logger.warning(
                "TensorBoard is unavailable (`tensorboard` is in the `viz` "
                "extra) and this arm does not declare `logging.tracking`, so "
                "tracking is off. Declare `logging.tracking.service: none` to "
                "make that a decision rather than an accident."
            )
            return
        self.event_dir.mkdir(parents=True, exist_ok=True)
        # `purge_step` drops events at or after `start_iteration`, which is what
        # a resumed run needs: without it the pre-crash tail stays in the event
        # file and every chart draws a fold back to the resume point.
        self._writer = SummaryWriter(
            log_dir=str(self.event_dir),
            purge_step=start_iteration or None,
        )

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def __bool__(self) -> bool:
        """Falsy when disabled, so the existing `if tb_writer:` guards keep
        their exact DDP meaning: a non-zero rank skips the block entirely rather
        than calling a no-op inside it."""
        return self.enabled

    def scalars(self, values: dict[str, Any], step: int, prefix: str) -> None:
        """One chart per metric, the ordinary per-step path."""
        if not self.enabled:
            return
        for name, value in values.items():
            if isinstance(value, numbers.Real):
                self._writer.add_scalar(f"{prefix}/{name}", value, global_step=step)

    def grouped_scalars(self, tag: str, values: dict[str, Any], step: int) -> None:
        """Several series on ONE axis -- generator vs discriminator, or the loss
        terms whose relative size is the thing you actually read.

        Separate ``add_scalar`` charts cannot answer "is D winning"; this can.
        """
        if not self.enabled:
            return
        numeric = {k: v for k, v in values.items() if isinstance(v, numbers.Real)}
        if numeric:
            self._writer.add_scalars(tag, numeric, global_step=step)

    def images(self, images: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        for tag, tensor in images.items():
            self._writer.add_images(tag, tensor, global_step=step)

    def histograms(self, model: torch.nn.Module, step: int, prefix: str) -> None:
        """Weight and gradient distributions, on the ``intervals.histogram`` cadence.

        The cadence check comes FIRST and returns before any tensor is touched:
        ``add_histogram`` copies each parameter to the host, so an ungated call
        is a GPU sync per parameter per step (non-negotiable #9).

        Reads distributions rather than norms because that is what separates the
        failure modes -- a collapsed layer and a saturated one can share a norm
        but never share a shape (pitfall #20).
        """
        if not self.enabled or step % self._interval != 0:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            safe = name.replace(".", "/")
            self._writer.add_histogram(f"{prefix}/weights/{safe}", param, step)
            if param.grad is not None:
                self._writer.add_histogram(f"{prefix}/grads/{safe}", param.grad, step)

    def text(self, tag: str, body: str, step: int = 0) -> None:
        """Free text, used for the resolved-config dump.

        Provenance travels with the events rather than only in the run
        directory, so a TensorBoard instance pointed at a bare event dir can
        still answer "what produced this curve".
        """
        if not self.enabled:
            return
        self._writer.add_text(tag, body, global_step=step)

    def record_hparam_metric(self, name: str, value: Any) -> None:
        """Remember a metric for the HParams dashboard written at close."""
        if self.enabled and isinstance(value, numbers.Real):
            self._hparam_metrics[name] = float(value)

    def hparams(self, hparams: dict[str, Any]) -> None:
        """Write the HParams dashboard entry for this run.

        This is the feature that pays for the rest. Pitfall #17 (confounded
        ablation) is this repo's second-largest failure class -- 195 findings --
        and the HParams view is what makes "these two arms differ in more than
        the one knob they claim to test" visible at a glance across runs.

        Called once at close, so the recorded metrics are the run's final
        values. TensorBoard requires scalars, so anything else is stringified
        rather than dropped: an unlogged hyper-parameter is exactly the
        confound this is meant to surface.
        """
        if not self.enabled or not hparams:
            return
        clean: dict[str, Any] = {}
        for key, value in hparams.items():
            clean[key] = value if isinstance(value, (bool, str, numbers.Real)) else repr(value)
        self._writer.add_hparams(clean, self._hparam_metrics or {"hparam/noop": 0.0})

    def flush(self) -> None:
        if self.enabled:
            self._writer.flush()

    def close(self) -> None:
        if self.enabled:
            self._writer.close()
            self._writer = None
