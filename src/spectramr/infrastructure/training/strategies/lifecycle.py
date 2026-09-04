"""Driver for the four :class:`BaseTrainingStrategy` lifecycle hooks.

``on_epoch_start`` / ``on_epoch_end`` / ``on_validation_start`` /
``on_validation_end`` are declared on
:class:`~spectramr.infrastructure.training.strategies.base.BaseTrainingStrategy`
and overridden by real strategies, but until this module landed **no code under
``src/spectramr`` ever called them** (audit dossier D12 §3.1, issue #1353). Two
schema-declared YAML features live only inside those overrides and were
therefore inert on every arm that declared them:

* ``training.pipeline.end_to_end_finetune_epoch`` — unfreezes every stage of a
  multi-stage pipeline once the epoch index reaches the threshold. It is read at
  ``pipeline_strategy.py:109`` and *logged as active* at startup, but its only
  behavioural use is inside ``MultiTrainingStrategy.on_epoch_start``.
* per-stage ``early_stopping`` — freezes a stage whose monitored validation
  metric has not improved for ``patience`` epochs. Populated, summarised in the
  startup log, and evaluated only inside ``MultiTrainingStrategy.on_epoch_end``.

This driver is the single owner of *when* each hook fires (non-negotiable 17):
the training loop hands it the epoch index and the validation boundaries, and it
decides. It never re-derives epoch arithmetic of its own.

Ordering, which is forced by the loop's shape
---------------------------------------------
The loop computes ``epoch = iteration // train_loader_len`` and treats
``iteration % train_loader_len == 0`` as the epoch boundary. At that boundary
iteration the epoch index has **already advanced**, its ``train_step`` is the
first step of the *new* epoch, and the boundary validation — the only source of
fresh end-of-epoch metrics — runs after that step. So within the boundary
iteration ``on_epoch_start(N + 1)`` necessarily precedes ``on_epoch_end(N)``.

Firing ``on_epoch_end`` one iteration earlier would restore the intuitive order
at the cost of handing it metrics measured in the *middle* of the epoch, which
is strictly worse: per-stage early stopping would then score every epoch on a
measurement taken before that epoch finished. The order is pinned by a test so
it reads as a decision rather than an accident.

:meth:`end_epoch` reports the epoch that actually *completed* (``N``), not the
loop's current ``epoch`` (``N + 1``), matching the hook's own docstring ("the
index of the epoch that just finished").

Rank symmetry
-------------
Nothing here is gated on ``is_main_process``. The hooks mutate model state —
``MultiTrainingStrategy`` flips ``requires_grad`` on whole stages — so a rank-0-only
dispatch would desynchronise DDP parameter groups. Both inputs are already
rank-identical: ``epoch`` is pure integer arithmetic on the shared iteration
counter, and ``val_metrics`` is all-reduced before it reaches the loop.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: The lifecycle contract, in the order a run drives it. ``base.py`` is the
#: single declaring owner (D16 §4 step 1); this tuple is what the driver knows
#: how to dispatch, and what the wiring tests enumerate.
LIFECYCLE_HOOKS: tuple[str, ...] = (
    "on_epoch_start",
    "on_epoch_end",
    "on_validation_start",
    "on_validation_end",
)


class StrategyLifecycleDriver:
    """Fire a strategy's lifecycle hooks at the right boundaries, once each.

    The driver is cheap to poll: :meth:`begin_epoch` is called every iteration
    and does an integer comparison before anything else, so the per-step cost is
    one attribute read and one ``!=`` (non-negotiable 9 — no allocation, no
    device sync, no dict copy on the hot path). The metrics dicts are copied
    only at the epoch/validation cadence, where a hook mutating the loop's own
    ``val_metrics`` would be a real hazard.

    A hook a strategy does not implement is **reported once and skipped**, never
    silently inferred to be intentional (non-negotiable 18). A hook that raises
    propagates: an exception inside ``on_epoch_end`` means early stopping did not
    evaluate, and a run that continues past that is a run reporting a guarantee
    it no longer has.
    """

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy
        self._started_epoch: int | None = None
        self._pending_epoch_end: int | None = None
        self._driven: set[str] = set()
        self._missing: set[str] = set()

    @property
    def driven(self) -> frozenset[str]:
        """Hook names this driver has actually dispatched at least once.

        This is the observation non-negotiable 16 asks for: "registered" and
        "declared" are not delivery, "fired" is. Tests assert on it and the
        first fire of each hook is also logged at INFO.
        """
        return frozenset(self._driven)

    def _dispatch(self, name: str, *args: Any) -> bool:
        hook = getattr(self._strategy, name, None)
        if not callable(hook):
            if name not in self._missing:
                self._missing.add(name)
                logger.warning(
                    "Strategy %s does not implement lifecycle hook '%s'; it will "
                    "not be driven for this run.",
                    type(self._strategy).__name__,
                    name,
                )
            return False
        if name in self._driven:
            logger.debug("Lifecycle hook '%s' fired.", name)
        else:
            self._driven.add(name)
            logger.info(
                "Lifecycle hook '%s' driven on %s (first fire).",
                name,
                type(self._strategy).__name__,
            )
        hook(*args)
        return True

    def begin_epoch(self, epoch: int) -> bool:
        """Fire ``on_epoch_start`` when the epoch index advances.

        Returns ``True`` on the iterations that open a new epoch. Idempotent
        within an epoch, so the loop can call it unconditionally every step.

        Resume is handled by construction: the first call after a restart at
        ``start_iteration`` opens whatever epoch that iteration lands in, with no
        spurious ``on_epoch_end`` for an epoch this process never ran.
        """
        if self._started_epoch == epoch:
            return False
        if self._started_epoch is not None:
            self._pending_epoch_end = self._started_epoch
        self._started_epoch = epoch
        self._dispatch("on_epoch_start", epoch)
        return True

    def end_epoch(self, metrics: Mapping[str, Any] | None = None) -> int | None:
        """Fire ``on_epoch_end`` for the epoch that completed, if any.

        Returns the completed epoch index, or ``None`` when no epoch has
        finished yet — which is exactly the case at the very first iteration,
        itself an epoch boundary under ``iteration % train_loader_len == 0``.
        Self-guarding: the pending index is cleared on fire, so a second call in
        the same epoch is a no-op rather than a double-count against a patience
        counter.
        """
        completed = self._pending_epoch_end
        if completed is None:
            return None
        self._pending_epoch_end = None
        self._dispatch("on_epoch_end", completed, dict(metrics or {}))
        return completed

    def begin_validation(self) -> None:
        """Fire ``on_validation_start`` immediately before a validation pass."""
        self._dispatch("on_validation_start")

    def end_validation(self, metrics: Mapping[str, Any] | None = None) -> None:
        """Fire ``on_validation_end`` with the aggregated validation metrics."""
        self._dispatch("on_validation_end", dict(metrics or {}))
