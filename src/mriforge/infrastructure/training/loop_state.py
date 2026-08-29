"""The mutable per-run loop state owned by the training loop (WS-3 PR-3).

:class:`~mriforge.infrastructure.training.builders.environment.TrainingEnvironment`
is ``frozen=True``, so its ``step`` field can never be advanced — every
``self.env.step`` read therefore returns the constant ``0``. That makes any
step-gated logic reading it an *inert* read (pitfall #16): the diffusion
curriculum diagnostic, the anomaly-log cadence, and the early-step target-leak
warning were all evaluating at a frozen step 0.

``LoopState`` is the mutable counterpart the training loop owns and updates each
iteration, exposed to the strategy through the ``strategy.loop_state`` seam. It
gives a single source of truth for the live iteration/epoch without threading
those values through every method signature (the threading is exactly what the
``current_step = self.env.step`` overwrite silently broke).

Layering note: this lives in ``infrastructure/training/`` (not the
``pipelines/`` loop module) so that ``BaseTrainingStrategy`` — itself in
``infrastructure/`` — can import it without a forbidden leftward dependency.
The ``pipelines`` loop imports it rightward, which is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopState:
    """Live, mutable loop counters exposed to the strategy.

    The training loop owns one instance per strategy (assigned to
    ``strategy.loop_state``) and writes :attr:`iteration` / :attr:`epoch`
    before each step. Strategies read them (e.g. the diffusion curriculum
    schedule in ``_compute_and_log_debug_info``) instead of the frozen,
    perpetually-zero ``env.step``.

    Only the counters that a strategy actually consumes live here — best-metric
    / checkpoint / sanity-trace bookkeeping remains loop-local until a consumer
    needs it, to avoid advertising unwired state.

    Attributes:
        iteration: The 1-based global training iteration currently executing
            (0 before the loop starts, matching the historical default).
        epoch: ``iteration // len(train_loader)`` for the current step.
        loss_weight_overrides: Dynamic ``{loss_term: weight}`` overrides written
            by :class:`~mriforge.infrastructure.training.loss_schedule_controller.LossScheduleController`
            each step and read by the loss computer (via the strategy seam) to
            override a term's static config weight. Empty by default (no
            scheduling => existing static behavior).
    """

    iteration: int = 0
    epoch: int = 0
    loss_weight_overrides: dict[str, float] = field(default_factory=dict)


def resolve_loop_iteration(strategy: object) -> int:
    """Return the live global training iteration from a strategy's seam.

    The SSOT replacement for the frozen, perpetually-zero ``self.env.step``
    (pitfall #16): the training loop advances ``strategy.loop_state.iteration``
    each step, so step-gated logic must read it here. Falls back to ``0`` when
    ``loop_state`` is absent — a mixin constructed standalone in a unit test, or
    direct/scripting construction — mirroring the historical
    ``self.env.step if self.env else 0`` defensiveness so call sites stay safe
    regardless of how ``strategy`` was built.

    Using one helper (rather than a base-class property) keeps the mixins
    — ``AdversarialMixin`` / k-space debug paths that are unit-tested with a
    bare stub ``self`` that does not inherit the strategy base — from depending
    on the MRO for the seam.
    """
    loop_state = getattr(strategy, "loop_state", None)
    return loop_state.iteration if loop_state is not None else 0
