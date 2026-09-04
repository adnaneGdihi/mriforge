"""Timestep <-> acceleration round-trip check for the cascade.

Split out of :mod:`cascading_validation` in the Wave 0 exit-criterion work
(#1400). Answers *does this arm's severity schedule survive a round trip*, which
is a separate question from which levels it declares (:mod:`cascade_levels`) and
from how a result row is assembled (the facade).

Re-exported from ``cascading_validation``, so all 24 importers are unaffected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple

#: A realised acceleration is "fully sampled" at or below this factor, so a
#: rung there measures an identity mask -- an infinite-PSNR reading that would
#: pollute best-metric selection if it were published as a severity point.
IDENTITY_ACCELERATION: Final[float] = 1.0


def training_band(num_timesteps: int) -> tuple[int, int]:
    """The ``[T/10, 9T/10]`` band the cascade used to clamp every level into.

    Kept for the two places it is still correct, and **only** those:

    * the linear-fallback inverse (:func:`legacy_linear_timestep`), which is
      unbounded above ``max_acceleration`` and has no schedule to check itself
      against;
    * the accel-agnostic image/latent-translation pass, where the timestep is
      inert and any legal value does.

    It is NOT applied to the schedule-aware inverse's answer any more (#1295):
    that answer is already contractually in ``[0, T-1]``, and clamping it moved
    the mask off the rung the metric column names.
    """
    return max(1, num_timesteps // 10), num_timesteps - num_timesteps // 10


def legacy_linear_timestep(
    *,
    acceleration: float,
    base_acceleration: float,
    max_acceleration: float,
    num_timesteps: int,
) -> int:
    """The pre-2026-05-28 linear inverse, band-clamped. Fallback path only.

    ``t = T * (R - base) / (max - base)``. It matches the forward schedule only
    when ``acceleration.schedule_type`` is ``linear``; for a ``step`` ladder it
    picks timesteps the schedule decodes as a *different* R, which is the bug
    :meth:`KSpaceAccelerator.timestep_for_acceleration` exists to fix. The
    strategy therefore reaches this only when no mask generator is wired.

    It lives here, rather than inline in ``validation_step``, because it was
    inline: the unit test that covered it re-implemented the formula in its own
    file and asserted against the copy, so it could not fail when production
    changed. That is the same hand-copied-vocabulary shape this module's other
    tests already call out.
    """
    span = max(1.0, float(max_acceleration) - float(base_acceleration))
    t_ideal = num_timesteps * (float(acceleration) - float(base_acceleration)) / span
    min_t, max_t = training_band(num_timesteps)
    return max(min_t, min(max_t, int(t_ideal)))


class RoundTrip(NamedTuple):
    """What the schedule actually decoded a chosen timestep back to.

    Attributes:
        realized: ``forward(timestep)`` -- the acceleration the mask at that
            timestep really has.
        tolerance: the largest residual attributable to the schedule's own
            discretisation rather than to a wrong timestep.
        locally_optimal: whether no neighbouring timestep realises the request
            more closely.
        ok: both conditions hold.
        reason: which condition failed, for the caller's log line; ``None``
            when ``ok``.
    """

    realized: float
    tolerance: float
    locally_optimal: bool
    ok: bool
    reason: str | None


def check_round_trip(
    *,
    requested: float,
    timestep: int,
    forward: Callable[[int], float],
    num_timesteps: int,
) -> RoundTrip:
    """Ask the forward schedule what ``timestep`` decodes to, and judge it.

    The cascade asks an inverse for "the timestep whose mask realises R=X" and
    then labels a whole row of metrics ``acceleration_level=X``. Nothing made
    the inverse prove it, and two different mechanisms silently produced a
    mislabelled row. **They need two different tests, and neither one catches
    the other** -- which is why this returns a conjunction rather than a single
    tolerance comparison.

    *Re-pointing.* The caller post-processed the returned timestep -- the
    ``[T/10, 9T/10]`` band clamp removed in #1295 -- landing on a neighbouring
    rung while the label kept naming the requested one. Caught by
    **local optimality**: if some adjacent timestep realises the request more
    closely, the one in hand was not the inverse's answer. This has to be the
    scale-free half, because a ``step`` ladder's buckets can be arbitrarily far
    apart (the experiment_11 ladder jumps 25.6 -> 29.444 -> 32.0 near the top),
    so any residual-based bound is either too wide there or too tight
    elsewhere.

    *Saturation.* A continuous schedule inverts by clamping ``progress`` into
    ``[0, 1]``, so a request above ``max_acceleration`` returns ``T-1`` and the
    row claims R=64 while measuring R=8. Local optimality **passes** here --
    ``T-1`` genuinely is the closest timestep available -- so only a
    **residual bound** catches it.

    The residual bound is the schedule's own resolution, not a fixed
    percentage. One timestep moves R by ``span / (T - 1)``: 0.007 on a linear
    T=1000 arm, over 1.0 on a T=29 one. A fixed relative tolerance would either
    reject the coarse arm's unavoidable quantisation or wave through real drift
    on the fine one. Half the larger neighbouring step is "no better answer
    exists", on any horizon, and still rejects an 8-vs-64 saturation by four
    orders of magnitude.

    ``forward`` is called at most three times, only at validation time, so this
    costs nothing in the training loop (non-negotiable 9).

    Args:
        requested: the acceleration the cascade level is labelled with.
        timestep: the timestep the inverse returned for it.
        forward: the schedule's ``get_acceleration_factor``.
        num_timesteps: horizon, used only to keep neighbour probes in range.

    Returns:
        A :class:`RoundTrip`. This function neither logs nor raises, so it
        stays pure and testable without an accelerator; the caller decides
        what a failure means.
    """
    requested = float(requested)
    realized = float(forward(timestep))
    residual = abs(realized - requested)

    steps: list[float] = []
    better_neighbour = False
    for probe in (timestep - 1, timestep + 1):
        if not (0 <= probe <= num_timesteps - 1):
            continue
        neighbour = float(forward(probe))
        steps.append(abs(neighbour - realized))
        if abs(neighbour - requested) < residual:
            better_neighbour = True

    # No neighbours means T == 1: there is only one timestep, so the schedule
    # has no resolution to spend and the match must be exact.
    local_step = max(steps) if steps else 0.0
    tolerance = max(local_step / 2.0, 1e-6)

    locally_optimal = not better_neighbour
    within = residual <= tolerance

    reason: str | None = None
    if not locally_optimal:
        reason = (
            "an adjacent timestep realises the request more closely, so this "
            "timestep is not the inverse's answer -- something moved it"
        )
    elif not within:
        reason = (
            "the residual exceeds the schedule's own resolution, so the "
            "request lies outside what this schedule can realise"
        )

    return RoundTrip(
        realized=realized,
        tolerance=tolerance,
        locally_optimal=locally_optimal,
        ok=locally_optimal and within,
        reason=reason,
    )


__all__ = [
    "IDENTITY_ACCELERATION",
    "RoundTrip",
    "check_round_trip",
    "legacy_linear_timestep",
    "training_band",
]
