"""Cascade severity levels: normalisation, resolution, and skip reconciliation.

Split out of :mod:`cascading_validation` in the Wave 0 exit-criterion work
(#1400): that module was 464 LOC against the 300 ceiling (NN20). Everything here
answers *which severity points does this arm evaluate at, and which did it
actually run* -- independent of the round-trip check and of row assembly, which
now live in :mod:`cascade_round_trip` and in the facade respectively.

Re-exported from ``cascading_validation``, so all 24 importers are unaffected.
"""

from __future__ import annotations

import math
from collections.abc import Container, Mapping, Sequence
from typing import Any, Final

#: In-distribution severity points for the cascade. The strategy evaluates at
#: exactly these, plus any opt-in held-out points, which are flagged by the
#: ``heldout`` column rather than by a separate naming convention.
CASCADING_LEVELS: Final[tuple[int, ...]] = (2, 8, 32)

#: Identity columns written before the metric columns. `iteration`/`epoch` place
#: the row in the run; `acceleration_level`/`heldout`/`timestep` are what used
#: to be encoded in the column name. `acceleration_realized` is what the
#: schedule DECODED that timestep back to (#1295) -- see `build_cascade_row`.
CASCADE_ID_COLUMNS: Final[tuple[str, ...]] = (
    "iteration",
    "epoch",
    "acceleration_level",
    "heldout",
    "timestep",
    "acceleration_realized",
)

#: Reason recorded for a rung that produced no columns and no explanation. Not a
#: diagnosis -- it is the admission that one is missing, which is the honest
#: thing to print and the thing a reader can act on.
UNRECORDED_SKIP_REASON: Final[str] = "skipped-without-recorded-reason"


def _cascade_attr(obj: Any, name: str) -> Any:
    """``obj.name`` / ``obj[name]``, or ``None`` when either the holder or the
    attribute is absent. Duck-typed so this module needs no import from
    ``config/`` -- see :func:`resolve_cascade_levels`."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def normalize_cascade_levels(levels: Any) -> tuple[int | float, ...]:
    """Validate a declared cascade ladder; return it deduplicated and ascending.

    **One owner for the whole rule** (non-negotiable 17). The Pydantic field
    validator on ``validation.cascade.levels`` calls this, and so does
    :func:`resolve_cascade_levels` for a ladder that arrives programmatically.
    Two spellings of "is this ladder legal" would drift, and a drifted
    *validator* fails the way this module's docstring already warns about: not
    with an error, but with a plausible ladder that nobody asked for.

    Integral values come back as ``int``, which is load-bearing rather than
    cosmetic. The flat metric names are built as ``f"_{accel}x"``, so ``2``
    renders ``val_psnr_2x`` while ``2.0`` renders ``val_psnr_2.0x`` -- renaming
    every column that the L4 input-dependence gate and ``_stamp_accel_psnr_gap``
    look up **by name**. Those consumers do not raise on a miss; they would
    quietly stop finding their inputs.

    Ordering is ascending because the accel-gap readout subtracts the last rung
    from the first and reports it as a *degradation*. "First rung is the
    mildest" is an invariant of the consumers, not a property of what a user
    happens to type.

    Raises:
        ValueError: on an empty ladder, a non-numeric or non-finite entry, or
            an acceleration below 1x (R<1 would be *more* than fully sampled).
            Each is refused rather than dropped, because a ladder that silently
            loses a rung still produces a complete-looking run with fewer
            severity points than the YAML asked for (non-negotiable 3).
    """
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise ValueError(
            f"validation.cascade.levels must be a sequence of accelerations, "
            f"got {type(levels).__name__}."
        )
    if not levels:
        raise ValueError(
            "validation.cascade.levels is empty. An empty ladder is refused "
            "rather than treated as 'use the default' or 'skip the cascade': "
            "both are real intentions and they need different spellings. Omit "
            "the key entirely for the default ladder "
            f"{list(CASCADING_LEVELS)}, or set "
            "validation.schedule.* to stop validating."
        )
    cleaned: list[int | float] = []
    for raw in levels:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"validation.cascade.levels entry {raw!r} is not a number "
                f"(type {type(raw).__name__})."
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"validation.cascade.levels entry {raw!r} is not finite.")
        if value < 1.0:
            raise ValueError(
                f"validation.cascade.levels entry {raw!r} is below 1x. An "
                "acceleration under 1 would sample more k-space than the fully "
                "sampled acquisition holds; R=1 is the mildest legal rung."
            )
        cleaned.append(int(value) if value.is_integer() else value)
    return tuple(sorted(set(cleaned)))


def resolve_cascade_levels(validation_config: Any) -> tuple[int | float, ...]:
    """The ladder this run evaluates: ``validation.cascade.levels``, else the default.

    **The single reader of :data:`CASCADING_LEVELS`.** Before this existed the
    ladder was hard-coded in two places -- the strategy and the
    ``validation_cascade_levels_in_range`` health check, whose own docstring
    conceded the two "should be updated in lockstep". That is the shape
    non-negotiable 17 forbids: the checker would go on green-lighting an
    ``acceleration_range`` against a ladder the strategy no longer ran, and the
    symptom is a missing column, not a failure.

    Duck-typed on purpose. It walks the path with ``getattr``/``Mapping``
    access instead of importing ``ValidationConfigSchema``, so ``core/`` takes
    no dependency on ``config/`` and a caller can hand it a ``SimpleNamespace``.
    ``None`` anywhere along ``validation -> cascade -> levels`` means *not
    declared* and yields :data:`CASCADING_LEVELS`, which is why an arm that
    says nothing evaluates exactly the ladder it evaluated before.
    """
    levels = _cascade_attr(_cascade_attr(validation_config, "cascade"), "levels")
    if levels is None:
        return tuple(CASCADING_LEVELS)
    return normalize_cascade_levels(levels)


def reconcile_skipped_levels(
    levels: Sequence[int],
    evaluated: Container[int],
    skipped: Mapping[int, str],
) -> dict[int, str]:
    """``skipped`` plus an entry for every rung that is in neither collection.

    The cascade's completeness VERDICT is safe without this: the strategy also
    compares ``len(evaluated)`` against the ladder, so a rung lost by a
    ``continue`` that forgot to record itself still reads as incomplete. What is
    not safe is the REASON. Recording it is a convention held by each exit from
    the loop, and a convention covers only the exits that existed when it was
    written -- #1295 adds two more, neither of which records. The warning would
    then print ``skipped {}`` next to a short evaluated list: a contradiction on
    the one line an operator reads to find out which rung was lost and why.

    Deriving the difference instead makes the reason complete by construction,
    so a future ``continue`` degrades the diagnosis to
    ``skipped-without-recorded-reason`` rather than to silence.

    ``levels`` must be the in-distribution ladder only. Held-out severity points
    are a robustness readout rather than part of the cascade contract, and
    passing them here would make an absent one look like a lost rung.
    """
    reconciled = dict(skipped)
    for level in levels:
        if level not in evaluated and level not in reconciled:
            reconciled[level] = UNRECORDED_SKIP_REASON
    return reconciled


__all__ = [
    "CASCADE_ID_COLUMNS",
    "CASCADING_LEVELS",
    "UNRECORDED_SKIP_REASON",
    "normalize_cascade_levels",
    "reconcile_skipped_levels",
    "resolve_cascade_levels",
]
