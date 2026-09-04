"""Time-varying classifier-free-guidance (CFG) schedules :math:`w(t)`.

Multi-axis CFG (see :mod:`spectramr.models.blocks.multi_axis_cfg`) combines
per-axis conditional and unconditional noise predictions with constant
per-axis weights :math:`w_i`. The constant-weight choice is a default
that often isn't optimal across the full diffusion trajectory:

* At **high noise levels** (early reverse steps), broad anatomical
  conditioning is more valuable than fine pathological detail.
* At **low noise levels** (late reverse steps), fine conditioning takes
  over while broad guidance starts pushing samples off the data
  manifold.

A *schedule* :math:`w_i(t)` maps the normalised diffusion progress
:math:`t \\in [0, 1]` to a per-axis weight multiplier. Multiplying the
constant :math:`w_i` by :math:`w_i(t)` gives a time-varying effective
guidance scale without changing any other part of the sampler.

Setting every axis to :class:`ConstantCFGSchedule` (or omitting the
schedule entirely) reproduces the original constant-weight behaviour
**bit-exact**, so the extension is non-breaking by construction.

References
----------
- Ho & Salimans, "Classifier-Free Diffusion Guidance," NeurIPS 2021
  Workshop on Deep Generative Models.
- Karras et al., "Analyzing and Improving the Training Dynamics of
  Diffusion Models," CVPR 2024 (\\S 3 — schedule analyses).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class CFGSchedule(Protocol):
    """Protocol for time-varying CFG schedules.

    Implementations map normalised diffusion progress ``t in [0, 1]`` to
    a weight multiplier; ``t=0`` is the highest-noise step (start of
    reverse sampling), ``t=1`` is the cleanest step.
    """

    def __call__(self, t: float) -> float: ...


@dataclass(frozen=True)
class ConstantCFGSchedule:
    """Constant-weight schedule (identity); ``w(t) = value`` for all ``t``.

    Used as the default when no schedule is configured. Setting every
    axis to this with ``value=1.0`` reproduces the original
    constant-weight behaviour of ``cfg_sample_step`` bit-exact.
    """

    value: float = 1.0

    def __call__(self, t: float) -> float:
        return self.value


@dataclass(frozen=True)
class LinearCFGSchedule:
    """Linear interpolation between ``start`` (at ``t=0``) and ``end`` (at ``t=1``).

    The canonical motivation: heavy guidance at high noise levels
    (``start > end``) softening toward low noise levels. The schedule
    saturates outside ``[0, 1]`` to handle small numerical excursions.
    """

    start: float
    end: float

    def __call__(self, t: float) -> float:
        t_clipped = max(0.0, min(1.0, t))
        return self.start + (self.end - self.start) * t_clipped


@dataclass(frozen=True)
class CosineCFGSchedule:
    """Half-cosine ramp from ``start`` (at ``t=0``) to ``end`` (at ``t=1``).

    Smoother transition than :class:`LinearCFGSchedule`; useful when a
    discontinuous slope produces visible banding in the sample
    trajectory.
    """

    start: float
    end: float

    def __call__(self, t: float) -> float:
        t_clipped = max(0.0, min(1.0, t))
        # 0.5 * (1 - cos(pi * t)) goes 0 -> 1 with zero-slope at the endpoints.
        ramp = 0.5 * (1.0 - math.cos(math.pi * t_clipped))
        return self.start + (self.end - self.start) * ramp


@dataclass(frozen=True)
class OscillatingCosineCFGSchedule:
    """Oscillating cosine schedule: ``w(t) = baseline + amplitude * cos(2*pi*t)``.

    A periodic schedule that returns to the baseline at the boundaries
    (``t=0`` and ``t=1``) and peaks in the middle. Useful when guidance
    should dip mid-trajectory (e.g. let the sampler explore the data
    manifold before re-applying conditioning at the cleanup phase).
    """

    amplitude: float
    baseline: float

    def __call__(self, t: float) -> float:
        return self.baseline + self.amplitude * math.cos(2.0 * math.pi * t)


@dataclass(frozen=True)
class StepCFGSchedule:
    """Two-level step schedule: ``high`` while ``t < transition``, else ``low``.

    Discrete switch; useful when the trajectory has a clear regime
    boundary (e.g. switch off broad conditioning once the anatomy is
    locked in).
    """

    high: float
    low: float
    transition: float = 0.5

    def __call__(self, t: float) -> float:
        return self.high if t < self.transition else self.low


@dataclass(frozen=True)
class PerAxisSchedule:
    """Per-axis schedule bundle for the multi-axis CFG sampler.

    ``schedules`` maps axis name -> :class:`CFGSchedule`. Axes absent
    from the map default to :class:`ConstantCFGSchedule` with value 1.0
    (i.e. no time-varying modulation).
    """

    schedules: Mapping[str, CFGSchedule] = field(default_factory=dict)

    def weight_at(self, axis: str, t: float) -> float:
        sched = self.schedules.get(axis)
        if sched is None:
            return 1.0
        return float(sched(t))


def make_schedule(spec: Mapping[str, object]) -> CFGSchedule:
    """Build a single-axis schedule from a YAML-style mapping.

    Dispatches on the ``type`` key and rejects unknown shapes loudly
    (CLAUDE.md pitfall #9 — no silent fallback to a "default" schedule
    when the user asked for something specific).
    """
    if "type" not in spec:
        raise ValueError(
            f"CFG schedule mapping must declare a ``type`` key (got keys: {sorted(spec.keys())})"
        )
    kind = str(spec["type"])
    if kind == "constant":
        return ConstantCFGSchedule(value=float(spec.get("value", 1.0)))
    if kind == "linear":
        return LinearCFGSchedule(start=float(spec["start"]), end=float(spec["end"]))
    if kind == "cosine_ramp":
        return CosineCFGSchedule(start=float(spec["start"]), end=float(spec["end"]))
    if kind == "cosine":
        # YAML uses the oscillating form (amplitude + baseline). The
        # ramp form is named ``cosine_ramp`` to keep the YAML surface
        # unambiguous.
        return OscillatingCosineCFGSchedule(
            amplitude=float(spec["amplitude"]),
            baseline=float(spec["baseline"]),
        )
    if kind == "step":
        return StepCFGSchedule(
            high=float(spec["high"]),
            low=float(spec["low"]),
            transition=float(spec.get("transition", 0.5)),
        )
    raise ValueError(
        f"Unknown CFG schedule type: {kind!r}. "
        "Allowed: 'constant', 'linear', 'cosine_ramp', 'cosine', 'step'."
    )


def make_per_axis_schedule(
    spec: Mapping[str, Mapping[str, object]] | None,
) -> PerAxisSchedule | None:
    """Build a :class:`PerAxisSchedule` from a YAML-style nested mapping.

    Returns ``None`` when ``spec`` is None or empty so callers can pass
    the result straight to ``cfg_sample_step(schedule=...)``.
    """
    if not spec:
        return None
    return PerAxisSchedule(schedules={axis: make_schedule(cfg) for axis, cfg in spec.items()})


__all__ = [
    "CFGSchedule",
    "ConstantCFGSchedule",
    "CosineCFGSchedule",
    "LinearCFGSchedule",
    "OscillatingCosineCFGSchedule",
    "PerAxisSchedule",
    "StepCFGSchedule",
    "make_per_axis_schedule",
    "make_schedule",
]
