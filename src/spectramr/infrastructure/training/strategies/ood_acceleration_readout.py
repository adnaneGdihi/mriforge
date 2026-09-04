"""Out-of-distribution acceleration readout for the twin-driven strategies (VF review 2026-09-03).

``physics.digital_twin.ood_acceleration_range`` names accelerations the arm
never trains at. At validation the twin is run at each of them, the model is
run again, and the scores land beside the in-distribution ones as
``val_ood_{R}x_<metric>``. The two strategies that undersample through the twin
(``virtual_fiducial``, ``vf_admm``) call this one module, so the key format,
the count key and the restore discipline have a single owner.

The 37 arms that declared ``undersampling.out_of_distribution_range`` before
this module existed declared a key nothing read (dropped-key baseline, 58
rows); the range now lives beside the twin's own ``acceleration`` because on a
twin arm that block is the acceleration's owner (``undersampling_checks``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "OOD_COUNT_KEY",
    "ood_acceleration_readout",
    "ood_accelerations",
    "ood_metric_key",
]

#: Written on every validation step of a reading strategy: ``0`` says the arm
#: declared no range, so "no OOD rows" and "OOD off" are told apart in the log.
OOD_COUNT_KEY = "val_ood_accelerations"


def ood_accelerations(config: Any) -> tuple[float, ...]:
    """The declared rungs, or ``()`` when the twin block or the range is absent.

    A range on a twin that is disabled or does not undersample cannot reach
    here: ``DigitalTwinConfig``'s validator refuses it at load.
    """
    physics = getattr(config, "physics", None)
    twin = getattr(physics, "digital_twin", None)
    if twin is None:
        return ()
    declared = twin.ood_acceleration_range
    if not declared:
        return ()
    return tuple(float(rate) for rate in declared)


def ood_metric_key(acceleration: float, name: str) -> str:
    """``val_ood_16x_psnr`` for ``(16.0, "val_psnr")``; the ``val_`` prefix is not doubled."""
    return f"val_ood_{acceleration:g}x_{name.removeprefix('val_')}"


def ood_acceleration_readout(
    simulator: Any,
    accelerations: tuple[float, ...],
    score: Callable[[], Mapping[str, float]],
) -> dict[str, float]:
    """Score once per rung with the twin held at that rung; always write the count.

    ``score`` is the strategy's own corrupt-reconstruct-score pass, closed over
    the batch; it reads the twin as it is set, so ``simulator.at_acceleration``
    is the only thing that changes between rungs. Every returned key is
    re-keyed through :func:`ood_metric_key`.
    """
    out: dict[str, float] = {}
    for rate in accelerations:
        with simulator.at_acceleration(rate):
            scored = score()
        for name, value in scored.items():
            out[ood_metric_key(rate, name)] = float(value)
    out[OOD_COUNT_KEY] = float(len(accelerations))
    return out
