"""Schedule-certification witnesses: nesting leak-freedom and inert steps.

Split out of :mod:`schedule_certification_checks` (#1400, 300-LOC ceiling).
Registration is by import -- the witness package walk in
``validation/witness/__init__.py::_discover`` imports every non-package module
under ``checks/``, so these register exactly as before. Pinned by
``test_schedule_certification_split.py``, which asserts all five names are in
the registry after discovery rather than assuming the walk found them.
"""

from __future__ import annotations

from mriforge.infrastructure.validation.witness.checks.schedule_certification_common import (
    _CATEGORY,
    _matrix,
    _not_applicable,
    build_process_from_config,
)
from mriforge.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from mriforge.infrastructure.validation.witness.subject import WitnessSubject

_NEST_NAME = "schedule.nesting_leakfree"


@register_witness(
    _NEST_NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T1,),
    subjects=(Subject.CONFIG,),
    severity=Severity.WARNING,
    description="C1: the mask cascade is leak-free (kept-sets nested K_T ⊆ ... ⊆ K_0)",
    fix_hint=(
        "Set undersampling.enforce_nested: true (and keep the fixed-seed cascade) "
        "so no removed k-space bin is ever re-introduced; a leak breaks the "
        "cold-diffusion cocycle and forces a schedule-induced fabrication floor."
    ),
)
def schedule_nesting_leakfree(subject: WitnessSubject) -> WitnessVerdict:
    """Leak-free ⇔ nested masks ⇔ the reverse theory is statable at every level."""
    process = build_process_from_config(subject.raw_config)
    if process is None:
        return _not_applicable(_NEST_NAME, Tier.T1)
    shape = _matrix(subject.raw_config)
    leaks = process.nesting_leak_report(shape)
    if not leaks:
        # A pass here says the cascade validation SEES is nested. When enforcement is
        # on that is true by construction, so the verdict alone cannot distinguish a
        # family that nests on its own from one whose leaks the cumulative
        # intersection deleted — and the intersection pays in sampling budget. Re-ask
        # without enforcement so the message says which of the two it is.
        provenance = ""
        if getattr(process, "enforce_nested", False):
            raw_leaks = process.nesting_leak_report(shape, raw=True)
            provenance = (
                " by construction (the family's own cascade is leak-free too, so "
                "enforce_nested is a no-op here)"
                if not raw_leaks
                else (
                    f" only because enforce_nested intersected it — the family's own "
                    f"cascade leaks at {len(raw_leaks)} of {process.num_timesteps} "
                    f"levels, so the enforced masks realise less than the declared "
                    f"sampling budget"
                )
            )
        return WitnessVerdict(
            witness_name=_NEST_NAME,
            passed=True,
            message=f"cascade is leak-free at {shape[0]}x{shape[1]} "
            f"({process.num_timesteps} levels){provenance}",
            severity=Severity.WARNING,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T1,
            yaml_keys=("undersampling.enforce_nested",),
        )
    worst = max(leaks, key=lambda d: d["reintroduced_bins"])
    return WitnessVerdict(
        witness_name=_NEST_NAME,
        passed=False,
        message=(
            f"{len(leaks)} of {process.num_timesteps} levels re-introduce removed "
            f"k-space bins (worst: t={worst['t']}, {worst['reintroduced_bins']} bins, "
            f"leak fraction {worst['leak_fraction']:.1%}); the cascade is not nested, "
            "per-level data consistency does not compose, and chainwise trust "
            "monitoring is unsound at the leaking levels."
        ),
        severity=Severity.WARNING,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T1,
        yaml_keys=("undersampling.enforce_nested",),
        fix_hint="Set undersampling.enforce_nested: true.",
    )


_INERT_NAME = "schedule.no_inert_steps"


@register_witness(
    _INERT_NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T1,),
    subjects=(Subject.CONFIG,),
    severity=Severity.WARNING,
    description="No forward level leaves the mask unchanged (degenerate timestep axis)",
    fix_hint=(
        "Reduce the number of timesteps, widen the acceleration range, or move to "
        "a schedule whose every level removes at least one bin at this matrix size."
    ),
)
def schedule_no_inert_steps(subject: WitnessSubject) -> WitnessVerdict:
    """Every level of the cascade should remove something."""
    process = build_process_from_config(subject.raw_config)
    if process is None:
        return _not_applicable(_INERT_NAME, Tier.T1)
    shape = _matrix(subject.raw_config)
    inert = process.inert_step_report(shape)
    if not inert:
        return WitnessVerdict(
            witness_name=_INERT_NAME,
            passed=True,
            message=f"no inert levels at {shape[0]}x{shape[1]}",
            severity=Severity.WARNING,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T1,
        )
    return WitnessVerdict(
        witness_name=_INERT_NAME,
        passed=False,
        message=(
            f"{len(inert)} of {process.num_timesteps} levels are inert "
            f"(mask unchanged; first at t={inert[0]}): the timestep axis is "
            "degenerate there and scheduled reverse steps have nothing to reveal "
            "(issue #535's forward-side counterpart)."
        ),
        severity=Severity.WARNING,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T1,
    )


__all__ = ["schedule_nesting_leakfree", "schedule_no_inert_steps"]
