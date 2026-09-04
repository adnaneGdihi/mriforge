"""Schedule-certification witnesses: line allocation, step-to-cap, defect margin.

Split out of :mod:`schedule_certification_checks` (#1400, 300-LOC ceiling).
Registration is by import -- see :mod:`schedule_nesting_checks` for the detail.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks.schedule_certification_common import (
    _CATEGORY,
    _declared_certification_levels,
    _is_cold_diffusion,
    _matrix,
    _not_applicable,
    build_process_from_config,
    synthetic_spectral_prior,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

_ALLOC_NAME = "schedule.line_allocation"


@register_witness(
    _ALLOC_NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T2,),
    subjects=(Subject.CONFIG,),
    severity=Severity.WARNING,
    description=(
        "C4: removed k-space energy is spread across levels under a 1/(1+|k|^2) "
        "spectral prior (no single level carries most of the budget)"
    ),
    fix_hint=(
        "Spread the low-frequency lines across several levels (smaller per-level "
        "acceleration increments near R=1) instead of removing the centre band in "
        "one step; refine T rather than stride."
    ),
)
def schedule_line_allocation(subject: WitnessSubject) -> WitnessVerdict:
    """No level should carry more than half of all removed spectral energy."""
    process = build_process_from_config(subject.raw_config)
    if process is None:
        return _not_applicable(_ALLOC_NAME, Tier.T2)
    shape = _matrix(subject.raw_config)
    prior = synthetic_spectral_prior(shape)
    per_level = process.removed_line_energy_stats(shape, prior, domain="kspace")
    top = max(per_level, key=lambda s: s["share"], default=None)
    threshold = 0.5
    if top is None or top["share"] <= threshold:
        share = 0.0 if top is None else top["share"]
        return WitnessVerdict(
            witness_name=_ALLOC_NAME,
            passed=True,
            message=(
                f"max per-level removed-energy share {share:.1%} <= {threshold:.0%} "
                f"under the synthetic spectral prior at {shape[0]}x{shape[1]}"
            ),
            severity=Severity.WARNING,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T2,
        )
    return WitnessVerdict(
        witness_name=_ALLOC_NAME,
        passed=False,
        message=(
            f"level t={top['t']} carries {top['share']:.1%} of all removed "
            f"spectral energy (> {threshold:.0%}) under the synthetic prior: its "
            "per-level ambiguity likely exceeds its step budget, making that "
            "level the fabrication bottleneck (C4)."
        ),
        severity=Severity.WARNING,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T2,
    )


_CAP_NAME = "schedule.step_to_reach_cap"


@register_witness(
    _CAP_NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T2,),
    subjects=(Subject.CONFIG,),
    severity=Severity.INFO,
    description=(
        "C2: declared step-to-reach ratios kappa_t = delta_hat/tau_hat stay at or "
        "below 1/2 (per-step Lipschitz factor <= 2)"
    ),
    fix_hint=(
        "Refine T (smaller delta_t per level) rather than striding, or re-estimate "
        "tau_hat with core/metrics/manifold_diagnostics.estimate_reach and declare "
        "the results under undersampling.certification.per_level."
    ),
)
def schedule_step_to_reach_cap(subject: WitnessSubject) -> WitnessVerdict:
    """C2 on declared estimates only — the config surface cannot estimate reach."""
    if not _is_cold_diffusion(subject.raw_config):
        return _not_applicable(_CAP_NAME, Tier.T2, Severity.INFO)
    from spectramr.core.metrics.manifold_diagnostics import step_budget_ratio

    levels = [
        lv
        for lv in _declared_certification_levels(subject.raw_config)
        if "delta_hat" in lv and "tau_hat" in lv
    ]
    if not levels:
        return WitnessVerdict(
            witness_name=_CAP_NAME,
            passed=True,
            message=(
                "C2 not evaluable on the config surface: no "
                "undersampling.certification.per_level entries with delta_hat and "
                "tau_hat; estimate offline with "
                "core/metrics/manifold_diagnostics and declare the results."
            ),
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T2,
        )
    verdicts = [
        (lv.get("t"), step_budget_ratio(float(lv["delta_hat"]), float(lv["tau_hat"])))
        for lv in levels
    ]
    offenders = [(t, v) for t, v in verdicts if not v["satisfies_c2"]]
    if not offenders:
        worst = max(v["kappa"] for _, v in verdicts)
        return WitnessVerdict(
            witness_name=_CAP_NAME,
            passed=True,
            message=(
                f"all {len(verdicts)} declared levels satisfy C2 "
                f"(max kappa {worst:.3f} <= 0.5, Lipschitz factor <= 2)"
            ),
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T2,
        )
    t_worst, v_worst = max(offenders, key=lambda tv: tv[1]["kappa"])
    ill_posed = sum(1 for _, v in offenders if not v["well_posed"])
    return WitnessVerdict(
        witness_name=_CAP_NAME,
        passed=False,
        message=(
            f"{len(offenders)} of {len(verdicts)} declared levels exceed the C2 cap "
            f"kappa <= 0.5 (worst: t={t_worst}, kappa {v_worst['kappa']:.3f}, "
            f"amplification {v_worst['amplification']:.2f}"
            + (f"; {ill_posed} level(s) even break well-posedness kappa < 1" if ill_posed else "")
            + "): errors compound through the product of per-step Lipschitz factors."
        ),
        severity=Severity.INFO,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T2,
        yaml_keys=("undersampling.certification",),
        fix_hint="Refine T rather than stride (C2); re-check tau_hat's uncertainty.",
    )


_DEFECT_NAME = "schedule.tangential_defect_margin"


@register_witness(
    _DEFECT_NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T3,),
    subjects=(Subject.CONFIG,),
    severity=Severity.WARNING,
    description=(
        "C3: declared tangential defects clear the contraction threshold "
        "theta_hat < 1 - kappa_t at every level"
    ),
    fix_hint=(
        "Reduce delta_t at the failing level (raises the threshold) or reorder "
        "which k-space lines leave at which level (changes theta_t); measure "
        "theta_hat with core/metrics/manifold_diagnostics.tangential_defect."
    ),
)
def schedule_tangential_defect_margin(subject: WitnessSubject) -> WitnessVerdict:
    """C3 on declared estimates: contraction needs theta_t < 1 - kappa_t (2.9)."""
    if not _is_cold_diffusion(subject.raw_config):
        return _not_applicable(_DEFECT_NAME, Tier.T3)
    from spectramr.core.metrics.manifold_diagnostics import step_budget_ratio

    levels = [
        lv
        for lv in _declared_certification_levels(subject.raw_config)
        if "delta_hat" in lv and "tau_hat" in lv and "theta_hat" in lv
    ]
    if not levels:
        return WitnessVerdict(
            witness_name=_DEFECT_NAME,
            passed=True,
            message=(
                "C3 not evaluable on the config surface: no "
                "undersampling.certification.per_level entries with theta_hat, "
                "delta_hat and tau_hat; estimate offline with "
                "core/metrics/manifold_diagnostics and declare the results."
            ),
            severity=Severity.WARNING,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T3,
        )
    offenders = []
    for lv in levels:
        kappa = step_budget_ratio(float(lv["delta_hat"]), float(lv["tau_hat"]))["kappa"]
        theta = float(lv["theta_hat"])
        if theta >= 1.0 - kappa:
            offenders.append((lv.get("t"), theta, 1.0 - kappa))
    if not offenders:
        return WitnessVerdict(
            witness_name=_DEFECT_NAME,
            passed=True,
            message=(
                f"all {len(levels)} declared levels satisfy the contraction "
                "condition theta_hat < 1 - kappa"
            ),
            severity=Severity.WARNING,
            category=_CATEGORY,
            stage=Stage.CONSTRUCT,
            tier=Tier.T3,
        )
    t_worst, theta_worst, threshold_worst = max(offenders, key=lambda o: o[1] - o[2])
    return WitnessVerdict(
        witness_name=_DEFECT_NAME,
        passed=False,
        message=(
            f"{len(offenders)} of {len(levels)} declared levels violate "
            f"theta_hat < 1 - kappa (worst: t={t_worst}, theta_hat "
            f"{theta_worst:.3f} >= threshold {threshold_worst:.3f}): the reverse "
            "step does not contract the ball family there and the per-step bias "
            "guarantee degrades to the do-nothing-order bound (C3)."
        ),
        severity=Severity.WARNING,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T3,
        yaml_keys=("undersampling.certification",),
        fix_hint=("Reduce delta_t at the failing level or reorder the line removals."),
    )


__all__ = [
    "schedule_line_allocation",
    "schedule_step_to_reach_cap",
    "schedule_tangential_defect_margin",
]
