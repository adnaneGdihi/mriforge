"""Meta-witness: a check that exists but is never called protects nothing.

``ConfigHealthChecker`` carries 126 ``check_*`` methods, and ``run_all_checks``
invokes them as a hand-written 275-line sequence of ``report.results.append(...)``
calls. There is no registry, no reflection, and no list — so a method can be
written, reviewed, merged, and never run. That already happened twice.

This is the ``META`` stage of the taxonomy: the guard itself is silent. It is the
class that has claimed a `yaml.safe_load` corpus gate reporting `defects: none`
for 47 arms, a `branches:` filter that disabled every check on every PR until
2026-07-12, a `check_experiment_configs_load` that did not load configs (#530),
and Sphinx warning rather than failing until 87 of 210 sidebar entries were dead.

The detector is a static AST walk of ``run_all_checks``' own source, which is a
complete substitute for instrumenting a live run **because every append/extend in
that method is unconditional** — there is no per-config branching that could skip
a wired call. If that ever changes, this witness becomes optimistic and must be
replaced with runtime instrumentation.

``KNOWN_INERT`` requires a one-line reason per entry, same discipline as
``ladder_baseline.txt``: an allowlist entry is tracked debt, not permission. A bare
diff would false-positive here, because the two current exclusions are deliberate
and documented in ``run_all_checks``' own trailing comment.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from mriforge.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from mriforge.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["KNOWN_INERT", "defined_check_methods", "invoked_check_methods"]

#: Deliberately-unwired checks. Each needs a reason; removing one is the goal.
KNOWN_INERT: dict[str, str] = {
    "check_svd_compression_phase_safety": (
        "superseded by a hard raise in the SVD coil-compression transform"
    ),
    "check_validation_image_domain_safe": (
        "superseded by the image-domain-only guard in the diffusion strategy"
    ),
}

_WITNESS_NAME = "meta.health_checker_no_orphan_checks"


def defined_check_methods() -> set[str]:
    from mriforge.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    return {
        name
        for name, _ in inspect.getmembers(ConfigHealthChecker, inspect.isfunction)
        if name.startswith("check_")
    }


def invoked_check_methods() -> set[str]:
    """Every ``self.check_*`` named inside ``run_all_checks``' body."""
    from mriforge.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    source = textwrap.dedent(inspect.getsource(ConfigHealthChecker.run_all_checks))
    tree = ast.parse(source)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("check_")
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


@register_witness(
    _WITNESS_NAME,
    category="meta",
    stage=Stage.META,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.NONE,),
    class_ids=("S11.3",),
    severity=Severity.ERROR,
    fix_hint=(
        "Wire the method into run_all_checks, or add it to KNOWN_INERT with a "
        "one-line reason explaining what supersedes it."
    ),
)
def health_checker_has_no_orphan_checks(_subject: WitnessSubject) -> WitnessVerdict:
    """Every ``check_*`` on ConfigHealthChecker is invoked or explicitly allowlisted."""
    defined = defined_check_methods()
    invoked = invoked_check_methods()
    orphaned = defined - invoked - set(KNOWN_INERT)

    if orphaned:
        return WitnessVerdict(
            witness_name=_WITNESS_NAME,
            passed=False,
            severity=Severity.ERROR,
            category="meta",
            class_ids=("S11.3",),
            stage=Stage.META,
            message=(
                f"{len(orphaned)} check_* method(s) are defined but never invoked "
                f"by run_all_checks, and not allowlisted: {sorted(orphaned)}. A "
                f"check that is never called protects nothing."
            ),
            fix_hint=(
                "Wire it into run_all_checks, or add it to KNOWN_INERT with a one-line reason."
            ),
        )

    stale = set(KNOWN_INERT) - (defined - invoked)
    if stale:
        return WitnessVerdict(
            witness_name=_WITNESS_NAME,
            passed=False,
            severity=Severity.WARNING,
            category="meta",
            class_ids=("S11.3",),
            stage=Stage.META,
            message=(
                f"KNOWN_INERT lists {sorted(stale)}, which are now invoked or no "
                f"longer exist. A stale allowlist hides the next real orphan."
            ),
            fix_hint="Remove the entry from KNOWN_INERT.",
        )

    return WitnessVerdict(
        witness_name=_WITNESS_NAME,
        passed=True,
        severity=Severity.INFO,
        category="meta",
        class_ids=("S11.3",),
        stage=Stage.META,
        message=(
            f"{len(defined)} check_* methods: {len(invoked & defined)} invoked, "
            f"{len(KNOWN_INERT)} allowlisted, 0 orphaned."
        ),
    )
