"""NEX-reference route witness: ``data.use_repetitions`` is read by one route.

``use_repetitions: true`` promises a high-SNR target -- the average of a scan's
repetitions (M4Raw ships 2-3 per contrast). Only the ``m4raw`` route
(``DatasetInstantiator._create_m4raw_repetition`` -> ``M4RawRepetitionDataset``)
builds that average. Every other route -- ``kspace``, ``bart_kspace``,
``ismrmrd_kspace``, ``oracle_bssfp``, ... -- never reads the knob and serves
each single, thermal-noise-limited record as its own "fully sampled" target.
The run trains and validates, the YAML says NEX, and the reference the metrics
grade against is one noisy scan (CLAUDE.md pitfall #15, an advertised knob with
no reader).

Measured 2026-09-02 across ``experiments/inprogress``: 79 arms carried the knob
on a route that ignores it (``vf`` 53, ``diffusion`` 8, ``cold_diffusion`` 4,
...). The predicate lived only in the kspace cohort's own test until then; it
is now one function, ``nex_reference_route_defect``, consumed by this witness
(audit, resolved or raw config) and by the corpus test under
``tests/unit/experiments`` (every tracked arm, raw YAML) -- one owner, two
drivers (non-negotiable 17).

Registration is by import: the witness package walk in
``validation/witness/__init__.py::_discover`` imports every non-package module
under ``checks/``.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["NEX_ROUTES", "nex_reference_route", "nex_reference_route_defect"]

_NAME = "nex_reference_route"
_CATEGORY = "data_reference"

#: The ``dataset_type`` values whose dataset builds a repetition-averaged
#: target. Adding a route here is a claim about ``DatasetInstantiator``: the
#: corpus test and the instantiator test both pin it.
NEX_ROUTES: frozenset[str] = frozenset({"m4raw"})


def nex_reference_route_defect(dataset_type: object, use_repetitions: object) -> str | None:
    """The one predicate. ``None`` when the declaration is honest.

    Args:
        dataset_type: ``data.dataset_type`` as declared (raw YAML or resolved).
        use_repetitions: ``data.use_repetitions`` as declared; ``None`` means
            "the route decides" and is never a defect.

    Returns:
        A one-paragraph defect message, or ``None``.
    """
    if use_repetitions is not True:
        return None
    if dataset_type in NEX_ROUTES:
        return None
    return (
        f"data.use_repetitions: true with data.dataset_type={dataset_type!r}. Only "
        f"{sorted(NEX_ROUTES)} builds a repetition-averaged (NEX) target; this route "
        "never reads the knob, so the validation reference is a single noisy scan "
        "while the YAML advertises a high-SNR average (pitfall #15). Set "
        "dataset_type: m4raw to get the NEX target, or drop the knob and stop "
        "calling the target NEX-averaged."
    )


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    # The failure-class taxonomy the registry refers to is not published
    # in-tree (no module consumes ``covered_class_ids``), so no id is claimed
    # rather than an invented one.
    severity=Severity.ERROR,
    description="data.use_repetitions is honoured only by dataset_type m4raw",
    fix_hint=(
        "Set data.dataset_type: m4raw (the only route that averages repetitions), "
        "or remove data.use_repetitions and describe the target as a single scan."
    ),
)
def nex_reference_route(subject: WitnessSubject) -> WitnessVerdict:
    """Error when an arm advertises a NEX target on a route that cannot build it."""
    data = subject.raw_config.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    defect = nex_reference_route_defect(data.get("dataset_type"), data.get("use_repetitions"))
    if defect is None:
        declared = data.get("use_repetitions")
        message = (
            "use_repetitions not declared; the route decides"
            if declared is None
            else f"use_repetitions={declared!r} on dataset_type={data.get('dataset_type')!r}"
        )
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=message,
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T0,
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=False,
        message=defect,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T0,
        yaml_keys=("data.use_repetitions", "data.dataset_type"),
        fix_hint=(
            "Set data.dataset_type: m4raw, or remove data.use_repetitions and stop "
            "describing the target as NEX-averaged."
        ),
    )
