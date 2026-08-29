"""The attention shootout's cohort-level invariants.

Ten arms whose contract is that they differ in ``attention_type`` and nothing
else that could move a metric. Two invariants, and they are invariants of
different kinds -- one scientific, one declarative.

**Effective batch size** is the scientific one, and the reason it is stated as a
PRODUCT is the whole point. Four arms (the three KAN variants and kernelized)
run ``batch_size: 1`` with ``accumulation_steps: 4`` while the other six run
``2`` x ``2``. Read as raw keys that is a cohort split down the middle; read as
the quantity the optimizer actually sees it is 4 everywhere. The four are the
heavy arms and the 1x4 split is a memory accommodation -- two of them also
enable gradient checkpointing -- chosen precisely to hold the optimization math
fixed. "Standardising" those keys to match the control would equalise the YAML
and change nothing about the experiment except to OOM the arms that needed the
accommodation. So the assertion is on the product, which is what a confound
would actually move.

Rank count is the other multiplier and no YAML can pin it: ``num_devices`` is
overwritten from ``LOCAL_WORLD_SIZE`` by ``pipelines/distributed.py`` on any
``train-distributed`` launch. The cohort is therefore comparable at any world
size PROVIDED every arm is launched at the same one. That is launch discipline,
not config, and it is stated here because this file is where someone will look.

**First-class scientific metadata** is the declarative one. ``hypothesis`` /
``primary_metric`` / ``baseline`` began life smuggled into ``metadata.tags``
because ``ExperimentMetadataSchema`` was ``extra="forbid"`` and had nowhere else
to put them. The 2026-06 validation campaign made them real fields; six arms
moved and four did not, which is not cosmetic: ``check_scientific_metadata``
returns ``[]`` outright when ``metadata.primary_metric`` is unset, so those four
were skipping the check entirely while looking, in the YAML, exactly as
annotated as the rest. A free-form bag that nothing reads is the facade shape --
the declaration was present and inert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.config.settings import TrainingSettings  # noqa: E402
from tests.utils.corpus import tracked_yamls  # noqa: E402

COHORT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "attention_shootout"
)
CONTROL = "experiment_11_attention_none"


def _arms() -> dict[str, TrainingSettings]:
    """``stem -> resolved settings`` for every shootout arm.

    Resolved, never raw YAML: this cohort has already been bitten once by a
    raw-text predicate that went vacuously green when a canonical-key drain
    moved a block without changing the resolved document
    (``test_exp11_attention_none_lookahead``). A ``metadata.tags.primary_metric``
    vs ``metadata.primary_metric`` split is exactly that shape of difference.
    """
    return {
        p.stem: TrainingSettings.from_yaml(str(p))
        for p in tracked_yamls(COHORT, "experiment_11_attention_*.yaml", recursive=False)
    }


@pytest.fixture(scope="module")
def arms() -> dict[str, TrainingSettings]:
    return _arms()


def _meta(cfg: TrainingSettings, field: str):
    """Read ``metadata.<field>`` through either shape it resolves to.

    ``TrainingSettings.metadata`` is ``dict[str, Any] | None``
    (``config/settings.py:176``), while the sub-schema path uses
    ``ExperimentMetadataSchema`` (an object). ``check_scientific_metadata``
    carries the same two-branch read for the same reason; a test that assumed
    only the object shape would ``AttributeError`` on every arm rather than
    assert anything -- loud, but about the wrong thing.
    """
    meta = getattr(cfg, "metadata", None)
    if isinstance(meta, dict):
        return meta.get(field)
    return getattr(meta, field, None) if meta is not None else None


def test_the_cohort_is_present(arms) -> None:
    """Anti-vacuity for every assertion below: an empty or singleton mapping
    would satisfy the `all(...)` shapes trivially."""
    assert len(arms) >= 2, f"cohort has {len(arms)} arm(s); comparisons are vacuous"
    assert CONTROL in arms, "the control arm is missing from the cohort"


def test_every_arm_shares_the_control_effective_batch(arms) -> None:
    """``batch_size x accumulation_steps`` is what the optimizer sees.

    Not ``batch_size`` alone -- see the module docstring. An arm that trades
    micro-batch for accumulation is holding this invariant, not breaking it.
    """
    effective = {
        stem: cfg.data.loader.batch_size * cfg.optimization.gradient.accumulation_steps
        for stem, cfg in arms.items()
    }
    control = effective[CONTROL]
    divergent = {s: e for s, e in effective.items() if e != control}
    assert not divergent, (
        f"effective batch (batch_size x accumulation_steps) is {control} on the "
        f"control but differs on {divergent}. That is a second axis in a cohort "
        "whose contract is attention_type alone (pitfall #17) -- every reported "
        "difference becomes unattributable. Restore the product; the split "
        "between micro-batch and accumulation is free to differ."
    )


def test_the_axis_under_test_actually_varies(arms) -> None:
    """Anti-vacuity in the other direction: a cohort that agreed on
    ``attention_type`` too would pass every assertion here and be no
    experiment at all."""
    kinds = {stem: cfg.model.model_kwargs.get("attention_type") for stem, cfg in arms.items()}
    assert len(set(kinds.values())) > 1, f"all arms share one attention_type: {kinds}"
    assert kinds[CONTROL] == "none", (
        f"the control declares attention_type={kinds[CONTROL]!r}, not 'none'"
    )


@pytest.mark.parametrize("field", ["hypothesis", "primary_metric"])
def test_every_arm_declares_its_science_first_class(arms, field: str) -> None:
    """``metadata.tags`` is a free-form bag; the audit reads the real fields.

    ``check_scientific_metadata`` returns ``[]`` when ``primary_metric`` is
    unset, so an arm carrying it under ``tags`` is not annotated-but-messy, it
    is unchecked -- and indistinguishable from an annotated arm by eye.
    """
    missing = [stem for stem, cfg in arms.items() if not _meta(cfg, field)]
    assert not missing, (
        f"metadata.{field} is unset on {missing}. If the value is under "
        "`metadata.tags:`, move it up -- tags is free-form and nothing reads it, "
        "so the arm silently skips check_scientific_metadata."
    )


def test_every_non_control_arm_names_its_baseline(arms) -> None:
    """The control needs none -- it IS the baseline (``metadata.role``)."""
    missing = [stem for stem, cfg in arms.items() if stem != CONTROL and not _meta(cfg, "baseline")]
    assert not missing, f"metadata.baseline is unset on {missing}"


def test_a_named_baseline_exists_in_the_cohort(arms) -> None:
    """A baseline pointing at nothing is worse than none: it reads as a
    comparison that was made."""
    dangling = {
        stem: _meta(cfg, "baseline")
        for stem, cfg in arms.items()
        if _meta(cfg, "baseline") and _meta(cfg, "baseline") not in arms
    }
    assert not dangling, f"metadata.baseline names a non-existent arm: {dangling}"


def test_the_science_fields_are_not_also_left_in_tags(arms) -> None:
    """Move, not copy. Two homes for one value is how the two drift apart, and
    the tags copy is the one that would go stale unnoticed."""
    duplicated = {
        stem: [f for f in ("hypothesis", "primary_metric", "baseline") if f in tags]
        for stem, cfg in arms.items()
        if isinstance(tags := (_meta(cfg, "tags") or {}), dict)
        and any(f in tags for f in ("hypothesis", "primary_metric", "baseline"))
    }
    assert not duplicated, (
        f"scientific metadata is still under metadata.tags on {duplicated}; it "
        "belongs in the first-class fields alone (config/schemas/base.py)."
    )
