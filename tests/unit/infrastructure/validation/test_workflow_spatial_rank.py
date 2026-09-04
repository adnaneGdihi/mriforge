"""Tests for check_workflow_spatial_rank."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.enums import Regime
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _canonical(dataset_type: str) -> str:
    """Reject a ``dataset_type`` no validated config could ever hold.

    The checker reads ``config.data.dataset_type`` *after*
    ``DataConfigSchema.validate_dataset_type`` has lower-cased it, mapped aliases
    onto canonical names and rejected the rest. A test that reaches the checker
    through a ``SimpleNamespace`` skips all of that, so it can assert on a string
    the production path cannot produce.

    That is not hypothetical: this file asserted ``"2d"`` was rejected for rank
    and ``"3d"`` accepted, and both passed only because ``DATASET_TYPE_RANKS``
    carried those keys. ``"2d"`` is an alias normalised to ``"image"`` and
    ``"3d"`` to ``"nifti"`` long before the check runs, so neither assertion
    described anything reachable. Routing every case through the real validator
    is what keeps these tests on the seam.
    """
    normalised = DataConfigSchema.validate_dataset_type(dataset_type)
    assert normalised == dataset_type, (
        f"{dataset_type!r} is not a canonical dataset_type: the validator "
        f"normalises it to {normalised!r}, so no loaded config reaches "
        f"check_workflow_spatial_rank carrying {dataset_type!r}."
    )
    return dataset_type


def _check(
    regime: Regime | None, dataset_type: str | None, spatial_rank: int | None = None
):
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = (
        WorkflowConfigSchema(regime=regime, spatial_rank=spatial_rank)
        if regime is not None
        else None
    )
    cfg = SimpleNamespace(
        workflow=workflow,
        data=SimpleNamespace(
            dataset_type=_canonical(dataset_type) if dataset_type else dataset_type
        ),
    )
    return checker.check_workflow_spatial_rank(cfg)


def test_functional_requires_rank3_rejects_a_rank2_dataset() -> None:
    # mri_functional is rank-3 only; npy_slice is annotated rank 2.
    r = _check(Regime.FUNCTIONAL, "npy_slice")
    assert not r.passed
    assert r.severity == "error"
    assert "rank 2" in r.message


def test_no_canonical_dataset_type_declares_rank_3() -> None:
    """The check can reject or skip a rank-3-only regime, never affirm one.

    Every rank-3-capable route degenerates to rank 2 through a knob this table
    cannot see: ``nifti`` emits per-slice records under ``variant:
    "2d_slices"``, and a single-slice ``cine`` arrives in TorchIO's 2-D layout.
    Annotating either rank 3 would be a claim the config can contradict, so
    ``DATASET_TYPE_RANKS`` honestly declares none — which means mri_functional
    cannot be *confirmed* by rank on any real dataset_type today.

    Pinned so the absence reads as a known limit rather than an oversight, and
    so that adding a rank-3 annotation forces a decision here.
    """
    from spectramr.data.datasets.axis_exposure import DATASET_TYPE_RANKS

    assert 3 not in DATASET_TYPE_RANKS.values()


def test_structural_accepts_a_rank2_dataset() -> None:
    # mri_structural spans ranks {2, 3}, so an annotated rank-2 type passes.
    assert _check(Regime.STRUCTURAL, "npy_slice").passed
    assert _check(Regime.STRUCTURAL, "m4raw").passed


def test_unannotated_dataset_type_is_skipped() -> None:
    r = _check(Regime.FUNCTIONAL, "image")  # "image" rank is ambiguous → skip
    assert r.passed
    assert "skipped" in r.message


def test_no_workflow_is_skipped() -> None:
    assert _check(None, "image").passed


class TestTheDeclaredRank:
    """``workflow.spatial_rank`` is the third statement of rank, alongside the
    regime's ``spatial_ranks`` and the rank the ``dataset_type`` provides. All
    three pairings are checked, in ascending order of what they need to know.
    """

    def test_a_rank_the_regime_forbids_is_an_error(self) -> None:
        r = _check(Regime.FUNCTIONAL, "npy_slice", spatial_rank=2)
        assert not r.passed and r.severity == "error"
        assert "workflow.spatial_rank=2" in r.message
        assert r.yaml_keys == ["workflow.regime", "workflow.spatial_rank"]

    def test_the_profile_check_runs_even_when_the_dataset_rank_is_unknown(self) -> None:
        """Ordering, not decoration. Declared-vs-profile needs neither the
        dataset nor its annotation, so putting it after the "unannotated
        dataset_type -> skip" branch would let an unannotated dataset swallow an
        outright contradiction."""
        r = _check(Regime.FUNCTIONAL, None, spatial_rank=2)
        assert not r.passed, (
            "an absent dataset_type swallowed a declared rank the regime "
            "forbids; the declared-vs-profile check is sitting behind the skip."
        )

    def test_declared_disagreeing_with_the_data_is_an_error(self) -> None:
        """The most interesting of the three: an arm claiming 3-D while feeding
        2-D slices makes a claim the run cannot honour."""
        r = _check(Regime.STRUCTURAL, "npy_slice", spatial_rank=3)
        assert not r.passed and r.severity == "error"
        assert "provides rank 2" in r.message
        assert r.yaml_keys == ["workflow.spatial_rank", "data.dataset_type"]

    def test_declared_agreeing_with_the_data_passes(self) -> None:
        r = _check(Regime.STRUCTURAL, "npy_slice", spatial_rank=2)
        assert r.passed and r.severity == "info"

    def test_omitting_it_leaves_the_original_behaviour(self) -> None:
        """Absent is advisory. The pre-existing dataset-vs-profile check is the
        only thing that fires."""
        assert _check(Regime.STRUCTURAL, "npy_slice").passed
        assert not _check(Regime.FUNCTIONAL, "npy_slice").passed
