"""DatasetInstantiator dispatch for dataset_type='mrixfields'."""

from __future__ import annotations

import torch

from mriforge.data.builders.dataset_instantiator import DatasetInstantiator
from mriforge.data.datasets.mrixfields_dataset import MRIxFieldsPairedDataset
from tests.utils.data_config_stub import DataConfigStub


def _cfg():
    """Stand-in with every phase-9 sub-block at its real schema defaults."""
    return DataConfigStub(
        dataset_type="mrixfields", mrixfields_pairing_policy="all_pairs"
    )


def _idx() -> list[dict]:
    base = {"subject_id": "s1", "contrast": "T1w", "pairing_group": "s1|T1w"}
    return [
        {**base, "primary_path": f"f{b}.nii", "field_strength": b}
        for b in (0.1, 3.0, 7.0)
    ]


def test_instantiator_builds_mrixfields(monkeypatch) -> None:
    # Patch the WHOLE-VOLUME loader: the default slice_mode is 'all_slices', which
    # decodes volumes at construction (the foreground scan) via
    # _default_nifti_volume_loader, not the central-slice loader. A depth-3 all-
    # foreground volume expands each of the 6 all_pairs pairs to 3 slices -> 18.
    monkeypatch.setattr(
        "mriforge.data.datasets.mrixfields_dataset._default_nifti_volume_loader",
        lambda p: torch.rand(1, 8, 8, 3),
    )
    train, val = DatasetInstantiator.create_datasets(_cfg(), _idx(), _idx(), None, None)
    assert isinstance(train, MRIxFieldsPairedDataset)
    assert isinstance(val, MRIxFieldsPairedDataset)
    assert len(train) == 6 * 3  # 6 pairs x 3 foreground slices (all_slices default)
    # The instantiator threads the resident-volume budget from the schema default —
    # the seam that silently broke when the knob was added.
    assert train._max_resident_volumes == 6
    assert train._slice_mode == "all_slices"


def _grp(subject: str, fields: tuple[float, ...]) -> list[dict]:
    return [
        {
            "subject_id": subject,
            "contrast": "T1w",
            "pairing_group": f"{subject}|T1w",
            "primary_path": f"{subject}_{f}.nii",
            "field_strength": f,
        }
        for f in fields
    ]


def test_regroup_honors_explicit_val_boundary() -> None:
    """When an explicit validation manifest is supplied, the train/val split is
    authoritative: field-pinned regroup must NOT merge+re-split (which would leak
    the disjoint validation subjects into the training pool). Anchor: the
    mrixfields2026_val.json wiring, 2026-07-06."""
    train = _grp("tr1", (0.1, 7.0)) + _grp("tr2", (0.1, 7.0))
    val = _grp("va1", (0.1, 7.0)) + _grp("va2", (0.1, 7.0))

    tr, vl, _src = DatasetInstantiator._regroup_mrixfields_multi_source(
        train, val, 7.0, 0.1, explicit_val=True
    )

    tr_subs = {r["subject_id"] for r in tr}
    vl_subs = {r["subject_id"] for r in vl}
    assert tr_subs == {"tr1", "tr2"}
    assert vl_subs == {"va1", "va2"}
    assert not (tr_subs & vl_subs)  # no leakage across the explicit boundary


def test_regroup_resplits_when_no_explicit_val() -> None:
    """Without an explicit manifest (val came from a random split), the field-pinned
    regroup still merges + group-slices so every field is co-resident per split
    (the field-stranding guard is preserved)."""
    # A single field-sorted manifest handed in as (train=all, val=[]): the regroup
    # must carve a non-empty val holding COMPLETE groups.
    full = _grp("s1", (0.1, 7.0)) + _grp("s2", (0.1, 7.0)) + _grp("s3", (0.1, 7.0))
    tr, vl, _src = DatasetInstantiator._regroup_mrixfields_multi_source(
        full, [], 7.0, 0.34, explicit_val=False
    )
    assert vl, "internal split must produce a non-empty validation set"
    # Each emitted split holds whole groups (both fields present for its subjects).
    for split in (tr, vl):
        by_sub: dict[str, set] = {}
        for r in split:
            by_sub.setdefault(r["subject_id"], set()).add(r["field_strength"])
        assert all(fields == {0.1, 7.0} for fields in by_sub.values())
