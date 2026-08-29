"""Tests for MRIxFields2026 paired-dataset data-config knobs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.data import DataConfigSchema


def test_mrixfields_knobs_default_off() -> None:
    cfg = DataConfigSchema(dataset_type="mrixfields", batch_size=2)
    assert cfg.mrixfields.pairing_policy == "all_pairs"
    assert cfg.mrixfields.target_field is None
    # Target field is intrinsic to paired translation -> emitted by default.
    assert cfg.expose.field_strength_target is True
    # slice_mode defaults to 'all_slices': every foreground slice is training signal, and
    # 'central' would discard ~363/364 of each volume. Affordable because of the epoch
    # ORDER (VolumeBlockedSliceSampler), not because slices are cheap to fetch — see
    # docs/mrixfields_gpu_starvation_fix.rst.
    assert cfg.mrixfields.slice_mode == "all_slices"
    # The per-worker resident-volume budget, ~226 MB each. Floor is the widest
    # container's arity (multi_source: 4 sources + target = 5).
    assert cfg.mrixfields.max_resident_volumes == 6


def test_max_resident_volumes_rejects_a_budget_below_the_container_arity() -> None:
    # ge=2 at the schema; the dataset and sampler raise for the policy-specific floor.
    with pytest.raises(ValidationError):
        DataConfigSchema(
            dataset_type="mrixfields",
            batch_size=2,
            mrixfields={"max_resident_volumes": 1},
        )


def test_mrixfields_slice_mode_accepts_all_three_modes() -> None:
    for mode in ("central", "all_slices", "volume"):
        cfg = DataConfigSchema(
            dataset_type="mrixfields",
            batch_size=2,
            mrixfields={"slice_mode": mode},
        )
        assert cfg.mrixfields.slice_mode == mode


def test_mrixfields_slice_mode_rejects_unknown() -> None:
    # ValidationError, not bare Exception: `pytest.raises(Exception)` also passes when
    # the constructor raises for an unrelated reason (a typo'd field name, a moved
    # import), so it would keep this test green while the Literal it exists to pin was
    # gone.
    with pytest.raises(ValidationError):
        DataConfigSchema(
            dataset_type="mrixfields",
            batch_size=2,
            mrixfields={"slice_mode": "every_other_slice"},
        )


def test_mrixfields_pairing_policy_validates() -> None:
    # Narrow type + `match=`, for the reason line 48 already gives one test up:
    # the flat spelling was promoted to `raise` (2026-08-18), so
    # `raises(Exception)` here would catch the RENAME and pass while pinning
    # nothing about the Literal it exists to guard.
    with pytest.raises(ValidationError, match="Input should be 'fixed_target'"):
        DataConfigSchema(
            dataset_type="mrixfields",
            batch_size=2,
            mrixfields={"pairing_policy": "not_a_policy"},
        )


def test_fixed_target_round_trips() -> None:
    cfg = DataConfigSchema(
        dataset_type="mrixfields",
        batch_size=2,
        mrixfields={"pairing_policy": "fixed_target", "target_field": 7.0},
        expose={"field_strength_target": True},
    )
    assert cfg.mrixfields.target_field == 7.0
    assert cfg.expose.field_strength_target is True


def test_prior_policy_accepted_without_target_field() -> None:
    # The 'prior' policy (train a field-conditioned prior on the unpaired retrospective
    # pool, validate recon on the paired set) must be a valid schema choice and must NOT
    # require mrixfields.target_field.
    cfg = DataConfigSchema(
        dataset_type="mrixfields",
        batch_size=2,
        mrixfields={"pairing_policy": "prior"},
    )
    assert cfg.mrixfields.pairing_policy == "prior"
    assert cfg.mrixfields.target_field is None


def test_image_undersampling_defaults_off() -> None:
    # The image→k-space bridge (exp_c1) must be opt-in: an arm that doesn't ask
    # for it gets fully-sampled images, not silently-aliased ones.
    assert (
        DataConfigSchema(dataset_type="mrixfields", batch_size=2).image_undersampling
        is False
    )


def test_image_undersampling_opt_in_accepted() -> None:
    cfg = DataConfigSchema(
        dataset_type="mrixfields",
        batch_size=2,
        mrixfields={"pairing_policy": "prior"},
        image_undersampling=True,
    )
    assert cfg.image_undersampling is True
