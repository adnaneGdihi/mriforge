import pytest
from pydantic import ValidationError

from mriforge.config.schemas.data import (
    CachingPolicy,
    DataConfigSchema,
    DatasetSourceSchema,
    DataSplitConfigSchema,
    ManifestRoleConfigSchema,
    PriorLoadingConfigSchema,
)


class TestDataConfigSchema:
    def test_defaults(self):
        schema = DataConfigSchema()
        # Default is "kspace"
        assert schema.dataset_type == "kspace"
        # `data_root` folded to `source.root` in the block decomposition. Ported
        # AND strengthened: `source.root` is a non-optional str defaulting to
        # "./data", so asserting only the default would pass without the fold
        # ever running -- the vacuous shape that turned 1282 red tests into 1282
        # that checked nothing. The second line is what actually exercises it.
        assert schema.source.root == "./data"
        assert DataConfigSchema(data_root="/tmp/legacy").source.root == "/tmp/legacy"
        assert schema.loader.batch_size == 4

    def test_slice_2d_defaults_off_and_accepts_true(self):
        # 2026-06-22: per-slice 2D sampling knob for volumetric paired NIfTI
        # (the stage-1 LDM VAEs). Backward-compatible default = off.
        assert DataConfigSchema().sampling.enable_slice_2d is False
        assert (
            DataConfigSchema(sampling={"enable_slice_2d": True}).sampling.enable_slice_2d
            is True
        )

    def test_custom_values(self):
        schema = DataConfigSchema(
            dataset_type="fastmri_kspace",
            data_root="/tmp/data",
            batch_size=16,
        )
        # "fastmri_kspace" alias gets normalized to "kspace" when explicitly set
        assert schema.dataset_type == "kspace"
        assert schema.source.root == "/tmp/data"
        assert schema.loader.batch_size == 16

    def test_validation_dataset_type(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(dataset_type="invalid_type")

    def test_validation_batch_size(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(batch_size=0)

    def test_manifest_roles(self):
        roles = {
            "inputs": [{"manifest": "in.pkl", "key": "in"}],
            "targets": [{"manifest": "out.pkl", "key": "out"}],
        }
        schema = DataConfigSchema(manifest_roles=roles)
        assert schema.manifest_roles.inputs[0]["manifest"] == "in.pkl"

    def test_subject_caps_default_none(self):
        schema = DataConfigSchema()
        assert schema.split.max_train_subjects is None
        assert schema.split.max_val_subjects is None

    def test_subject_caps_accept_positive(self):
        schema = DataConfigSchema(
            split={"max_train_subjects": 4, "max_val_subjects": 2}
        )
        assert schema.split.max_train_subjects == 4
        assert schema.split.max_val_subjects == 2

    def test_subject_caps_reject_non_positive(self):
        """Declared canonically: `data.max_train_subjects` now RAISES, and that
        error is also a `ValidationError`, so the legacy spelling would satisfy
        `pytest.raises` without ever reaching the positivity constraint."""
        with pytest.raises(ValidationError):
            DataConfigSchema(split={"max_train_subjects": 0})
        with pytest.raises(ValidationError):
            DataConfigSchema(split={"max_val_subjects": -1})


class TestDatasetSourceSchema:
    def test_defaults(self):
        schema = DatasetSourceSchema()
        assert schema.name == "default"
        assert schema.split == "train"

    def test_validation(self):
        with pytest.raises(ValidationError):
            DatasetSourceSchema(weight=0)

    def test_validate_variant_rejects_unknown(self):
        """Regression WS1-schemas-data-01: an unrecognised variant must RAISE,
        not silently pass through (NN#3). The old code had a no-op ``pass`` with
        a 'warn but don't fail' comment, letting typos reach dataset
        instantiation and fail with an opaque downstream error.
        """
        with pytest.raises(ValidationError):
            DatasetSourceSchema(variant="bogus_variant_xyz")

    def test_validate_variant_accepts_canonical_and_alias(self):
        """A known variant and a known alias still validate."""
        assert DatasetSourceSchema(variant="2d_slices").variant == "2d_slices"
        # '2d' is an alias mapped to the canonical '2d_slices'.
        assert DatasetSourceSchema(variant="2d").variant == "2d_slices"


class TestCachingPolicy:
    def test_defaults(self):
        schema = CachingPolicy()
        assert schema.strategy == "none"


class TestPriorLoadingConfigSchema:
    def test_defaults(self):
        schema = PriorLoadingConfigSchema()
        assert schema.enabled is False


class TestManifestRoleConfigSchema:
    def test_defaults(self):
        schema = ManifestRoleConfigSchema()
        assert len(schema.inputs) == 1
        assert schema.inputs[0]["key"] == "input"


class TestQueueLengthLoadTimeRejection:
    """``queue_length <= 0`` must be rejected at config-LOAD time (2026-06 audit).

    The runtime auto-correct it replaced silently rewrote 0 to a "sane"
    default with only a warning (pitfall #9/#10) — guarding a config state
    that no longer exists (the 7 ``experiment_11*`` arms that set
    ``queue_length: 0`` have all been migrated; the v6.1 sampler schema
    already enforces ``ge=1``). The legacy flat field must match.
    """

    def test_queue_length_zero_rejected(self):
        with pytest.raises(ValidationError, match="queue_length"):
            DataConfigSchema(queue_length=0)

    def test_queue_length_negative_rejected(self):
        with pytest.raises(ValidationError, match="queue_length"):
            DataConfigSchema(queue_length=-1)

    def test_queue_length_positive_accepted(self):
        assert DataConfigSchema(queue_length=1).sampling.queue_length == 1


# ---------------------------------------------------------------------------
# Leave-one-SUBJECT-out (2026-07-26)
# ---------------------------------------------------------------------------
def test_loso_subject_is_distinct_from_the_site_based_loso() -> None:
    """The two spellings are different designs and the collision is dangerous:
    'loso' holds out a SITE and needs a site tag; 'loso_subject' holds out a
    SUBJECT, which is what a 10-subject cohort needs."""
    from mriforge.config.schemas.data import DataConfigSchema

    assert (
        DataConfigSchema(
            dataset_type="nifti_paired",
            split_strategy="loso_subject",
            split={"loso_fold": 0},
        ).split.type
        == "loso_subject"
    )
    # the site variant still demands its own companion field
    with pytest.raises(ValidationError, match="holdout_site"):
        DataConfigSchema(dataset_type="nifti_paired", split_strategy="loso")


def test_loso_subject_requires_exactly_one_selector() -> None:
    """Both would disagree; neither leaves the held-out subject undefined and
    the run would silently validate on training subjects."""
    from mriforge.config.schemas.data import DataConfigSchema

    with pytest.raises(ValidationError, match="EXACTLY ONE"):
        DataConfigSchema(dataset_type="nifti_paired", split_strategy="loso_subject")
    with pytest.raises(ValidationError, match="EXACTLY ONE"):
        DataConfigSchema(
            dataset_type="nifti_paired",
            split_strategy="loso_subject",
            split={"loso_fold": 0, "holdout_subject": "0011"},
        )


def test_loso_selectors_are_inert_without_the_strategy() -> None:
    """Declaring a fold without asking for the strategy must not silently
    change the split — the knob is read only under loso_subject."""
    from mriforge.config.schemas.data import DataConfigSchema

    cfg = DataConfigSchema(dataset_type="nifti_paired", split={"loso_fold": 3})
    assert cfg.split.type == "auto" and cfg.split.loso_fold == 3


# --------------------------------------------------------------------------- #
# Phase 9b: the `data.split` sub-block
# --------------------------------------------------------------------------- #
class TestDataSplitSubBlock:
    """`split:` answers one question: what decides train vs validation.

    The strategy now sits with the companions that make it valid, which is what
    ``validate_split_strategy`` has always checked -- previously across eight
    fields scattered through an 85-scalar wall.
    """

    def test_every_legacy_spelling_folds_into_the_sub_block(self) -> None:
        """Totality over the rename table, not a sample.

        Reads the fold records rather than a hand-written list, so a record
        added later without a home is caught here instead of by an arm.
        """
        from mriforge.config.schemas.data import DataConfigSchema
        from mriforge.config.schemas.renames import RENAMES

        moved = {
            legacy: rec.canonical
            for legacy, rec in RENAMES.items()
            # Posture-independent: a record that has been PROMOTED to
            # `raise` still points at `data.split.*` and still has to point
            # somewhere real. Filtering on `fold` made this count shrink as
            # the ratchet advanced, which is the opposite of a totality claim.
            if rec.canonical.startswith("data.split.")
        }
        assert len(moved) == 8, f"expected 8 split records, got {sorted(moved)}"
        for legacy, canonical in moved.items():
            leaf = canonical.split(".")[-1]
            assert leaf in DataSplitConfigSchema.model_fields, (
                f"{legacy} folds to {canonical}, which is not a field on the "
                "sub-block -- the record points at nothing"
            )
            assert legacy.split(".")[-1] not in DataConfigSchema.model_fields, (
                f"{legacy} still exists FLAT as well; two spellings of one knob "
                "is the problem this move exists to remove"
            )

    def test_the_legacy_document_still_loads_into_the_new_home(self) -> None:
        from mriforge.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(
            dataset_type="nifti_paired",
            split_strategy="loso_subject",  # legacy, still folds
            validation_split=0.25,  # legacy, still folds, AND renamed
            split={"loso_fold": 2, "max_train_subjects": 4},
        )
        assert cfg.split.type == "loso_subject"
        assert cfg.split.validation_fraction == 0.25
        # Declared canonically because `data.loso_fold` and
        # `data.max_train_subjects` were promoted to `raise` once their corpus
        # count hit zero. They stay in the assertions -- this test also pins
        # that a fold MERGES into a partially-authored `split:` rather than
        # replacing it.
        assert cfg.split.loso_fold == 2
        assert cfg.split.max_train_subjects == 4

    def test_the_canonical_document_loads(self) -> None:
        from mriforge.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(
            dataset_type="nifti_paired",
            split={"type": "loso", "holdout_site": "siteA", "validation_fraction": 0.2},
        )
        assert cfg.split.type == "loso"
        assert cfg.split.holdout_site == "siteA"
        assert cfg.split.validation_fraction == 0.2

    def test_the_validator_still_fires_through_both_spellings(self) -> None:
        """The move must not weaken the companion-field checks.

        These are the guards that stop a run silently validating on training
        subjects, so they matter more than the grouping does.
        """
        from mriforge.config.schemas.data import DataConfigSchema

        for kwargs in (
            {"split_strategy": "loso"},  # legacy spelling
            {"split": {"type": "loso"}},  # canonical
        ):
            with pytest.raises(ValidationError, match="holdout_site"):
                DataConfigSchema(dataset_type="nifti_paired", **kwargs)

        for kwargs in (
            {"split_strategy": "loso_subject"},
            {"split": {"type": "loso_subject"}},
            {"split": {"type": "loso_subject", "loso_fold": 0, "holdout_subject": "x"}},
        ):
            with pytest.raises(ValidationError, match="EXACTLY ONE"):
                DataConfigSchema(dataset_type="nifti_paired", **kwargs)

    def test_test_split_deliberately_stays_flat(self) -> None:
        """The asymmetry is the point -- do not "tidy" this into the sub-block.

        Nothing reads ``test_split``: its only consumer,
        ``scripts/evaluation/run_test_inference.py``, raises on three attribute
        reads before it gets there (issue #665). Moving it into ``split:``
        beside the fields that DO work would advertise a held-out test set that
        does not exist -- the same call made for ``use_async_dataloader`` and
        ``optimization.num_steps``. It stays flat and visibly odd until #665
        decides to wire or delete it.
        """
        from mriforge.config.schemas.data import DataConfigSchema
        from mriforge.config.schemas.renames import RENAMES

        assert "test_split" in DataConfigSchema.model_fields
        assert "test_fraction" not in DataSplitConfigSchema.model_fields
        assert "data.test_split" not in RENAMES, (
            "test_split acquired a rename record -- if it is being wired, "
            "close #665 and delete this test; if not, it must not move"
        )

    def test_the_sub_block_forbids_unknown_keys(self) -> None:
        """Born strict. A typo inside `split:` must not be silently ignored."""
        from mriforge.config.schemas.data import DataConfigSchema

        with pytest.raises(ValidationError):
            DataConfigSchema(
                dataset_type="nifti_paired", split={"validaton_fraction": 0.2}
            )
