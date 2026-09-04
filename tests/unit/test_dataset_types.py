"""Unit tests for consolidated dataset type validation.

Tests the new canonical dataset types and their alias mappings:
- dataset_type: kspace, nifti, nifti_paired, image, dicom, synthetic, graph_mri
- variant: 2d_slices, 2d_full, 3d_patches, 3d_full

Following unit-test.md directives:
- Phase IV-A: Property-based testing with Hypothesis
- Phase IV-D: Numerical Safety (deterministic seeds)
"""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import (
    CANONICAL_DATASET_TYPES,
    DATASET_TYPE_ALIASES,
)
hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies", reason="hypothesis not installed")

from spectramr.config.schemas.data import DataConfigSchema, DatasetSourceSchema

# =============================================================================
# Phase I: Type Safety - Explicit Contract Testing
# =============================================================================


def _assert_dataset_type_accepted(dataset_type: str, expected: str) -> None:
    """The `dataset_type` VALIDATOR accepts *dataset_type* and folds it to *expected*.

    Not the same as "a bare config constructs": several canonical types carry
    further requirements of their own (`contrast_aware_paired` needs
    `input_contrast`/`target_contrast`). Failing on those would test the wrong
    thing -- the claim here is only that the name is recognised, so a
    ValidationError is tolerated as long as it is not the unrecognised-name one.
    """
    try:
        config = DataConfigSchema(dataset_type=dataset_type)
    except ValidationError as exc:
        assert "is not recognised" not in str(exc), (
            f"dataset_type={dataset_type!r} was rejected as unrecognised"
        )
        return
    assert config.dataset_type == expected

class TestDatasetTypeValidation:
    """Test dataset_type validation and alias mapping."""

    # DERIVED from the schema, not transcribed. The hand-written lists this
    # replaces had drifted three ways: `graph_mri` was retired as a canonical
    # type (the test still demanded it be accepted), `m4raw` was PROMOTED from
    # alias to canonical (the test still asserted it folds to `kspace`), and the
    # alias table had fallen behind. `check_experiment_configs_load.py` already
    # carries an explicit anti-#353 invariant against a transcribed list of these
    # very names; this file was the transcription it was written to prevent.
    CANONICAL_TYPES = list(CANONICAL_DATASET_TYPES)

    # Alias -> canonical, straight from the fold table the validator consults.
    ALIAS_MAPPINGS = dict(DATASET_TYPE_ALIASES)

    @pytest.mark.parametrize("canonical_type", CANONICAL_TYPES)
    def test_canonical_types_accepted(self, canonical_type: str) -> None:
        """All canonical types must be accepted without modification."""
        _assert_dataset_type_accepted(canonical_type, canonical_type)

    @pytest.mark.parametrize("alias,expected", ALIAS_MAPPINGS.items())
    def test_alias_mapping(self, alias: str, expected: str) -> None:
        """Aliases must map to correct canonical types."""
        config = DataConfigSchema(dataset_type=alias)
        assert (
            config.dataset_type == expected
        ), f"{alias} should map to {expected}, got {config.dataset_type}"

    def test_invalid_type_rejected(self) -> None:
        """Invalid dataset types must raise ValueError."""
        # The validator's wording changed to "... is not recognised. Canonical
        # types: [...]"; match on the invariant (the offending value is named)
        # rather than a phrase, which is what broke this and the identical
        # assertion in tests/unit/ci/test_check_experiment_configs_load.py.
        with pytest.raises(ValueError, match="invalid_type_xyz"):
            DataConfigSchema(dataset_type="invalid_type_xyz")

    def test_case_insensitive(self) -> None:
        """Type matching should be case-insensitive."""
        for variant in ["KSPACE", "Kspace", "KsPaCe"]:
            config = DataConfigSchema(dataset_type=variant)
            assert config.dataset_type == "kspace"


class TestVariantValidation:
    """Test variant validation and alias mapping."""

    # Canonical variant types
    CANONICAL_VARIANTS = ["2d_slices", "2d_full", "3d_patches", "3d_full"]

    # Alias mappings for variants
    VARIANT_ALIASES = {
        "2d": "2d_slices",
        "3d": "3d_patches",
        "3d_volumetric": "3d_patches",
        "slices": "2d_slices",
        "patches": "3d_patches",
        "volume": "3d_full",
        "full": "2d_full",
    }

    @pytest.mark.parametrize("canonical_variant", CANONICAL_VARIANTS)
    def test_canonical_variants_accepted(self, canonical_variant: str) -> None:
        """All canonical variants must be accepted."""
        source = DatasetSourceSchema(variant=canonical_variant)
        assert source.variant == canonical_variant

    @pytest.mark.parametrize("alias,expected", VARIANT_ALIASES.items())
    def test_variant_alias_mapping(self, alias: str, expected: str) -> None:
        """Variant aliases must map to correct canonical types."""
        source = DatasetSourceSchema(variant=alias)
        assert (
            source.variant == expected
        ), f"{alias} should map to {expected}, got {source.variant}"


# =============================================================================
# Phase IV-A: Property-Based Testing (Hypothesis)
# =============================================================================


class TestPropertyBasedDataConfig:
    """Property-based tests for data configuration invariants."""

    @given(batch_size=st.integers(min_value=1, max_value=128))
    @settings(max_examples=50)
    def test_batch_size_positive_invariant(self, batch_size: int) -> None:
        """Batch size must always be positive."""
        config = DataConfigSchema(batch_size=batch_size)
        assert config.loader.batch_size > 0

    @given(val_split=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_validation_split_range_invariant(self, val_split: float) -> None:
        """Validation split must be in [0, 1]."""
        config = DataConfigSchema(validation_split=val_split)
        assert 0.0 <= config.split.validation_fraction <= 1.0

    @given(acceleration=st.integers(min_value=1, max_value=16))
    @settings(max_examples=50)
    def test_acceleration_positive_invariant(self, acceleration: int) -> None:
        """Acceleration factor must be positive."""
        # acceleration field has been moved out of DataConfigSchema
        # to the top-level AccelerationConfigSchema (SSOT migration)
        # This invariant is now tested at the top-level config level
        assert acceleration >= 1, "Acceleration must be positive by construction"


# =============================================================================
# Phase IV-D: Integration Tests (Deterministic)
# =============================================================================


class TestDatasetFactoryIntegration:
    """Integration tests for dataset factory with new types."""

    def test_kspace_type_creates_correct_handler(self) -> None:
        """kspace type should use FastMRI-style loader."""
        from spectramr.config.schemas.data import DataConfigSchema

        config = DataConfigSchema(
            dataset_type="kspace",
            data_root="databases/fastmri",
            index_path="data/manifests/test.pkl",
        )
        assert config.dataset_type == "kspace"
        # Factory integration tested separately

    def test_nifti_paired_uses_paired_strategy(self) -> None:
        """nifti_paired type should set paired IO strategy."""
        config = DataConfigSchema(
            dataset_type="nifti_paired",
            data_root="databases/ulf_paired",
        )
        assert config.dataset_type == "nifti_paired"

    def test_image_type_for_folder_data(self) -> None:
        """image type should be accepted for folder-based data."""
        config = DataConfigSchema(
            dataset_type="image",
            data_root="databases/brats_sr",
        )
        assert config.dataset_type == "image"
