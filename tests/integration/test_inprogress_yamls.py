"""
Tests for checking in-progress experiment YAML data components, including dataset
existence, patch sizes, and schema compliance.

Environment-aware: Tests that require physical data files (cluster-side MRI datasets)
are automatically **skipped** when the data is unavailable locally.  Only genuine
schema violations, configuration bugs, and tensor-shape mismatches are reported as
failures.
"""

import os
from pathlib import Path

import pytest

# Disable WANDB to prevent teardown I/O errors during tests
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

# Important: Always load via TrainingSettings to enforce Pydantic v6 validation
from spectramr.config.schemas.data import DataSourceConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)
from tests.utils.corpus import tracked_yamls

# `data.data_root` folded to `data.source.root`, which is a non-optional str
# carrying a schema default -- so "the arm declared a root" cannot be spelled
# `is not None` (that is vacuously true) and must instead exclude the default.
# Read it off the schema so changing the default cannot silently rot this test.
_DATA_SOURCE_ROOT_DEFAULT = DataSourceConfigSchema.model_fields["root"].default

# A root is not the only way an arm says where its data is: 22 of the 642
# inprogress arms carry no root at all and locate every sample through a
# manifest index instead. Requiring a root would fail all of them for a
# completeness they already have by another route.
_DATA_LOCATOR_FIELDS = ("index_path", "paired_manifest_path", "preprocessing_dir")

# Dataset types whose samples are generated on the fly, for which no locator is
# meaningful. Verified against `dataset_instantiator`: both
# `_create_synthetic_dataset` and `_create_pde_synthetic_dataset` build tensors
# in-process and never read `source.root`.
_GENERATED_DATASET_TYPES = frozenset({"synthetic", "pde_synthetic"})


def _declared_data_locator(data_cfg) -> str | None:
    """Return how *data_cfg* says where its data is, or None if it never does.

    Either a ``source.root`` that is not the schema default, or any manifest
    path. Reads the canonical paths directly so a future fold surfaces as an
    AttributeError naming the moved field rather than a false "unspecified".
    """
    if data_cfg.source.root != _DATA_SOURCE_ROOT_DEFAULT:
        return data_cfg.source.root
    for field in _DATA_LOCATOR_FIELDS:
        value = getattr(data_cfg.source, field)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# File extensions recognised as actual MRI/image data
# ---------------------------------------------------------------------------
_DATA_EXTENSIONS = frozenset({
    ".h5", ".hdf5",          # FastMRI / multicoil k-space
    ".nii", ".nii.gz",       # NIfTI volumes
    ".npy", ".npz",          # NumPy arrays
    ".pt", ".pth",           # PyTorch tensors
    ".png", ".jpg", ".tif",  # Image slices
    ".dcm",                  # DICOM
    ".mha", ".mhd",          # MetaImage
    ".parquet",              # Manifest / index tables
    ".pkl",                  # Pickle manifests
})

# Patterns in error messages that indicate *missing data* rather than a code bug.
_DATA_MISSING_PATTERNS = (
    "FileNotFoundError",
    "No such file or no access",
    "No such file or directory",
    "num_samples should be a positive integer",
    "num_samples=0",
    "Subjects list is empty",
    "No H5 files found",
    "No .nii files found",
    "No NIfTI files found",
    "No files found",
    "No valid files found",
    "Could not find any data",
    "does not exist",
    "not found",
    "empty dataset",
    "Index file not found",
    "Data root not found",
    "Data path not found",
    "requires either 'index_path'",
)


def _data_root_has_files(data_root: str) -> bool:
    """Walk *data_root* and return True if at least one data file exists.

    Limits scan to 500 entries to avoid hanging on huge cluster mounts.
    """
    if not os.path.isdir(data_root):
        return False

    count = 0
    for root, _dirs, filenames in os.walk(data_root):
        for fname in filenames:
            suffix = Path(fname).suffix.lower()
            # Handle double-extensions like .nii.gz
            if suffix == ".gz" and fname.lower().endswith(".nii.gz"):
                suffix = ".nii.gz"
            if suffix in _DATA_EXTENSIONS:
                return True
        count += 1
        if count > 500:
            break
    return False


def _is_data_missing_error(exc: Exception) -> bool:
    """Return True if *exc* is caused by absent/empty data, not a code bug."""
    msg = repr(exc)
    return any(pat in msg for pat in _DATA_MISSING_PATTERNS)


def get_inprogress_yamls():
    """Every **tracked** in-progress arm, sorted.

    Tracked rather than globbed, and this module is a reason the helper exists:
    the raw ``glob`` this replaces also swept up untracked scratch left on the
    cluster's working tree, so the subject differed per machine. Two of cluster
    job ``8012333``'s failures here are the ``reverse_sampler_ab`` pair, which
    exists in no git history on any branch -- unfixable from a clean checkout
    because there is nothing to fix. See :mod:`tests.utils.corpus`.

    The old form was cwd-relative too (``os.path.exists("experiments/...")``),
    so it enumerated *nothing* when pytest ran from anywhere but the repo root
    -- silently, as a zero-length parametrisation. ``tracked_yamls`` resolves
    against the repo root instead.
    """
    base_dirs = ["experiments/inprogress", "experiments/training/inprogress"]
    return sorted(y for d in base_dirs for y in tracked_yamls(d))

YAML_FILES = get_inprogress_yamls()


@pytest.mark.parametrize("yaml_path", YAML_FILES, ids=lambda x: Path(x).stem)
def test_experiment_yaml_schema_validation(yaml_path):
    """Verify that every in-progress YAML parses against the Pydantic v6 schema.

    This test is purely about configuration correctness and does NOT require
    physical data.  It should never be skipped due to missing datasets.
    """
    try:
        config = TrainingSettings.from_yaml(yaml_path)
    except Exception as e:
        pytest.fail(f"YAML Schema Validation Failed for {yaml_path}:\n{e}")

    data_cfg = config.data
    assert data_cfg is not None, "Configuration is missing 'data:' component block."

    # --- Static field checks (no I/O) ---
    # Patch Size. `data.patch_size` folded to `data.sampling.patch_size`, and
    # the sentinel this replaces made the whole block below unreachable: the
    # getattr returned None for every arm, so `if patch_size is not None` was
    # never entered and these assertions were inert corpus-wide. Unlike the
    # data_root miss, this one never showed up as a failure.
    #
    # Only positivity survives the port. The field is
    # `tuple[int, int, int] | tuple[int, int]` and non-optional, so pydantic
    # already rejects a non-tuple and a wrong arity -- re-asserting those here
    # would test pydantic, not the corpus. It does NOT reject a non-positive
    # extent: `(0, 320)` and `(-8, 320, 1)` both validate (probed 2026-08-06).
    patch_size = data_cfg.sampling.patch_size
    for idx, dim in enumerate(patch_size):
        assert dim > 0, f"patch_size[{idx}] must be a positive integer, got {dim}"

    # Batch / workers. Read the canonical path DIRECTLY, not through a
    # defaulted getattr: both are declared fields, so a miss means the schema
    # moved, and a sentinel turns that into a misleading "batch_size must be
    # > 0" instead of an AttributeError that names the real cause. The flat
    # spellings folded to `data.loader.*` in phase 9a.
    assert data_cfg.loader.batch_size > 0, "data.loader.batch_size must be > 0"
    assert data_cfg.loader.num_workers >= 0, "data.loader.num_workers must be >= 0"

    # Dataset type
    assert getattr(data_cfg, "dataset_type", None) is not None, \
        "data.dataset_type is required"

    # Data location declared. Same phase-9a fold as `loader.*` above: the flat
    # `data.data_root` no longer exists on the schema, so the old
    # `getattr(..., None)` sentinel reported all 642 arms as "not specified"
    # while every one of them resolved a location the loader was happy with.
    if data_cfg.dataset_type not in _GENERATED_DATASET_TYPES:
        assert _declared_data_locator(data_cfg) is not None, \
            f"this arm says nowhere where its data is: `data.source.root` is " \
            f"the schema default {_DATA_SOURCE_ROOT_DEFAULT!r} and no manifest " \
            f"path ({', '.join(_DATA_LOCATOR_FIELDS)}) is set"


# Builds loaders via the deprecated ConsolidatedDatasetFactory (legacy path);
# H3 added a DeprecationWarning there. This test checks data integrity, not the
# deprecation, so silence it locally.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("yaml_path", YAML_FILES, ids=lambda x: Path(x).stem)
def test_experiment_yaml_data_integrity(yaml_path):
    """Load a YAML, build a DataLoader, and fetch one batch.

    **Environment-aware**: This test is skipped when:
    - ``data_root`` does not exist locally
    - ``data_root`` exists but contains no usable data files
    - The DataLoader fails with a data-missing error (FileNotFoundError, etc.)

    Only *real* bugs (schema parse errors, tensor-shape mismatches, code
    exceptions) are reported as failures.
    """
    # 1. Parse config -------------------------------------------------------
    try:
        config = TrainingSettings.from_yaml(yaml_path)
    except Exception as e:
        pytest.fail(f"YAML Schema Validation Failed for {yaml_path}:\n{e}")

    data_cfg = config.data
    assert data_cfg is not None, "Configuration is missing 'data:' component block."

    # 2. Check physical data availability -----------------------------------
    # Generated datasets have no root to look at; the director builds their
    # samples in-process, so availability is not a question that applies.
    if data_cfg.dataset_type in _GENERATED_DATASET_TYPES:
        pytest.skip(
            f"dataset_type={data_cfg.dataset_type!r} generates its samples "
            f"in-process -- no data root to check."
        )

    assert _declared_data_locator(data_cfg) is not None, \
        f"this arm says nowhere where its data is: `data.source.root` is the " \
        f"schema default {_DATA_SOURCE_ROOT_DEFAULT!r} and no manifest path " \
        f"({', '.join(_DATA_LOCATOR_FIELDS)}) is set."

    data_root = data_cfg.source.root
    # Manifest-driven arms leave `root` at the default and name every sample
    # through an index, so there is no directory to walk. Fall through to the
    # loader build below, which resolves the manifest and whose data-missing
    # errors are already converted to a skip.
    if data_root != _DATA_SOURCE_ROOT_DEFAULT:
        if not os.path.exists(data_root):
            pytest.skip(
                f"Data root '{data_root}' does not exist locally. "
                f"Run on the cluster to verify."
            )

        assert os.path.isdir(data_root), \
            f"Data root '{data_root}' exists but is not a directory."

        if not _data_root_has_files(data_root):
            pytest.skip(
                f"Data root '{data_root}' exists but contains no recognised "
                f"data files ({', '.join(sorted(_DATA_EXTENSIONS)[:6])}, …). "
                f"Run on the cluster to verify."
            )

    # 3. Build DataLoader and fetch one batch --------------------------------
    try:
        train_loader, val_loader = DataPipelineDirector(
            BuilderContext(config=config)
        ).build_dataloaders()
    except Exception as e:
        if _is_data_missing_error(e):
            pytest.skip(
                f"DataLoader construction skipped — data unavailable: {e}"
            )
        raise  # Real code error → let pytest report it

    assert train_loader is not None, \
        "the director returned None for train_loader"
    assert val_loader is not None, \
        "the director returned None for val_loader"

    # Fetch a single batch
    # Keys that are known scalar/1D metadata (not spatial data tensors)
    _METADATA_KEYS = frozenset({
        "contrast_idx", "location", "index", "subject_id", "slice_idx",
        "file_id", "affine", "stem", "path", "label", "class_label",
    })

    try:
        for batch in train_loader:
            import torch
            assert isinstance(batch, dict), \
                "Dataset must return a dict of tensors/metadata"

            _found_tensor = False
            patch_size = getattr(data_cfg, "patch_size", None)

            for k, v in batch.items():
                tensor_val = None

                # Unwrap TorchIO structures
                if isinstance(v, torch.Tensor):
                    tensor_val = v
                elif isinstance(v, dict) and "data" in v \
                        and isinstance(v["data"], torch.Tensor):
                    tensor_val = v["data"]
                elif hasattr(v, "data") and isinstance(v.data, torch.Tensor):
                    tensor_val = v.data

                if tensor_val is not None:
                    _found_tensor = True

                    assert isinstance(tensor_val, torch.Tensor), \
                        f"Batch attr '{k}': expected Tensor, got {type(tensor_val)}"

                    # Skip strict shape checks for scalar/1D metadata tensors
                    if k in _METADATA_KEYS:
                        continue

                    # Batch-dimension sanity (only for multi-dim tensors).
                    # No AttributeError guard: swallowing one silently retired
                    # this whole assertion the moment `data.batch_size` folded
                    # to `data.loader.batch_size`.
                    if len(tensor_val.shape) >= 2:
                        expected_b = data_cfg.loader.batch_size
                        assert tensor_val.shape[0] <= expected_b, (
                            f"Batch dim mismatch for '{k}'. "
                            f"Expected <={expected_b}, got {tensor_val.shape[0]}. "
                            f"Shape: {tensor_val.shape}"
                        )
                        assert tensor_val.shape[0] > 0, \
                            "Batch dimension is empty (0)"

            assert _found_tensor, \
                "Batch dict contained no extractable PyTorch tensors!"

            # One batch is enough
            break

    except Exception as e:
        if _is_data_missing_error(e):
            pytest.skip(
                f"Batch iteration skipped — data unavailable: {e}"
            )
        import traceback
        pytest.fail(
            f"DataLoader batch failure for {yaml_path}: {repr(e)}\n\n"
            f"{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# ldm_ulf_to_hf cohort — schema-version currency
#
# The cohort was migrated 6.0 -> 6.1 on 2026-07-25. Before that every arm already
# used 4-6 keys that only exist in the v6.1 reference template
# (logging.debug_snapshots, training.{device,output_dir,task},
# physics.field_strength, data.collation) while declaring config_version: '1.0',
# so the declared version understated what the files actually used. Since
# TrainingSettings.from_yaml validates config_version and then DELETES it, that
# mislabel is invisible at runtime — it only surfaces in the provenance the run
# stamps (ResolvedExperimentContext.config_version). Nothing else would catch a
# silent drift back, or a half-migration where the two version fields disagree.
# ---------------------------------------------------------------------------

_LDM_COHORT_DIR = Path("experiments/inprogress/ldm_two_stage_ulf_to_hf")


def _ldm_cohort_yamls() -> list[Path]:
    return tracked_yamls(_LDM_COHORT_DIR, recursive=False)


@pytest.mark.skipif(not _LDM_COHORT_DIR.is_dir(), reason="ldm cohort not present")
def test_ldm_cohort_is_on_the_latest_accepted_schema_version():
    """Every cohort arm declares the canonical schema version.

    "Latest" is :data:`CANONICAL_CONFIG_VERSION`, read directly. It is emphatically
    *not* ``max(ACCEPTED_CONFIG_VERSIONS)``: that set is ``{CANONICAL} | LEGACY``,
    and legacy versions are by construction the **older** ones. The 5.x/6.x -> 1.0
    restart means numeric ordering over the accepted set points *backwards*, so the
    moment ``LEGACY_CONFIG_VERSIONS`` stops being empty a ``max()`` here would
    demand every arm bump onto the legacy version this migration exists to drain.
    """
    import yaml as _yaml

    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

    latest = CANONICAL_CONFIG_VERSION
    files = _ldm_cohort_yamls()
    assert files, f"no YAMLs found under {_LDM_COHORT_DIR}"

    # Parse once per file: the old comprehension re-read and re-parsed every
    # YAML a second time just to evaluate its own filter. (Ported from #1195.)
    declared = {p.name: _yaml.safe_load(p.read_text()).get("config_version") for p in files}
    stale = {name: version for name, version in declared.items() if version != latest}
    assert not stale, (
        f"cohort arms not on the latest schema version ({latest}): {stale}. "
        f"Bump config_version AND metadata.version together."
    )


def test_numeric_max_over_accepted_versions_would_invert():
    """Demonstrate *why* the cohort test reads the constant instead of ranking.

    Not a style preference: the 5.x/6.x -> 1.0 restart inverted numeric ordering,
    so a single legacy entry is enough to make ``max()`` outrank the canonical
    version. This asserts the inversion is real, so the guard below has teeth.
    """
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

    hypothetical_accepted = {CANONICAL_CONFIG_VERSION, "6.1"}
    numeric_max = max(hypothetical_accepted, key=lambda v: tuple(int(p) for p in v.split(".")))
    assert numeric_max != CANONICAL_CONFIG_VERSION, (
        f"numeric max over {sorted(hypothetical_accepted)} returned "
        f"{numeric_max!r}; if this ever equals the canonical version the "
        f"restart has been undone and this guard needs revisiting."
    )


def test_cohort_version_test_does_not_rank_the_accepted_set():
    """The cohort test must read ``CANONICAL_CONFIG_VERSION``, never rank a set.

    ``LEGACY_CONFIG_VERSIONS`` is empty today (PR #891), so a regression to
    ``max(ACCEPTED_CONFIG_VERSIONS)`` would still pass every assertion above --
    it only inverts once that frozenset is repopulated, which is the one reason
    it exists. Source inspection is therefore the only guard that can fire while
    the defect is still latent.
    """
    import inspect

    source = inspect.getsource(test_ldm_cohort_is_on_the_latest_accepted_schema_version)
    body = source.split('"""')[-1]  # drop the docstring, which names both spellings

    assert "CANONICAL_CONFIG_VERSION" in body, (
        "the cohort version test must derive 'latest' from CANONICAL_CONFIG_VERSION"
    )
    assert "max(" not in body, (
        "the cohort version test ranks a version set again -- "
        "ACCEPTED_CONFIG_VERSIONS holds LEGACY (older) versions, so ranking it "
        "returns the version the migration exists to drain. Read the constant."
    )


# --------------------------------------------------------------------------
# `metadata.version` is the AUTHOR's arm revision, never the schema tier
# --------------------------------------------------------------------------
#
# This replaces `test_ldm_cohort_metadata_version_matches_config_version`, which
# asserted the two must be EQUAL. That is the conflation, not the invariant, and
# the repo had already elected against it: `tests/unit/config/schemas/
# test_defaults_provider.py` fixed the provider default to None and asserts
# `metadata.version` must never be a config tier -- "even the *current* tier is
# wrong here". Two enforcers, opposite rules, one of them defence-in-depth for
# the losing side (non-negotiable 17), so the loser's enforcement is deleted
# rather than kept in sync.
#
# It was also RED, on `dev`, before this change touched anything: all six cohort
# arms sat at `config_version: '1.0'` with `metadata.version: '6.1'`. A test
# demanding the wrong thing had simply been left failing.
#
# What replaces it is strictly stronger: the corpus, not one cohort, and the
# elected rule rather than its negation.
#
#: The schema lineage the 2026-08-03 restart retired. Spelled out rather than
#: derived: `LEGACY_CONFIG_VERSIONS` is empty since PR #891 and
#: `ACCEPTED_CONFIG_VERSIONS` is `{"1.0"}`, so no constant in the codebase still
#: names 5.x/6.x -- a membership test against those sets is green on exactly the
#: values this guard exists to catch. (That was a real defect in the two unit
#: tests above, found by planting `"6.0"` and watching them pass.)
_RETIRED_TIER_LINEAGE = ("5.", "6.")

#: Deliberately NOT forbidden here, though the unit tests forbid it as a
#: *default*: `1.0` as a declared `metadata.version` is textually identical to a
#: genuine author revision 1.0, and 23 `inprogress/` arms carry it. Failing them
#: would be a guess about author intent, and this file cannot recover that intent.
#: The unit-test guards cover the mechanism that could re-introduce it wholesale.
_AMBIGUOUS_WITH_AN_AUTHOR_REVISION = frozenset({"1.0"})


@pytest.mark.parametrize("yaml_path", get_inprogress_yamls(), ids=lambda p: p.stem)
def test_metadata_version_is_not_a_retired_schema_tier(yaml_path):
    """No `inprogress/` arm carries a v5/v6 schema tier as its author revision.

    `metadata.version` versions the ARM (`experiment_12_image_cold_diffusion_v2
    .yaml` declares `'2.0'` and has `_v2` in its filename); `config_version`
    versions the SCHEMA. The v5/v6 -> 1.0 restart smeared the second into the
    first through a template, leaving 468 of 647 arms claiming an author revision
    of `6.0` or `6.1` that no author ever wrote.

    A ratchet, not a drain: the corpus was cleaned in the same change that added
    this, so what it guards is re-entry -- a new arm copied from an old template.
    """
    import yaml as _yaml

    data = _yaml.safe_load(yaml_path.read_text())
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict) or "version" not in metadata:
        return  # absent is the canonical state: "the author did not say"

    declared = str(metadata["version"])
    assert not declared.startswith(_RETIRED_TIER_LINEAGE), (
        f"{yaml_path.name}: metadata.version={declared!r} is a retired config "
        f"tier, not an author revision. Delete the key (absent == 'the author "
        f"did not say') or replace it with the arm's own revision. "
        f"config_version={data.get('config_version')!r} is where the schema tier "
        f"lives."
    )
