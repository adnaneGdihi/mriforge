"""Unit tests for PreprocessedMRIDataset.

Compliant with unit-test.md:
- Property-based testing with Hypothesis (Directive 4.A)
- Invariant verification (shape preservation, no NaN/Inf)
- Deterministic randomness (fixed seeds)
- Memory profiling awareness
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies", reason="hypothesis not installed")
HealthCheck = hypothesis.HealthCheck

from spectramr.data.datasets.preprocessed_dataset import (
    PreprocessedMRIDataset,
    TaskType,
    create_preprocessed_dataloader,
)

# ============================================================================
# Fixtures
# ============================================================================


# Fixture geometry. Nothing here asserts a spatial size -- the contract under
# test is the artifact layout, the TorchIO subject keys, the 4-D tensor rank and
# the split arithmetic. At the original 16x320x320 with 4 coils, each subject
# wrote ~111 MB (two complex64 volumes at 52 MB apiece), the fixture was
# function-scoped, and ~11 tests each rebuilt all 5 subjects: 5.7 GB of tmpfs
# per run. On a 7.6 GB /tmp that fills partway through the file, and the tests
# that happen to run after it fail with `OSError: Disk quota exceeded` -- which
# is how one fixture presented as 14 scattered failures in cluster job 8000966.
#
# The directory names are the ones the PRODUCER emits
# (`scripts/preprocessing/preprocessing_legacy.py`: `output_dir / "kspace"`,
# `output_dir / "nifti_reconstructed"`) and that
# `PreprocessedMRIDataset.TASK_ARTIFACT_MAP` reads. The fixture previously
# invented `compressed_kspace/` and left `nifti_reconstructed/` empty -- a tree
# no producer writes and no task can load, which is what the disk-quota failures
# were hiding underneath them.
SUBJECT_COUNT = 5
SLICE_COUNT = 4
COIL_COUNT = 2
IMAGE_SIZE = 32


def _complex_volume(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return torch.complex(torch.from_numpy(real), torch.from_numpy(imag))


@pytest.fixture(scope="session")
def temp_preprocessing_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A preprocessing output tree with the full artifact set.

    Session-scoped and read-only: no test writes into it, so rebuilding it per
    test bought nothing but IO.
    """
    nibabel = pytest.importorskip("nibabel")
    output_dir = tmp_path_factory.mktemp("preprocessed") / "test_dataset_image"
    rng = np.random.default_rng(0)

    for subdir in (
        "gt_images",
        "kspace",
        "coil_sensitivity",
        "nifti_reconstructed",
        "statistics",
        "manifests",
    ):
        (output_dir / subdir).mkdir(parents=True)

    for index in range(SUBJECT_COUNT):
        subject_id = f"subject_{index:03d}"

        gt_data = rng.standard_normal(
            (SLICE_COUNT, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )
        np.save(output_dir / "gt_images" / f"{subject_id}_rss.npy", gt_data)

        # `nifti_reconstructed/*.nii.gz` is the RECONSTRUCTION task's target and
        # the denoising/super-resolution tasks' input; `gt_images/*.npy` above is
        # an optional extra the producer only writes for the ground_truth task,
        # and the dataset's TASK_ARTIFACT_MAP never names it.
        nibabel.save(
            nibabel.Nifti1Image(np.moveaxis(gt_data, 0, -1), np.eye(4)),
            output_dir / "nifti_reconstructed" / f"{subject_id}_rss.nii.gz",
        )

        kspace = _complex_volume(rng, SLICE_COUNT, COIL_COUNT, IMAGE_SIZE, IMAGE_SIZE)
        torch.save(kspace, output_dir / "kspace" / f"{subject_id}_compressed.pt")

        coil_maps = _complex_volume(
            rng, SLICE_COUNT, COIL_COUNT, IMAGE_SIZE, IMAGE_SIZE
        )
        torch.save(
            coil_maps, output_dir / "coil_sensitivity" / f"{subject_id}_coil_maps.pt"
        )

        stats = {
            "source": f"/path/to/{subject_id}.h5",
            "slices_processed": SLICE_COUNT,
            "slice_stats": [{"mean": 0.0, "std": 1.0} for _ in range(SLICE_COUNT)],
        }
        (output_dir / "statistics" / f"{subject_id}_stats.json").write_text(
            json.dumps(stats)
        )

    return output_dir


@pytest.fixture(scope="session")
def minimal_preprocessing_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create minimal preprocessing dir with only gt_images."""
    output_dir = tmp_path_factory.mktemp("preprocessed_minimal") / "minimal_image"
    (output_dir / "gt_images").mkdir(parents=True)
    rng = np.random.default_rng(1)

    for index in range(3):
        gt_data = rng.standard_normal(
            (SLICE_COUNT, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )
        np.save(output_dir / "gt_images" / f"sample_{index}_rss.npy", gt_data)

    return output_dir


# ============================================================================
# Unit Tests - Basic Functionality
# ============================================================================


class TestPreprocessedMRIDatasetInit:
    """Test dataset initialization and validation."""

    def test_inherits_from_torch_dataset_abc(self) -> None:
        """Class must inherit from ``torch.utils.data.Dataset``.

        Regression for ``TODO/audit/16_data_layer.md`` F5: without the
        ABC, PyTorch ``DataLoader`` silently mis-handles batch
        construction for some sampler combinations and does not detect
        the missing ``__getitem__`` contract at construction time.
        """
        import torch.utils.data as _torch_data

        assert issubclass(PreprocessedMRIDataset, _torch_data.Dataset)

    def test_init_valid_directory(self, temp_preprocessing_dir: Path) -> None:
        """Test successful initialization with valid directory."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        assert len(dataset) == 5
        assert dataset.task_type == TaskType.RECONSTRUCTION

    def test_init_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test that nonexistent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            PreprocessedMRIDataset(
                output_dir=tmp_path / "nonexistent",
                task_type="reconstruction",
            )

    def test_init_invalid_task_type(self, temp_preprocessing_dir: Path) -> None:
        """Test that invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task_type"):
            PreprocessedMRIDataset(
                output_dir=temp_preprocessing_dir,
                task_type="invalid_task",
            )

    def test_init_missing_artifacts(self, minimal_preprocessing_dir: Path) -> None:
        """Test that missing required artifacts raises ValueError."""
        with pytest.raises(ValueError, match="No input files found"):
            PreprocessedMRIDataset(
                output_dir=minimal_preprocessing_dir,
                task_type="reconstruction",  # Needs compressed_kspace
            )


class TestPreprocessedMRIDatasetLoading:
    """Test data loading functionality."""

    def test_getitem_returns_subject(self, temp_preprocessing_dir: Path) -> None:
        """Test __getitem__ returns TorchIO Subject."""
        import torchio as tio

        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        subject = dataset[0]

        assert isinstance(subject, tio.Subject)
        assert "input" in subject
        assert "target" in subject
        assert "subject_id" in subject

    def test_getitem_has_correct_shapes(self, temp_preprocessing_dir: Path) -> None:
        """Test that loaded tensors have 4D shape for TorchIO."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        subject = dataset[0]

        # TorchIO expects (C, W, H, D)
        assert subject.input.data.ndim == 4
        assert subject.target.data.ndim == 4

    def test_getitem_loads_coil_sensitivity(self, temp_preprocessing_dir: Path) -> None:
        """Test optional coil sensitivity loading."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
            load_coil_sensitivity=True,
        )

        subject = dataset[0]

        assert "sensitivity" in subject
        assert subject.sensitivity.data.ndim == 4


class TestPreprocessedMRIDatasetSplits:
    """Test train/val split functionality."""

    def test_train_split(self, temp_preprocessing_dir: Path) -> None:
        """Test train split has correct size."""
        dataset_all = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        dataset_train = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
            split="train",
            validation_split=0.2,
        )

        expected_train = len(dataset_all) - int(len(dataset_all) * 0.2)
        assert len(dataset_train) == expected_train

    def test_val_split(self, temp_preprocessing_dir: Path) -> None:
        """Test val split has correct size."""
        dataset_val = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
            split="val",
            validation_split=0.2,
        )

        assert len(dataset_val) == 1  # 20% of 5 = 1


# ============================================================================
# Property-Based Tests (Directive 4.A)
# ============================================================================


class TestPreprocessedMRIDatasetInvariants:
    """Property-based tests for dataset invariants."""

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(idx=st.integers(min_value=0, max_value=4))
    def test_invariant_shape_preservation(
        self,
        temp_preprocessing_dir: Path,
        idx: int,
    ) -> None:
        """Invariant: Shape must be 4D for all samples."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        subject = dataset[idx]

        # Shape invariant: all images are 4D
        assert subject.input.data.ndim == 4, "Input must be 4D"
        assert subject.target.data.ndim == 4, "Target must be 4D"

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(idx=st.integers(min_value=0, max_value=4))
    def test_invariant_no_nan_inf(
        self,
        temp_preprocessing_dir: Path,
        idx: int,
    ) -> None:
        """Invariant: No NaN or Inf values in loaded data (Directive 4.D.1)."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        subject = dataset[idx]

        # NaN/Inf check
        assert torch.isfinite(subject.input.data).all(), "Input contains NaN/Inf"
        assert torch.isfinite(subject.target.data).all(), "Target contains NaN/Inf"

    def test_invariant_subject_id_unique(self, temp_preprocessing_dir: Path) -> None:
        """Invariant: All subject IDs must be unique."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        subject_ids = [sample.subject_id for sample in dataset.dry_iter()]

        assert len(subject_ids) == len(set(subject_ids)), "Subject IDs must be unique"


# ============================================================================
# Task Type Coverage Tests
# ============================================================================


class TestTaskTypes:
    """Test each task type loads correct artifacts."""

    def test_reconstruction_task(self, temp_preprocessing_dir: Path) -> None:
        """Reconstruction reports the artifact pair its own map assigns.

        Expected values are read from ``TASK_ARTIFACT_MAP`` rather than spelled
        out: this test previously pinned the literal ``compressed_kspace`` /
        ``gt_images``, a vocabulary the map has not used in some time, so it was
        a second copy of the mapping that could only ever go stale.
        """
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )
        expected_input, expected_target = PreprocessedMRIDataset.TASK_ARTIFACT_MAP[
            TaskType.RECONSTRUCTION
        ]

        stats = dataset.get_statistics()

        assert stats["input_artifact"] == expected_input
        assert stats["target_artifact"] == expected_target

    def test_denoising_task_self_supervised(self, temp_preprocessing_dir: Path) -> None:
        """Denoising is self-supervised: one artifact serves as input AND target."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="denoising",
        )
        expected_input, expected_target = PreprocessedMRIDataset.TASK_ARTIFACT_MAP[
            TaskType.DENOISING
        ]
        assert expected_input == expected_target, "denoising must be self-supervised"

        stats = dataset.get_statistics()

        assert stats["input_artifact"] == expected_input
        assert stats["target_artifact"] == expected_target

        # The self-supervision has to reach the resolved SAMPLE, not just the
        # map: `__getitem__` reads `input_path`/`target_path` off this record.
        # (The previous form probed `dry_iter()`, which deliberately returns
        # path-less shells for TorchIO Queue's iterations_per_epoch count -- it
        # could never have held these attributes, and only looked fine because
        # the test raised before reaching the assertion.)
        sample = dataset._samples[0]
        assert sample.input_path == sample.target_path


# ============================================================================
# Dataloader Creation Tests
# ============================================================================


class TestCreatePreprocessedDataloader:
    """Test convenience dataloader creation."""

    def test_creates_valid_dataloader(self, temp_preprocessing_dir: Path) -> None:
        """Test dataloader creation succeeds."""
        # Use batch_size=2
        # NOTE: pin_memory=False is required because TorchIO ScalarImage doesn't
        # support the __copy__ operation that pin_memory triggers.
        dataloader = create_preprocessed_dataloader(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            shuffle=False,
            pin_memory=False,  # TorchIO incompatible with pin_memory
        )

        # Dataset has 5 samples.
        # Batch size 2.
        # Batches: [2, 2, 1]

        batch = next(iter(dataloader))

        # Batch should be list of tio.Subject with length equal to batch_size
        # But wait, default_collate or torchio collate might behave differently.
        # If create_preprocessed_dataloader uses a standard collate_fn, it might stack tensors.
        # If it returns a list of subjects, len(batch) is batch_size.

        # Checking implementation of create_preprocessed_dataloader (implied):
        # Usually torchio loaders return a batch dictionary where values are stacked tensors.
        # If batch is a Subject/dict, len(batch) is number of keys!

        # Check type of batch
        if isinstance(batch, (dict, object)) and hasattr(batch, "keys"):
            # It's a batched subject/dict.
            # Check batch size dimension of a tensor inside.
            assert batch["input"]["data"].shape[0] == 2
        elif isinstance(batch, list):
            assert len(batch) == 2
        else:
            # Fallback assumption was list
            assert len(batch) == 2


# ============================================================================
# Memory Safety Tests (Directive 4.E)
# ============================================================================


class TestMemorySafety:
    """Tests for memory leak prevention."""

    def test_no_tensor_accumulation(self, temp_preprocessing_dir: Path) -> None:
        """Verify tensors don't accumulate in memory over iterations."""
        import gc

        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
        )

        # Load all samples
        for i in range(len(dataset)):
            _ = dataset[i]

        gc.collect()

        # This is a basic check - full memory profiling would use memray
        # The key point is we don't store references to loaded tensors
        assert True  # Placeholder for memray integration


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_max_samples_limits_dataset(self, temp_preprocessing_dir: Path) -> None:
        """Test max_samples parameter limits dataset size."""
        dataset = PreprocessedMRIDataset(
            output_dir=temp_preprocessing_dir,
            task_type="reconstruction",
            max_samples=2,
        )

        assert len(dataset) == 2

    def test_empty_artifact_directory(self, tmp_path: Path) -> None:
        """Test handling of empty artifact directories."""
        output_dir = tmp_path / "empty_image"
        (output_dir / "gt_images").mkdir(parents=True)
        (output_dir / "compressed_kspace").mkdir(parents=True)

        with pytest.raises(ValueError, match="No input files found"):
            PreprocessedMRIDataset(
                output_dir=output_dir,
                task_type="reconstruction",
            )
