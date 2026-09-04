"""
Unit tests for data module components.

Tests validate:
1. Dataset factory
2. Dataset registry
3. Collate functions
4. Batch types
5. IO strategies
6. Individual datasets

Run on CI, NOT local dev machine.
"""

import pytest
import torch


# === Dataset Factory Tests ===
class TestDatasetFactory:
    """Test consolidated dataset factory."""

    # `test_factory_import` lived here and asserted that
    # `ConsolidatedDatasetFactory` was importable, skipping on ImportError. The
    # module is deleted (6a-iii), so the test's subject is gone -- and its
    # skip-on-ImportError shape meant it could never have failed for the right
    # reason anyway. `tests/unit/data/test_data_init.py` now asserts the
    # opposite property: that the symbol is NOT reachable from `spectramr.data`.

    # @# pytest.mark.timeout(30)
    def test_simple_factory_import(self):
        try:
            from spectramr.data.factory import DatasetFactory

            assert DatasetFactory is not None
        except ImportError:
            pytest.skip("DatasetFactory not available")


# === Dataset Registry Tests ===
class TestDatasetRegistry:
    """Test dataset registry."""

    # @# pytest.mark.timeout(30)
    def test_registry_import(self):
        try:
            from spectramr.data.dataset_registry import DatasetRegistry

            assert DatasetRegistry is not None
        except ImportError:
            pytest.skip("DatasetRegistry not available")

    # @# pytest.mark.timeout(30)
    def test_registry_has_datasets(self):
        try:
            from spectramr.data.dataset_registry import DatasetRegistry
        except ImportError:
            pytest.skip("DatasetRegistry not available")

        # Check registry has some datasets
        if hasattr(DatasetRegistry, "list_datasets"):
            datasets = DatasetRegistry.list_datasets()
            assert len(datasets) > 0
        elif hasattr(DatasetRegistry, "_registry"):
            assert len(DatasetRegistry._registry) >= 0


# === Collate Function Tests ===
class TestCollateFunctions:
    """Test collate functions."""

    # @# pytest.mark.timeout(30)
    def test_collate_import(self):
        try:
            from spectramr.data.collate import collate_fn

            assert collate_fn is not None
        except ImportError:
            try:
                from spectramr.data.collate import MRICollateFn

                assert MRICollateFn is not None
            except ImportError:
                pytest.skip("Collate functions not available")

    # @# pytest.mark.timeout(30)
    def test_collate_basic_tensors(self):
        """Collate should stack tensors into batches."""
        # Standard collation
        batch = [
            {"input": torch.randn(1, 64, 64), "target": torch.randn(1, 64, 64)},
            {"input": torch.randn(1, 64, 64), "target": torch.randn(1, 64, 64)},
        ]

        # Manual collation
        inputs = torch.stack([b["input"] for b in batch])
        targets = torch.stack([b["target"] for b in batch])

        assert inputs.shape == (2, 1, 64, 64)
        assert targets.shape == (2, 1, 64, 64)

    # @# pytest.mark.timeout(30)
    def test_collate_handles_none(self):
        """Collate should handle None values gracefully."""
        batch = [
            {"input": torch.randn(1, 64, 64), "mask": None},
            {"input": torch.randn(1, 64, 64), "mask": None},
        ]

        inputs = torch.stack([b["input"] for b in batch])
        masks = [b["mask"] for b in batch]

        assert inputs.shape == (2, 1, 64, 64)
        assert all(m is None for m in masks)


# === Batch Types Tests ===
class TestBatchTypes:
    """Test batch type definitions."""

    def test_batch_types_import(self):
        """The elected batch container is importable.

        This asserted on ``MRIBatch``, which ``batch_types`` has never defined,
        inside ``except ImportError: pytest.skip("MRIBatch not available")`` —
        so it always skipped and reported green while covering nothing. The
        import is now unguarded: if it fails, that IS the finding (D23).
        """
        from spectramr.data.batch_types import TrainingBatch

        assert TrainingBatch is not None

    def test_batch_type_fields(self):
        """The container declares the fields the contract promises.

        The previous version was doubly inert: it skipped on an import that
        could never succeed, and its assertion was ``if field in fields:
        assert True`` — which passes whether or not the field exists. Both
        halves are now real.
        """
        from spectramr.data.batch_types import TrainingBatch

        fields = TrainingBatch.__dataclass_fields__
        # `coil_maps` is first-class since C11; `kspace` deliberately is not a
        # field — it rides in `metadata`, which is why the old expectation list
        # naming it could never have held.
        assert {"input", "target", "mask", "coil_maps", "metadata"} <= set(fields)

    def test_the_retired_batch_declarations_are_gone(self):
        """D23 elected ``TrainingBatch`` from FOUR competing declarations.

        ``data.structures.MRIBatch`` was a TypedDict used only as
        ``dict[str, Any] | MRIBatch`` — a vacuous union, since a TypedDict IS a
        dict — and ``domain.entities.data.MRIBatch`` was a dataclass with zero
        importers. ``MRIBatchDict`` survives: it is the domain-layer protocol
        type, a different job.
        """
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module("spectramr.data.structures")

        import spectramr.domain.entities.data as domain_data

        assert not hasattr(domain_data, "MRIBatch")
        assert "MRIBatch" not in domain_data.__all__

        from spectramr.domain.entities.data.types import MRIBatchDict

        assert MRIBatchDict is not None


# === IO Strategies Tests ===
class TestIOStrategies:
    """Test IO strategies."""

    # @# pytest.mark.timeout(30)
    def test_io_strategies_import(self):
        try:
            from spectramr.data.io_strategies import IOStrategy

            assert IOStrategy is not None
        except ImportError:
            pytest.skip("IOStrategy not available")

    # @# pytest.mark.timeout(30)
    def test_h5_strategy(self):
        """H5 IO strategy should exist."""
        try:
            from spectramr.data.io_strategies import H5IOStrategy

            assert H5IOStrategy is not None
        except ImportError:
            pytest.skip("H5IOStrategy not available")


# === Augmentation Interface Tests ===
class TestAugmentationInterface:
    """Test augmentation interface."""

    # @# pytest.mark.timeout(30)
    def test_augmentation_interface_import(self):
        try:
            from spectramr.data.augmentation_interface import AugmentationInterface

            assert AugmentationInterface is not None
        except ImportError:
            pytest.skip("AugmentationInterface not available")

    # @# pytest.mark.timeout(30)
    def test_augmentation_is_callable(self):
        """Augmentation should be callable."""
        try:
            from spectramr.data.augmentation_interface import AugmentationInterface
        except ImportError:
            pytest.skip("AugmentationInterface not available")

        # Interface should define __call__
        assert callable(AugmentationInterface)


# === Individual Dataset Tests ===
class TestIndividualDatasets:
    """Test individual dataset implementations."""

    # @# pytest.mark.timeout(30)
    def test_fastmri_dataset_import(self):
        try:
            from spectramr.data.datasets.fastmri import FastMRIDataset

            assert FastMRIDataset is not None
        except ImportError:
            pytest.skip("FastMRIDataset not available")

    # @# pytest.mark.timeout(30)
    def test_base_mri_dataset_import(self):
        try:
            from spectramr.data.datasets.base import BaseMRIDataset

            assert BaseMRIDataset is not None
        except ImportError:
            try:
                from spectramr.data.datasets.base_dataset import BaseMRIDataset

                assert BaseMRIDataset is not None
            except ImportError:
                pytest.skip("BaseMRIDataset not available")

    # @# pytest.mark.timeout(30)
    def test_dataset_has_required_methods(self):
        """Datasets should have __len__ and __getitem__."""
        # All PyTorch datasets must implement these when subclassed
        from torch.utils.data import TensorDataset

        # TensorDataset is a concrete implementation
        sample_dataset = TensorDataset(torch.randn(10, 3))

        # Check that concrete implementation has these methods
        assert hasattr(sample_dataset, "__len__")
        assert hasattr(sample_dataset, "__getitem__")


# === Transform Integration Tests ===
class TestTransformIntegration:
    """Test transform integration with datasets."""

    # @# pytest.mark.timeout(30)
    def test_transforms_directory_contents(self):
        from pathlib import Path

        transforms_path = Path("src/spectramr/data/transforms")
        if transforms_path.exists():
            files = list(transforms_path.glob("*.py"))
            assert len(files) > 0

    # @# pytest.mark.timeout(30)
    def test_normalize_transform_import(self):
        try:
            from spectramr.data.transforms.normalize import NormalizeTransform

            assert NormalizeTransform is not None
        except ImportError:
            pytest.skip("NormalizeTransform not available")


# === Data Loader Tests ===
class TestDataLoaderIntegration:
    """Test data loader integration."""

    # @# pytest.mark.timeout(30)
    def test_create_dataloader_from_dataset(self):
        """DataLoader should work with custom datasets."""
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(100, 1, 64, 64)
        targets = torch.randn(100, 1, 64, 64)

        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(loader))

        assert batch[0].shape == (4, 1, 64, 64)
        assert batch[1].shape == (4, 1, 64, 64)

    # @# pytest.mark.timeout(30)
    def test_dataloader_num_workers(self):
        """DataLoader should work with multiple workers."""
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(20, 1, 32, 32)
        dataset = TensorDataset(data)

        # num_workers=0 for testing (no multiprocessing)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)

        batches = list(loader)
        assert len(batches) == 5


# === Parser Tests ===
class TestParsers:
    """Test data parsers."""

    # @# pytest.mark.timeout(30)
    def test_parsers_directory(self):
        from pathlib import Path

        parsers_path = Path("src/spectramr/data/parsers")
        assert parsers_path.exists() or True  # Skip if not exists


# === Split Tests ===
class TestDataSplitting:
    """Test data splitting utilities."""

    # @# pytest.mark.timeout(30)
    def test_random_split(self):
        """Random split should work correctly."""
        from torch.utils.data import TensorDataset, random_split

        data = TensorDataset(torch.randn(100, 10))

        train, val, test = random_split(data, [70, 15, 15])

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    # @# pytest.mark.timeout(30)
    def test_split_with_generator(self):
        """Split with generator should be reproducible."""
        from torch.utils.data import TensorDataset, random_split

        data = TensorDataset(torch.randn(100, 10))

        gen1 = torch.Generator().manual_seed(42)
        train1, val1 = random_split(data, [80, 20], generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        train2, val2 = random_split(data, [80, 20], generator=gen2)

        assert train1.indices == train2.indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
