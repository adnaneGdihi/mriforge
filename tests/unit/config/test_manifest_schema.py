from spectramr.config.schemas.manifest_schema import (
    DatasetManifest,
    DatasetManifestCSV,
    DatasetManifestIndex,
    FileMetadata,
)


class TestFileMetadata:
    def test_defaults(self):
        meta = FileMetadata(file_path="/path/to/file")
        assert meta.file_path == "/path/to/file"
        assert meta.shape is None
        assert meta.dtype is None
        assert meta.contrast is None
        assert meta.preprocessing_status == "original"
        assert meta.file_size_bytes is None
        assert meta.checksum is None
        assert meta.metadata == {}

    def test_custom_values(self):
        meta = FileMetadata(
            file_path="/path/to/file",
            shape=(1, 256, 256),
            dtype="float32",
            contrast="T1w",
            preprocessing_status="normalized",
            file_size_bytes=1024,
            checksum="abc123hash",
            metadata={"key": "value"},
        )
        assert meta.shape == (1, 256, 256)
        assert meta.dtype == "float32"
        assert meta.contrast == "T1w"
        assert meta.preprocessing_status == "normalized"
        assert meta.file_size_bytes == 1024
        assert meta.checksum == "abc123hash"
        assert meta.metadata == {"key": "value"}


class TestDatasetManifest:
    def test_defaults(self):
        manifest = DatasetManifest(
            dataset_key="test_dataset",
            variant="original",
            display_name="Test Dataset",
            dataset_type="2d",
        )
        assert manifest.dataset_key == "test_dataset"
        assert manifest.variant == "original"
        assert manifest.display_name == "Test Dataset"
        assert manifest.dataset_type == "2d"
        assert manifest.files == []
        assert manifest.total_files == 0
        assert manifest.total_size_bytes == 0
        assert manifest.metadata == {}
        assert manifest.manifest_version == "1.0"

    def test_methods(self):
        files = [
            FileMetadata(file_path="f1", shape=(1, 256, 256), contrast="T1w"),
            FileMetadata(file_path="f2", shape=(1, 256, 256), contrast="T2w"),
            FileMetadata(file_path="f3", shape=(1, 128, 128), contrast="T1w"),
        ]
        manifest = DatasetManifest(
            dataset_key="test_dataset",
            variant="original",
            display_name="Test Dataset",
            dataset_type="2d",
            files=files,
        )

        shapes = manifest.get_shapes()
        assert len(shapes) == 2
        assert (1, 256, 256) in shapes
        assert (1, 128, 128) in shapes

        contrasts = manifest.get_contrasts()
        assert len(contrasts) == 2
        assert "T1w" in contrasts
        assert "T2w" in contrasts

        counts = manifest.get_file_count_by_contrast()
        assert counts["T1w"] == 2
        assert counts["T2w"] == 1


class TestDatasetManifestIndex:
    def test_methods(self):
        manifest1 = DatasetManifest(
            dataset_key="d1", variant="original", display_name="D1", dataset_type="2d"
        )
        manifest2 = DatasetManifest(
            dataset_key="d1",
            variant="normalized",
            display_name="D1 Norm",
            dataset_type="2d",
        )
        manifest3 = DatasetManifest(
            dataset_key="d2", variant="original", display_name="D2", dataset_type="2d"
        )

        index = DatasetManifestIndex(
            manifests={
                ("d1", "original"): manifest1,
                ("d1", "normalized"): manifest2,
                ("d2", "original"): manifest3,
            }
        )

        assert index.get_manifest("d1", "original") == manifest1
        assert index.get_manifest("d1", "normalized") == manifest2
        assert index.get_manifest("d1", "missing") is None

        variants = index.get_all_variants("d1")
        assert len(variants) == 2
        assert manifest1 in variants
        assert manifest2 in variants

        keys = index.get_dataset_keys()
        assert "d1" in keys
        assert "d2" in keys


class TestDatasetManifestCSV:
    def test_from_dict(self):
        row = {
            "dataset_key": "d1",
            "display_name": "D1",
            "dataset_type": "2d",
            "description": "Desc",
            "total_files": "100",
            "total_size_mb": "500",
            "scanned_at": "1234567890",
            "dataset_dir": "/path/to/d1",
            "variants_available": "original;normalized",
        }
        manifest = DatasetManifestCSV.from_dict(row)
        assert manifest.dataset_key == "d1"
        assert manifest.total_files == 100
        assert manifest.variants_available == ["original", "normalized"]

    def test_to_dict(self):
        manifest = DatasetManifestCSV(
            dataset_key="d1",
            display_name="D1",
            dataset_type="2d",
            description="Desc",
            total_files=100,
            total_size_mb=500,
            scanned_at=1234567890.0,
            dataset_dir="/path/to/d1",
            variants_available=["original", "normalized"],
        )
        row = manifest.to_dict()
        assert row["dataset_key"] == "d1"
        assert row["total_files"] == 100
        assert row["variants_available"] == "original;normalized"
