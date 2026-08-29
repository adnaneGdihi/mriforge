"""Held-out test-split roster resolution from the paired v4 manifest.

Moved here with the helper itself, which used to be
``DataPipelineDirector._resolve_manifest_test_paths`` -- a private static
method on a class whose only caller was deleted. The capability is unique
(``pipelines/infer.py`` otherwise globs the filesystem and has no manifest
route at all), so it was preserved rather than dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from mriforge.config.schemas.data import DataConfigSchema, DataSourceConfigSchema
from mriforge.data.metadata.test_split_resolver import resolve_manifest_test_paths


def test_no_manifest_configured_returns_empty() -> None:
    """No paired manifest ⇒ no test roster, not a crash."""
    cfg = DataConfigSchema()
    assert cfg.source.paired_manifest_path is None
    assert resolve_manifest_test_paths(cfg) == []


class TestResolveManifestTestPaths:
    """The held-out test split is the unpaired-ULF cohort (``split_hint ==
    "test"``, no HF target). See ``preprocess_ulf_paired.py`` for how it is
    emitted. Moved here from ``test_data_pipeline_director.py`` with the
    helper."""

    @staticmethod
    def _manifest(tmp_path):
        manifest = {
            "manifest_version": "4.0",
            "dataset": "ulf_paired_brain",
            "records": [
                {
                    "pairing_status": "paired",
                    "split_hint": "train",
                    "hf_resolution": "highres",
                    "contrast": "FLAIR",
                    "primary_path": "databases/x/sub-0011_FLAIR_ulf.nii.gz",
                    "target_path": "databases/x/sub-0011_FLAIR_hf.nii.gz",
                },
                {
                    "pairing_status": "unpaired_ulf",
                    "split_hint": "test",
                    "hf_resolution": None,
                    "contrast": "T2w",
                    "primary_path": "databases/x/sub-0011_T2w_ulf.nii.gz",
                    "target_path": None,
                },
                {
                    "pairing_status": "unpaired_ulf",
                    "split_hint": "test",
                    "hf_resolution": None,
                    "contrast": "FLAIR",
                    "primary_path": "databases/x/sub-9999_FLAIR_ulf.nii.gz",
                    "target_path": None,
                },
            ],
        }
        p = tmp_path / "ulf_paired_v6.json"
        p.write_text(json.dumps(manifest))
        return p

    def test_resolves_unpaired_test_inputs(self, tmp_path) -> None:
        from tests.utils.data_config_stub import DataConfigStub

        mpath = self._manifest(tmp_path)
        cfg = DataConfigStub(paired_manifest_path=str(mpath), contrasts=None)
        paths = resolve_manifest_test_paths(cfg)
        names = sorted(p.name for p in paths)
        # only the two split_hint=='test' (unpaired) records, never the paired one
        assert names == ["sub-0011_T2w_ulf.nii.gz", "sub-9999_FLAIR_ulf.nii.gz"]

    def test_no_manifest_returns_empty(self) -> None:
        import types

        # No paired manifest configured → legacy empty-dataset placeholder.
        cfg = types.SimpleNamespace(
            # `data.source.paired_manifest_path` since phase 9g.
            source=types.SimpleNamespace(paired_manifest_path=None)
        )
        assert resolve_manifest_test_paths(cfg) == []


def _write_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A v4 paired manifest with one test record and one train record."""
    test_input = tmp_path / "sub-001_ULF.nii.gz"
    train_input = tmp_path / "sub-002_ULF.nii.gz"
    for p in (test_input, train_input):
        p.write_bytes(b"")

    manifest = tmp_path / "paired_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "4.0",
                "records": [
                    {
                        "subject_id": "sub-001",
                        "primary_path": str(test_input),
                        "split_hint": "test",
                        "pairing_status": "unpaired_ulf",
                    },
                    {
                        "subject_id": "sub-002",
                        "primary_path": str(train_input),
                        "split_hint": "train",
                        "pairing_status": "unpaired_ulf",
                    },
                ],
            }
        )
    )
    return manifest, test_input, train_input


def test_resolves_only_the_test_split(tmp_path: Path) -> None:
    """The roster is the ``split_hint == "test"`` cohort and nothing else.

    Asserts the train record is EXCLUDED, not merely that the test record is
    present: a resolver that returned every record would pass the weaker
    check while quietly running inference over the training corpus.
    """
    manifest, test_input, train_input = _write_manifest(tmp_path)
    cfg = DataConfigSchema(source=DataSourceConfigSchema(paired_manifest_path=str(manifest)))

    paths = resolve_manifest_test_paths(cfg)

    resolved = {Path(p).name for p in paths}
    assert test_input.name in resolved
    assert train_input.name not in resolved
