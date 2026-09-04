"""parse_fastmri_index must propagate a record's site tag into metadata.

The multi-site / federated manifests (schema ``federated_with_site_tag``, e.g.
``data/manifests/cross_site_pretrain.json``) carry a top-level ``site_id`` per
record. The site-aware loader paths consume ``record['metadata']['site']``:
``ManifestDatasetLoader`` Leave-One-Site-Out holdout (manifest_loader.py:174-195)
and, downstream, ``attach_site_id`` for ``data.expose_site_id``. The parser used
to keep only ``metadata/target_path/sensitivity_path/shape`` and DROP the
top-level ``site_id``, so the site partition never reached the site logic — the
federated arm silently trained as if single-site (every sample defaulting to
site 0). This pins the propagation.
"""
from __future__ import annotations

import json

from spectramr.data.datasets.universal_dataset import parse_fastmri_index


def _write_manifest(tmp_path, records) -> str:
    manifest = {
        "manifest_version": "3.1",
        "dataset_name": "cross_site_test",
        "data_root": str(tmp_path),
        "file_type": "h5",
        "schema": "federated_with_site_tag",
        "sites": ["site_0", "site_1"],
        "records": records,
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest))
    return str(p)


def test_parse_propagates_top_level_site_id_into_metadata(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"relative_path": "/abs/a_FLAIR01.h5", "file_id": "a_FLAIR01", "site_id": "site_0"},
            {"relative_path": "/abs/b_FLAIR02.h5", "file_id": "b_FLAIR02", "site_id": "site_1"},
        ],
    )
    index = parse_fastmri_index(path)
    assert len(index) == 2
    assert index[0]["metadata"]["site"] == "site_0"
    assert index[1]["metadata"]["site"] == "site_1"


def test_parse_does_not_invent_a_site_when_absent(tmp_path):
    """A plain (non-federated) manifest record must not gain a fabricated site."""
    path = _write_manifest(
        tmp_path,
        [{"relative_path": "/abs/c.h5", "file_id": "c"}],
    )
    index = parse_fastmri_index(path)
    assert index[0].get("metadata", {}).get("site") is None
