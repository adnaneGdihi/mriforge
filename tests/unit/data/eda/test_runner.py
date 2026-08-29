"""End-to-end tests for the per-dataset runner."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mriforge.data.eda.catalog import discover
from mriforge.data.eda.runner import default_output_dir, run_all, run_dataset


def test_default_output_dir_is_timestamped():
    p = default_output_dir(now=datetime(2026, 6, 11, 9, 30, 15))
    assert p == Path("experiments/results/eda_20260611_093015")


def _entry(manifest_path, tmp_path, dataset_id):
    return {e.dataset_id: e for e in discover(manifest_path.parent, tmp_path)}[dataset_id]


def test_run_dataset_present_emits_core_figures(present_kspace_manifest, tmp_path):
    entry = _entry(present_kspace_manifest, tmp_path, "tiny_kspace")
    out_dir = tmp_path / "out"
    produced = run_dataset(entry, out_dir, budget=2)
    names = {p.name for p in produced}
    assert "00_card.png" in names
    assert "20_norm_voxel_dist.png" in names           # the requested normalized-voxel figure
    assert "41_kspace_logmag.png" in names             # kspace modality ran
    assert (out_dir / "tiny_kspace" / "profile.json").exists()


def test_run_dataset_survives_2d_kspace_dummy(tmp_path):
    """Regression: a 2-D k-space dummy (e.g. a fastMRI placeholder) must not crash run_dataset.

    The middle-slice helper returns None for 2-D data so the strategy loads it whole; the
    k-space probe block is wrapped so any residual shape issue is contained per-dataset.
    """
    import json

    import h5py
    import numpy as np

    root = tmp_path / "databases" / "dummyset"
    root.mkdir(parents=True)
    with h5py.File(root / "d.h5", "w") as f:
        f.create_dataset("kspace", data=np.zeros((16, 16), np.complex64))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "dummy_kspace.json").write_text(json.dumps({
        "dataset_name": "dummy_kspace", "data_root": str(root),
        "file_type": "h5", "records": [{"relative_path": "d.h5"}],
    }))
    entry = {e.dataset_id: e for e in discover(manifests, tmp_path)}["dummy_kspace"]
    produced = run_dataset(entry, tmp_path / "out", budget=2)  # must not raise
    assert any(p.name == "00_card.png" for p in produced)


def test_run_dataset_noncartesian_skips_cartesian_probes(tmp_path):
    """Radial/spiral data must not get a Cartesian recon montage or the sampling/coil-map
    probes — those assume a Cartesian grid and are physically meaningless without NUFFT
    gridding (they produced the 'heart dataset shows a strip' figures). Only the raw
    k-space log-magnitude (the honest acquired-samples view) is shown."""
    import json

    import h5py
    import numpy as np

    root = tmp_path / "databases" / "radialset"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    ksp = (rng.standard_normal((4, 32, 64)) + 1j * rng.standard_normal((4, 32, 64))).astype(np.complex64)
    with h5py.File(root / "spokes.h5", "w") as f:
        f.create_dataset("kspace", data=ksp)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "tiny_radial.json").write_text(json.dumps({
        "dataset_name": "tiny_radial", "data_root": str(root), "file_type": "h5",
        "records": [{"relative_path": "spokes.h5"}],
        "source": {"role": "RAW", "raw_kspace": True, "anatomy": "cardiac"},
    }))
    entry = {e.dataset_id: e for e in discover(manifests, tmp_path)}["tiny_radial"]
    assert entry.modality == "noncartesian"
    produced = run_dataset(entry, tmp_path / "out", budget=2)
    names = {p.name for p in produced}
    assert "41_kspace_logmag.png" in names         # honest raw-samples view kept
    assert "43_sampling_pattern.png" not in names  # Cartesian-only → skipped
    assert "44_coil_maps.png" not in names         # ESPIRiT is Cartesian-only → skipped
    assert "10_sample_montage.png" not in names    # no fake ifft2c recon montage


def test_run_dataset_absent_emits_only_card(absent_external_manifest, tmp_path):
    entry = _entry(absent_external_manifest, tmp_path, "calgary_campinas")
    produced = run_dataset(entry, tmp_path / "out", budget=2)
    assert [p.name for p in produced] == ["00_card.png"]


@pytest.mark.smoke
def test_run_all_two_datasets_plus_overview(present_kspace_manifest, absent_external_manifest, tmp_path):
    out = tmp_path / "out"
    rows = run_all(present_kspace_manifest.parent, data_root=tmp_path, out_root=out, budget=2)
    ids = {r["dataset_id"] for r in rows}
    assert {"tiny_kspace", "calgary_campinas"} <= ids
    assert (out / "_overview" / "00_landscape.png").exists()
    assert (out / "_overview" / "04_coverage_matrix.png").exists()


def test_run_all_surfaces_read_failures_as_issues(
    present_kspace_manifest, absent_external_manifest, tmp_path
):
    """coverage.json rows must surface a per-dataset ``issues`` list so a dataset that was not
    read/shown correctly stops masquerading as a clean ``error=null`` row (CLAUDE.md #10).

    A clean dataset has ``issues == []``; a dataset with no resolvable scan dir reports the
    failure in ``issues`` WITHOUT crashing the run (``error`` stays None).
    """
    out = tmp_path / "out"
    rows = {r["dataset_id"]: r for r in run_all(
        present_kspace_manifest.parent, data_root=tmp_path, out_root=out, budget=2)}

    clean = rows["tiny_kspace"]
    assert "issues" in clean and "notes" in clean        # new top-level surface
    assert clean["issues"] == []                          # a good dataset is not flagged

    degraded = rows["calgary_campinas"]
    assert degraded["error"] is None                      # surfaced, not crashed
    assert degraded["issues"]                              # but NOT silently clean
    assert any("absent or no scan dir" in s for s in degraded["issues"])


def test_skipped_malformed_read_surfaces_as_issue():
    """The 1-D 'wrong-key' read note (loader._is_plane guard) is a genuine read degradation, so
    it must appear in coverage.json ``issues`` — not be swallowed (CLAUDE.md #10)."""
    from mriforge.data.eda.runner import _issues

    notes = ["skipped malformed read (shape (150,)): scan0.h5", "harmless info note"]
    assert _issues(notes) == ["skipped malformed read (shape (150,)): scan0.h5"]


def test_run_dataset_renders_stub_manifest_with_on_disk_data(tmp_path):
    """The 2026-06-13 headline fix end-to-end: a descriptor stub (empty records[],
    status=not_downloaded) whose data is actually on disk must render its figure suite — the EDA
    trusts the filesystem, so a re-run on the cluster no longer shows it as a bare 'absent' card.
    """
    import json

    import h5py
    import numpy as np

    base = tmp_path / "databases" / "external" / "osi2one_x"
    (base / "raw").mkdir(parents=True)              # archives only
    rng = np.random.default_rng(0)
    ksp = (rng.standard_normal((2, 16, 16)) + 1j * rng.standard_normal((2, 16, 16))).astype(np.complex64)
    (base / "extracted").mkdir()
    with h5py.File(base / "extracted" / "scan0.h5", "w") as f:  # real data in the sibling
        f.create_dataset("kspace", data=ksp)
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    (manifests / "osi2one_x.json").write_text(json.dumps({
        "dataset_name": "osi2one_x",
        "data_root": "databases/external/osi2one_x/raw",
        "file_type": "h5", "records": [], "status": "not_downloaded",
        "source": {"role": "RAW", "raw_kspace": True},
    }))
    entry = {e.dataset_id: e for e in discover(manifests, tmp_path)}["osi2one_x"]
    assert entry.tier == "present"  # promoted from disk despite empty records[]
    produced = run_dataset(entry, tmp_path / "out", budget=2)
    names = {p.name for p in produced}
    assert "00_card.png" in names
    assert "10_sample_montage.png" in names  # actually rendered the data, not just a card


def test_run_dataset_temporal_renders_strip(tmp_path):
    """A temporal dataset (cine .mat) renders 61_temporal_strip.png — the dedicated time-axis view
    the previous suite lacked (cine/4D-flow/5D were card-only)."""
    import json

    import h5py
    import numpy as np

    root = tmp_path / "databases" / "cine_ds"
    root.mkdir(parents=True)
    with h5py.File(root / "cine.mat", "w") as f:  # v7.3 .mat: (T, H, W) cine
        f.create_dataset("cine", data=np.random.default_rng(0).standard_normal((12, 24, 24)).astype(np.float32))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "dualstage_bssfp_cine.json").write_text(json.dumps({
        "dataset_name": "dualstage_bssfp_cine", "data_root": str(root),
        "file_type": "mat", "records": [{"relative_path": "cine.mat"}],
        "source": {"role": "img"},
    }))
    entry = {e.dataset_id: e for e in discover(manifests, tmp_path)}["dualstage_bssfp_cine"]
    assert entry.modality == "temporal"  # 'cine' name hint
    produced = run_dataset(entry, tmp_path / "out", budget=2)
    assert any(p.name == "61_temporal_strip.png" for p in produced)
