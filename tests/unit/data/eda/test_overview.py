"""Tests for the cross-dataset overview figures."""
from __future__ import annotations

from mriforge.data.eda.overview import render_overview


def test_render_overview_writes_landscape_and_coverage(tmp_path):
    entries = [
        {
            "dataset_id": "a", "modality": "kspace", "tier": "present", "status": "downloaded",
            "field_T": [3.0], "anatomy": "knee", "role": "RAW", "n_records": 5,
            "figures": ["00_card.png", "41_kspace_logmag.png"], "error": None,
            "norm_intensity": [0.1, 0.5, 0.9],
        },
        {
            "dataset_id": "b", "modality": "kspace", "tier": "absent", "status": "not_downloaded",
            "field_T": [1.5], "anatomy": "brain", "role": "RAW", "n_records": 0,
            "figures": ["00_card.png"], "error": None, "norm_intensity": [],
        },
    ]
    produced = render_overview(entries, out_dir=tmp_path / "_overview")
    names = {p.name for p in produced}
    assert "00_landscape.png" in names
    assert "01_domain_shift.png" in names
    assert "02_sample_counts.png" in names
    assert "04_coverage_matrix.png" in names
