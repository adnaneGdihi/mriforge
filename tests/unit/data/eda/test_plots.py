"""Smoke tests for the EDA renderers — each writes a non-empty PNG."""
from __future__ import annotations

from pathlib import Path

import torch

from mriforge.data.eda import plots, probes
from mriforge.data.eda.catalog import DatasetEntry


def _entry(modality="kspace", tier="present"):
    return DatasetEntry(
        "ds", (), Path("m.json"), Path("."), "h5", (), modality, tier, "downloaded",
        {"role": "RAW", "field_T": [3.0], "anatomy": "knee", "provides": "raw", "links": []},
    )


def test_render_card_writes_png(tmp_path):
    out = plots.render_card(_entry(tier="absent"), profile={}, out_dir=tmp_path)
    assert out.exists() and out.stat().st_size > 0 and out.name == "00_card.png"


def test_render_norm_voxel_dist(tmp_path):
    dist = probes.normalized_distributions([torch.rand(32, 32) * 50])
    out = plots.render_norm_voxel_dist(dist, out_dir=tmp_path)
    assert out.exists() and out.name == "20_norm_voxel_dist.png"


def test_render_montage(tmp_path):
    out = plots.render_montage([torch.rand(32, 32) for _ in range(4)], tmp_path, "10_sample_montage.png", "t")
    assert out.exists() and out.stat().st_size > 0


def test_render_quality_and_percentile(tmp_path):
    assert plots.render_quality({"records": 3, "nan_slices": 0}, tmp_path).exists()
    assert plots.render_percentile_table(probes.percentiles([torch.rand(16, 16)]), tmp_path).exists()


def test_render_inventory(tmp_path):
    out = plots.render_inventory({"contrast": {"T1": 2, "T2": 3}, "field": {}}, tmp_path)
    assert out.exists() and out.name == "02_inventory.png"


def test_render_kspace_logmag(tmp_path):
    ksp = torch.randn(4, 32, 32) + 1j * torch.randn(4, 32, 32)
    out = plots.render_kspace_logmag(ksp, out_dir=tmp_path)
    assert out.exists() and out.name == "41_kspace_logmag.png"


def test_render_radial_energy(tmp_path):
    ksp = torch.randn(1, 32, 32) + 1j * torch.randn(1, 32, 32)
    radii, energy, cumulative = probes.radial_energy(ksp)
    out = plots.render_radial_energy(radii, energy, cumulative, out_dir=tmp_path)
    assert out.exists() and out.name == "42_radial_energy.png"


def test_render_sampling_pattern(tmp_path):
    ksp = torch.randn(1, 32, 32) + 1j * torch.randn(1, 32, 32)
    info = probes.detect_sampling(ksp)
    out = plots.render_sampling_pattern(info["mask"], info, out_dir=tmp_path)
    assert out.exists() and out.name == "43_sampling_pattern.png"


def test_render_sampling_pattern_fully_sampled_is_bright(tmp_path):
    """A fully-sampled (all-ones) mask must render brighter than an all-zero mask. Without a
    pinned vmin/vmax, matplotlib normalizes a constant array to vmin==vmax and renders it
    black — so a fully-sampled scan looked identical to an unsampled one (the ocmr bug)."""
    import numpy as np
    from PIL import Image

    def _lum(p):
        return float(np.asarray(Image.open(p).convert("L")).mean())

    full = {"acceleration": 1.0, "acs_size": 40, "fully_sampled": True}
    empty = {"acceleration": 40.0, "acs_size": 0, "fully_sampled": False}
    b_full = _lum(plots.render_sampling_pattern(np.ones(40, np.uint8), full, tmp_path / "f"))
    b_empty = _lum(plots.render_sampling_pattern(np.zeros(40, np.uint8), empty, tmp_path / "e"))
    assert b_full > b_empty + 20


def test_render_phase(tmp_path):
    ph = probes.phase_stats(torch.randn(16, 16) + 1j * torch.randn(16, 16))
    out = plots.render_phase(ph["phase"], ph["grad_mag"], out_dir=tmp_path)
    assert out.exists() and out.name == "47_phase.png"


def test_render_paired(tmp_path):
    src, tgt = torch.rand(32, 32), torch.rand(32, 32)
    assert plots.render_paired_montage([(src, tgt)], tmp_path).name == "30_paired_montage.png"
    out = plots.render_paired_bland_altman(src, tgt, out_dir=tmp_path)
    assert out.exists() and out.name == "32_paired_bland_altman.png"


def test_render_map_montage(tmp_path):
    out = plots.render_map_montage([torch.rand(32, 32) * 2000], out_dir=tmp_path, unit="ms")
    assert out.exists() and out.name == "50_map_montage.png"


def test_render_trajectory(tmp_path):
    import numpy as np
    t = np.linspace(0, 1, 200)
    out = plots.render_trajectory(np.cos(t * 20) * t, np.sin(t * 20) * t, t, out_dir=tmp_path)
    assert out.exists() and out.name == "60_trajectory.png"


def test_render_temporal_strip_writes_png(tmp_path):
    """The temporal suite: a row of sampled frames + a temporal-std panel → 61_temporal_strip.png."""
    import torch

    from mriforge.data.eda import plots

    frames = [torch.rand(16, 20) for _ in range(5)]
    out = plots.render_temporal_strip(frames, tmp_path)
    assert out.name == "61_temporal_strip.png" and out.exists()
