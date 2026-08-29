"""Spec E2 — BartKspaceDataset (BART k-space → tio.Subject) tests.

Three layers: the pure ``canonicalize_bart_array`` dim-map logic (no deps), the
``build_bart_index`` glob, and the dataset's ``__getitem__`` Cartesian recon
(numerically exact, via SSOT ``ifft2c``). The non-Cartesian path's *mechanism-
fires* check (S1 C4: the NUFFT recon is non-degenerate) lives behind a
torchkbnufft guard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mriforge.config.schemas.data import BartConfigSchema
from mriforge.data.datasets.bart_dataset import (
    BartKspaceDataset,
    build_bart_index,
    canonicalize_bart_array,
)


def _write_bart(directory, name: str, arr: np.ndarray):
    dims = list(arr.shape)
    (directory / f"{name}.hdr").write_text(
        "# Dimensions\n" + " ".join(str(d) for d in dims) + "\n"
    )
    arr.astype(np.complex64).flatten(order="F").tofile(directory / f"{name}.cfl")
    return directory / f"{name}.cfl"


# ── pure: canonicalize_bart_array ─────────────────────────────────────────────

def test_canonicalize_puts_coil_first_and_squeezes_singletons():
    # raw BART layout [readout=4, phase=3, dummy=1, coil=2]
    arr = torch.zeros(4, 3, 1, 2, dtype=torch.complex64)
    arr[1, 2, 0, 0] = 5 + 1j  # tag a voxel
    out, roles = canonicalize_bart_array(arr, {"readout": 0, "phase": 1, "coil": 3})
    assert roles == ["coil", "readout", "phase"]  # canonical order
    assert tuple(out.shape) == (2, 4, 3)
    assert out[0, 1, 2] == 5 + 1j  # value preserved through permute


def test_canonicalize_raises_on_undeclared_nonsingleton_dim():
    arr = torch.zeros(4, 3, 5, 2, dtype=torch.complex64)  # dim2 size 5, no role
    with pytest.raises(ValueError, match="silently drop"):
        canonicalize_bart_array(arr, {"readout": 0, "phase": 1, "coil": 3})


def test_canonicalize_raises_on_index_beyond_rank():
    arr = torch.zeros(4, 3, dtype=torch.complex64)
    with pytest.raises(ValueError, match="exceeds BART array rank"):
        canonicalize_bart_array(arr, {"readout": 0, "coil": 9})


def test_canonicalize_keeps_echo_axis():
    # [readout=4, spoke=3, coil=2, echo=2] radial multi-echo
    arr = torch.randn(4, 3, 2, 2, dtype=torch.complex64)
    out, roles = canonicalize_bart_array(
        arr, {"readout": 0, "spoke": 1, "coil": 2, "echo": 3}
    )
    assert roles == ["coil", "echo", "readout", "spoke"]
    assert tuple(out.shape) == (2, 2, 4, 3)


# ── build_bart_index ──────────────────────────────────────────────────────────

def test_build_bart_index_lists_basenames_excluding_traj(tmp_path):
    _write_bart(tmp_path, "vol1", np.ones((2, 2), dtype=np.complex64))
    _write_bart(tmp_path, "vol2", np.ones((2, 2), dtype=np.complex64))
    _write_bart(tmp_path, "vol1_traj", np.ones((2, 2), dtype=np.complex64))
    idx = build_bart_index(tmp_path)
    stems = sorted(Path(p).name for p in idx)
    assert stems == ["vol1", "vol2"]  # _traj excluded


def test_build_bart_index_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_bart_index(tmp_path / "nope")


def test_build_bart_index_file_pattern_filters(tmp_path):
    _write_bart(tmp_path, "data_05_kspace", np.ones((2, 2), dtype=np.complex64))
    _write_bart(tmp_path, "data_05_b1map", np.ones((2, 2), dtype=np.complex64))
    idx = build_bart_index(tmp_path, file_pattern="b1map")
    assert len(idx) == 1
    assert idx[0].endswith("data_05_b1map")


# ── dataset: Cartesian recon (numerically exact) ──────────────────────────────

def test_cartesian_getitem_returns_subject_with_recon_and_kspace(tmp_path):
    import torchio as tio

    from mriforge.infrastructure.physics.fft_ops import ifft2c

    # single-coil Cartesian k-space [readout=8, phase=8, coil=1]
    rng = np.random.default_rng(0)
    img = rng.standard_normal((8, 8)).astype(np.complex64)
    ksp = np.fft.fftshift(np.fft.fft2(img))  # arbitrary k-space; recon checked vs ifft2c
    arr = ksp[:, :, None].astype(np.complex64)
    _write_bart(tmp_path, "scan", arr)

    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)
    assert len(ds) == 1
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    assert "input" in subj and "target" in subj and "kspace" in subj
    recon = subj["target"].tensor  # (C, H, W, 1)
    assert recon.shape[-3:] == torch.Size([8, 8, 1])
    assert torch.isfinite(recon).all()
    # recon must equal |ifft2c(canonical kspace)| (RSS over single coil)
    kc = torch.from_numpy(arr[:, :, 0]).to(torch.complex64).unsqueeze(0)  # [1,8,8]
    expected = ifft2c(kc).abs().squeeze(0)  # [8,8]
    assert torch.allclose(recon[0, :, :, 0], expected, atol=1e-4)
    # input mirrors target; the physics transform pipeline degrades input downstream
    assert torch.allclose(subj["input"].tensor, subj["target"].tensor)


def test_dataset_requires_enabled_config(tmp_path):
    cfg = BartConfigSchema(enabled=False)
    with pytest.raises(ValueError, match="enabled"):
        BartKspaceDataset([], cfg)


# ── dataset: non-Cartesian mechanism-fires (S1 C4) ────────────────────────────

def test_b0_map_from_echoes_emitted(tmp_path):
    """b0_from_echo derives a real B0 (Hz) from the multi-echo data and carries
    it as the batch `b0_map` (the VF real-reference field-scoring seam)."""
    pytest.importorskip("torchkbnufft")
    import torchio as tio

    rng = np.random.default_rng(3)
    # raw BART layout [readout=8, spoke=4, coil=2, echo=2]
    arr = (
        rng.standard_normal((8, 4, 2, 2)) + 1j * rng.standard_normal((8, 4, 2, 2))
    ).astype(np.complex64)
    _write_bart(tmp_path, "v01", arr)
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "spoke": 1, "coil": 2, "echo": 3},
        sampling="radial",
        trajectory_source="golden_angle",
        density_compensation="radial",
        b0_from_echo=True,
        delta_te=0.0025,
    )
    subj = BartKspaceDataset(build_bart_index(tmp_path), cfg)[0]
    assert isinstance(subj, tio.Subject)
    assert "b0_map" in subj
    b0 = subj["b0_map"].tensor  # (1, H, W, 1)
    assert tuple(b0.shape) == (1, 4, 4, 1)  # im_side = readout // 2
    assert torch.isfinite(b0).all()


def test_b1_map_from_dam_emitted(tmp_path):
    """b1_from_dam derives a real B1+ efficiency map (double-angle) from a
    Cartesian 2-flip acquisition and carries it as `b1_map`."""
    import torchio as tio

    rng = np.random.default_rng(4)
    # raw BART layout [readout=8, phase=8, coil=2, flip=2] — Cartesian DAM
    arr = (
        rng.standard_normal((8, 8, 2, 2)) + 1j * rng.standard_normal((8, 8, 2, 2))
    ).astype(np.complex64)
    _write_bart(tmp_path, "b1", arr)
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2, "flip": 3},
        sampling="cartesian",
        b1_from_dam=True,
        b1_nominal_flip_deg=60.0,
    )
    subj = BartKspaceDataset(build_bart_index(tmp_path), cfg)[0]
    assert isinstance(subj, tio.Subject)
    assert "b1_map" in subj
    b1 = subj["b1_map"].tensor  # (1, H, W, 1)
    assert tuple(b1.shape) == (1, 8, 8, 1)
    assert torch.isfinite(b1).all()


# ── F5 (smoke 2026-06-16): header-only recon shape → queue-build OOM avoidance ─
#
# Each ``__getitem__`` runs a full NUFFT adjoint for radial arms; the queue-build
# patch-compat filter's slow ``for subj in dataset`` probe therefore reconstructed
# EVERY volume at build time → host OOM. The dataset now exposes ``index`` as dict
# records carrying a recon ``shape`` derived from the ``.hdr`` header ALONE, so the
# filter runs its no-voxel fast path. See [[project_queue_build_oom_patch_filter]].


def test_recon_hw_header_only_cartesian(tmp_path):
    """Cartesian recon (H, W) = (readout, phase), read from the .hdr alone."""
    _write_bart(tmp_path, "scan", np.ones((64, 48, 1), dtype=np.complex64))
    hw = BartKspaceDataset._recon_hw_header_only(
        str(tmp_path / "scan"), {"readout": 0, "phase": 1, "coil": 2}, "cartesian"
    )
    assert hw == (64, 48)


def test_recon_hw_header_only_radial_is_half_readout_square(tmp_path):
    """Radial NUFFT recon is the ~2x-oversampled square (readout // 2)²."""
    _write_bart(tmp_path, "rad", np.ones((16, 8, 1), dtype=np.complex64))
    hw = BartKspaceDataset._recon_hw_header_only(
        str(tmp_path / "rad"), {"readout": 0, "spoke": 1, "coil": 2}, "radial"
    )
    assert hw == (8, 8)


def test_recon_hw_header_only_missing_header_returns_none(tmp_path):
    assert (
        BartKspaceDataset._recon_hw_header_only(
            str(tmp_path / "nope"), {"readout": 0, "phase": 1}, "cartesian"
        )
        is None
    )


def test_index_records_are_dicts_with_path_and_shape(tmp_path):
    """The dataset index is now ``[{"path", "shape": [1, H, W]}, ...]``."""
    _write_bart(tmp_path, "scan", np.ones((64, 48, 1), dtype=np.complex64))
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)
    rec = ds.index[0]
    assert isinstance(rec, dict)
    assert Path(rec["path"]).name == "scan"
    assert rec["shape"] == [1, 64, 48]


def test_dry_iter_returns_subject_shells_without_recon(tmp_path, monkeypatch):
    """REGRESSION: ``dry_iter`` returns one lightweight Subject per record.

    ``tio.Queue`` calls ``dataset.dry_iter()`` to size the epoch; the dataset
    lacked it, so every BART-backed arm crashed at the first training step with
    ``'BartKspaceDataset' object has no attribute 'dry_iter'`` (cluster VF smoke
    2026-06-16, exp_vf_21..30). It must produce shells WITHOUT reconstructing.
    """
    _write_bart(tmp_path, "scan", np.ones((64, 48, 1), dtype=np.complex64))
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)

    # Poison the recon path: dry_iter must read only ``index`` (no .cfl payload).
    def _boom(*_a, **_k):
        raise AssertionError("dry_iter reconstructed a volume")

    monkeypatch.setattr(BartKspaceDataset, "__getitem__", _boom)
    shells = ds.dry_iter()
    assert len(shells) == len(ds)  # one shell per index record
    assert Path(shells[0]["path"]).name == "scan"
    assert tuple(shells[0]["input"].shape) == (1, 1, 1, 1)  # 1x1x1x1 stub, no voxels


def test_dry_iter_lets_tio_queue_size_the_epoch(tmp_path):
    """The exact crash path: ``tio.Queue.iterations_per_epoch`` → ``dry_iter()``."""
    import torchio as tio

    _write_bart(tmp_path, "scan", np.ones((64, 64, 1), dtype=np.complex64))
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)
    queue = tio.Queue(
        ds,
        max_length=4,
        samples_per_volume=2,
        sampler=tio.data.UniformSampler(patch_size=(32, 32, 1)),
        num_workers=0,
    )
    # Pre-fix this raised: 'BartKspaceDataset' object has no attribute 'dry_iter'.
    assert queue.iterations_per_epoch == len(ds) * 2


def test_queue_filter_uses_fast_path_without_recon(tmp_path, monkeypatch):
    """REGRESSION: the patch-compat filter must NOT reconstruct any volume.

    ``__getitem__`` (which runs the NUFFT/IFFT recon) is poisoned to raise; the
    fast path reads only ``index``+``shape``, so a fitting patch keeps every
    record without touching the recon — the exact build-time OOM avoided.
    """
    from mriforge.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    _write_bart(tmp_path, "scan", np.ones((64, 64, 1), dtype=np.complex64))
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)

    def _boom(*_a, **_k):
        raise AssertionError("reconstructed a volume — slow path taken (OOM risk)")

    monkeypatch.setattr(BartKspaceDataset, "__getitem__", _boom)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, (32, 32))
    assert len(out) == 1


def test_queue_filter_drops_too_small_bart_via_header(tmp_path, monkeypatch):
    """A recon smaller than the patch is dropped from ``index`` with no recon."""
    from mriforge.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    _write_bart(tmp_path, "scan", np.ones((16, 16, 1), dtype=np.complex64))
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
        sampling="cartesian",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)

    def _boom(*_a, **_k):
        raise AssertionError("slow path taken")

    monkeypatch.setattr(BartKspaceDataset, "__getitem__", _boom)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, (32, 32))
    assert len(out) == 0  # 16x16 < 32x32 patch → dropped via header shape


def test_radial_recon_is_non_degenerate(tmp_path):
    pytest.importorskip("torchkbnufft")
    import torchio as tio

    # radial k-space [readout=16, spoke=8, coil=1]
    rng = np.random.default_rng(1)
    arr = (
        rng.standard_normal((16, 8, 1)) + 1j * rng.standard_normal((16, 8, 1))
    ).astype(np.complex64)
    _write_bart(tmp_path, "radial", arr)

    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 0, "spoke": 1, "coil": 2},
        sampling="radial",
        trajectory_source="golden_angle",
        density_compensation="radial",
    )
    ds = BartKspaceDataset(build_bart_index(tmp_path), cfg)
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    recon = subj["target"].tensor
    assert torch.isfinite(recon).all()
    # mechanism fires: the NUFFT adjoint produced a non-constant image, not a blob
    assert recon.std() > 0
