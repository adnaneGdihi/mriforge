"""Tests for :mod:`spectramr.data.builders.eval_dataset_builder`.

This module is the Phase-1 plug for the eval-bypass CLAUDE.md #11
violation. It exposes role-aware loaders that the campaign evaluator
delegates to — see :mod:`spectramr.infrastructure.orchestration.campaign_evaluator`.

Coverage:

- Each role (``clean_reference``, ``paired_samples``,
  ``per_sample_metrics``, ``uncertainty_bundle``) handles the file
  formats it claims to handle.
- Missing-file behavior is consistent (returns ``None`` / empty list
  rather than raising).
- Malformed-file behavior is fail-soft for missing keys (returns
  ``None`` and logs a warning), not silent.
- Reference tensors are **unnormalized** (no [-1, 1] rescaling like
  :func:`spectramr.infrastructure.io.adapters.load_nifti` applies).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from spectramr.data.builders.eval_dataset_builder import (
    load_clean_reference,
    load_paired_samples,
    load_per_sample_metrics,
    load_uncertainty_bundle,
)


# ── clean_reference ─────────────────────────────────────────────────────────


def test_clean_reference_loads_pt(tmp_path: Path) -> None:
    """``.pt`` file → returns the stored tensor unchanged."""
    expected = torch.arange(24).float().reshape(2, 3, 4)
    target = tmp_path / "ref.pt"
    torch.save(expected, target)
    loaded = load_clean_reference(target)
    assert loaded is not None
    assert torch.equal(loaded, expected)


def test_clean_reference_loads_npy(tmp_path: Path) -> None:
    """``.npy`` file → float32 tensor with the array's values."""
    arr = np.linspace(-100, 100, 60, dtype=np.float32).reshape(3, 4, 5)
    target = tmp_path / "ref.npy"
    np.save(target, arr)
    loaded = load_clean_reference(target)
    assert loaded is not None
    assert loaded.dtype == torch.float32
    np.testing.assert_allclose(loaded.numpy(), arr)


def test_clean_reference_does_not_normalize_npy(tmp_path: Path) -> None:
    """Critical: eval references must NOT be rescaled to [-1, 1].

    :func:`spectramr.infrastructure.io.adapters.load_nifti` min-max
    rescales to ``[-1, 1]`` for training-input ingestion. The eval
    loader must NOT apply this — metrics need the original range.
    """
    big = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
    target = tmp_path / "big.npy"
    np.save(target, big)
    loaded = load_clean_reference(target)
    assert loaded is not None
    # If normalization were applied, max would be near 1.0. We want 4000.
    assert loaded.max().item() == pytest.approx(4000.0)
    assert loaded.min().item() == pytest.approx(1000.0)


def test_clean_reference_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_clean_reference(tmp_path / "nonexistent.pt") is None


def test_clean_reference_returns_none_for_unsupported_suffix(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.txt"
    target.write_text("not really data")
    assert load_clean_reference(target) is None


def test_clean_reference_loads_h5_recognized_key(tmp_path: Path) -> None:
    """Common dataset keys (``data``, ``image``, ``reconstruction``,
    ``kspace``) are detected in priority order."""
    h5py = pytest.importorskip("h5py")
    target = tmp_path / "ref.h5"
    arr = np.random.RandomState(0).randn(4, 6).astype(np.float32)
    with h5py.File(target, "w") as f:
        f["image"] = arr
    loaded = load_clean_reference(target)
    assert loaded is not None
    np.testing.assert_allclose(loaded.numpy(), arr)


def test_clean_reference_h5_returns_none_for_no_recognized_key(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    target = tmp_path / "weird.h5"
    with h5py.File(target, "w") as f:
        f["some_other_key"] = np.array([1, 2, 3])
    assert load_clean_reference(target) is None


def test_clean_reference_loads_nifti(tmp_path: Path) -> None:
    """NIfTI loader returns unnormalized float32."""
    nib = pytest.importorskip("nibabel")
    vol = np.random.RandomState(42).randn(8, 10, 12).astype(np.float32) * 200
    affine = np.eye(4, dtype=np.float32)
    target = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), str(target))
    loaded = load_clean_reference(target)
    assert loaded is not None
    assert loaded.shape == (8, 10, 12)
    # Unnormalized — values should match the input, not be in [-1, 1].
    assert loaded.abs().max().item() > 1.5


# ── paired_samples ──────────────────────────────────────────────────────────


def test_paired_samples_loads_npz(tmp_path: Path) -> None:
    inputs = np.random.RandomState(1).randn(5, 3, 4).astype(np.float32)
    targets = np.random.RandomState(2).randn(5, 3, 4).astype(np.float32)
    target = tmp_path / "paired.npz"
    np.savez(target, inputs=inputs, targets=targets)
    samples = load_paired_samples(target)
    assert len(samples) == 5
    for i, (inp, tgt) in enumerate(samples):
        np.testing.assert_allclose(inp.numpy(), inputs[i])
        np.testing.assert_allclose(tgt.numpy(), targets[i])


def test_paired_samples_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_paired_samples(tmp_path / "nope.npz") == []


def test_paired_samples_returns_empty_when_keys_missing(
    tmp_path: Path,
) -> None:
    """``.npz`` without ``inputs``/``targets`` → empty list (not raise)."""
    target = tmp_path / "wrong_keys.npz"
    np.savez(target, foo=np.zeros(3), bar=np.zeros(3))
    assert load_paired_samples(target) == []


# ── per_sample_metrics ──────────────────────────────────────────────────────


def test_per_sample_metrics_returns_full_dict(tmp_path: Path) -> None:
    target = tmp_path / "metrics.npz"
    np.savez(
        target,
        psnr=np.array([30.0, 31.0, 32.0]),
        ssim=np.array([0.9, 0.91, 0.92]),
        lpips=np.array([0.05, 0.04, 0.03]),
    )
    metrics = load_per_sample_metrics(target)
    assert metrics is not None
    assert set(metrics.keys()) == {"psnr", "ssim", "lpips"}
    np.testing.assert_allclose(metrics["psnr"], [30.0, 31.0, 32.0])


def test_per_sample_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_per_sample_metrics(tmp_path / "absent.npz") is None


# ── uncertainty_bundle ──────────────────────────────────────────────────────


def test_uncertainty_bundle_returns_three_tensors(tmp_path: Path) -> None:
    preds = np.random.RandomState(3).randn(4, 8, 8).astype(np.float32)
    tgts = np.random.RandomState(4).randn(4, 8, 8).astype(np.float32)
    uncs = np.abs(np.random.RandomState(5).randn(4, 8, 8)).astype(np.float32)
    target = tmp_path / "unc.npz"
    np.savez(target, predictions=preds, targets=tgts, uncertainties=uncs)
    bundle = load_uncertainty_bundle(target)
    assert bundle is not None
    pp, tt, uu = bundle
    np.testing.assert_allclose(pp.numpy(), preds)
    np.testing.assert_allclose(tt.numpy(), tgts)
    np.testing.assert_allclose(uu.numpy(), uncs)


def test_uncertainty_bundle_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_uncertainty_bundle(tmp_path / "absent.npz") is None


def test_uncertainty_bundle_returns_none_for_missing_keys(
    tmp_path: Path,
) -> None:
    """Missing any required key → None + warning (not raise)."""
    target = tmp_path / "incomplete.npz"
    np.savez(target, predictions=np.zeros((2, 4)), targets=np.zeros((2, 4)))
    # Missing 'uncertainties'.
    assert load_uncertainty_bundle(target) is None


# ── Round-trip with the campaign-evaluator path it replaces ─────────────────


def test_campaign_evaluator_uses_eval_dataset_builder() -> None:
    """Pin: the refactor in
    :mod:`spectramr.infrastructure.orchestration.campaign_evaluator` must
    import the eval-dataset-builder loaders. If a future refactor
    re-introduces direct ``np.load`` / ``nib.load`` / ``h5py.File``
    calls there, this test (plus
    :mod:`tests.data_integrity.test_data_loading_ssot`) trips.
    """
    src = Path(
        "src/spectramr/infrastructure/orchestration/campaign_evaluator.py"
    ).read_text(encoding="utf-8")
    assert "from spectramr.data.builders.eval_dataset_builder import" in src, (
        "campaign_evaluator must import the SSOT eval loaders — Phase 1 "
        "of the data-layer unification plan."
    )
    for name in (
        "load_clean_reference",
        "load_paired_samples",
        "load_per_sample_metrics",
        "load_uncertainty_bundle",
    ):
        assert name in src, (
            f"campaign_evaluator must use {name!r} — Phase 1 refactor "
            "incomplete."
        )


# ── Pickle manifest stays inline (Amendment C) ──────────────────────────────


def test_pickle_manifest_remains_inline_per_amendment_c(tmp_path: Path) -> None:
    """Amendment C of the data-layer unification plan: a ``.pkl``
    manifest carries paths/config, NOT MRI data. Loading it inline
    (with ``pickle.load``) is correct and out of SSOT scope.

    This test pins the boundary: if we ever decide to route pkl
    manifests through the data layer too, the assertion below has to
    flip — forcing the contributor to update Amendment C consciously.
    """
    src = Path(
        "src/spectramr/infrastructure/orchestration/campaign_evaluator.py"
    ).read_text(encoding="utf-8")
    assert "pickle.load(f)" in src, (
        "If pkl-manifest inline-loading was removed, Amendment C is "
        "obsolete — update the data-layer unification plan."
    )
