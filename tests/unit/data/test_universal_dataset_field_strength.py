"""``UniversalMRIDataset`` must emit ``field_strength_target`` on the paired path.

The ``nifti_paired`` loader carries no per-sample field key. The field-conditioned
ULF→HF restoration strategies ported from the MRIxFields2026 task2 cohort
(``ulf_dps``, ``monotone_field``, ``field_conditioned_inr``, ``generative_refiner``,
``ulf_redegrad_tta``) read ``batch['field_strength_target']`` directly and RAISE on
its absence (their own errors say "set data.expose_field_strength_target"). On the
``mrixfields`` dataset that key is stamped per-record; on ``nifti_paired`` it was
never emitted, so those strategies could not run on the paired-ULF data at all.

The paired-ULF task is a FIXED 64mT→3T translation, so the target field is a
constant (3.0 T) coming from config (``data.mrixfields_target_field``), not the
manifest. This pins the fix: when the dataset is configured with a
``target_field_strength`` (threaded from ``mrixfields_target_field`` gated on
``expose_field_strength_target``), each sample carries
``field_strength_target`` as a scalar float tensor — the exact analog of the
``contrast_idx`` fix. Arms that leave it unset are unchanged (no key, no
regression). A non-positive value RAISES (no silent fabrication, pitfall #9).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from spectramr.data.datasets.universal_dataset import (  # noqa: E402
    UniversalMRIDataset,
    resolve_field_strength,
)

# --- pure helper ------------------------------------------------------------


def test_resolve_maps_value_to_scalar_float_tensor():
    fs = resolve_field_strength(3.0)
    assert isinstance(fs, torch.Tensor)
    assert fs.dtype == torch.float32
    assert fs.ndim == 0
    assert float(fs) == pytest.approx(3.0)


def test_resolve_returns_none_when_unset():
    # Feature off: field-agnostic arms (geomamba baselines, unpaired_ulf) unchanged.
    assert resolve_field_strength(None) is None


def test_resolve_raises_on_non_positive_field():
    # A field strength must be positive Tesla; refuse to fabricate a 0 / negative.
    with pytest.raises((ValueError, AssertionError)):
        resolve_field_strength(0.0)
    with pytest.raises((ValueError, AssertionError)):
        resolve_field_strength(-1.0)


# --- dataset emission -------------------------------------------------------


def _stub_dataset(monkeypatch, target_field_strength):
    ds = UniversalMRIDataset(
        index=[{"primary_path": "/x/sub-0001_T1w_ulf.nii.gz", "contrast": "T1w"}],
        io_strategy="nifti",
        paired_io_strategy="nifti",
        load_sensitivity=False,
        target_field_strength=target_field_strength,
    )

    def _fake_build(_record):
        return tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 4)))

    monkeypatch.setattr(ds.subject_builder, "build", _fake_build)
    return ds


def test_getitem_stamps_field_strength_target_when_configured(monkeypatch):
    ds = _stub_dataset(monkeypatch, 3.0)
    subject = ds[0]
    assert "field_strength_target" in subject
    fs = subject["field_strength_target"]
    assert isinstance(fs, torch.Tensor)
    assert float(fs) == pytest.approx(3.0)


def test_getitem_omits_field_strength_target_without_config(monkeypatch):
    ds = _stub_dataset(monkeypatch, None)
    subject = ds[0]
    assert "field_strength_target" not in subject
