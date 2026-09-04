"""``UniversalMRIDataset`` must emit a ``contrast_idx`` tensor on the paired path.

The ``nifti_paired`` loader (the ULF→HF cohorts: geomamba_ulf, the LDM/diffusion
arms, ulf_physics) preserves each record's ``contrast`` *string* but historically
emitted no ``contrast_idx``. ``GeoMambaUNet.forward`` and the latent-diffusion
generator both fall back to contrast 0 when ``contrast_id``/``contrast_idx`` is
absent, so the advertised FiLM / contrast-guidance conditioning silently collapsed
to a single token for every contrast (pitfall #16 facade; the ``abl_no_film``
ablation had no measurable delta — pitfall #17).

This pins the fix: when the dataset is configured with a ``contrast_map`` (the
per-arm ``data.multi_contrast.contrast_map``), each sample carries
``contrast_idx = contrast_map[record['contrast']]`` as a long tensor. An
out-of-vocabulary contrast RAISES (no silent fallback, pitfall #9). Arms without a
``contrast_map`` are unchanged — no ``contrast_idx`` key, no regression.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from spectramr.data.datasets.universal_dataset import (  # noqa: E402
    UniversalMRIDataset,
    resolve_contrast_idx,
)

_MAP = {"T1w": 0, "T2w": 1, "FLAIR": 2}


# --- pure helper ------------------------------------------------------------


def test_resolve_maps_known_contrast_to_index():
    idx = resolve_contrast_idx({"contrast": "FLAIR"}, _MAP)
    assert isinstance(idx, torch.Tensor)
    assert idx.dtype == torch.long
    assert int(idx) == 2


def test_resolve_returns_none_when_no_map_configured():
    # Feature off: contrast-agnostic arms (ulf_physics, unpaired_ulf) are unchanged.
    assert resolve_contrast_idx({"contrast": "T1w"}, None) is None


def test_resolve_raises_on_out_of_vocabulary_contrast():
    # ADC is a paired contrast in the fixed manifest; a 3-contrast anatomical arm
    # that does not map it must FAIL LOUD rather than silently stamp contrast 0.
    with pytest.raises((KeyError, ValueError)):
        resolve_contrast_idx({"contrast": "ADC"}, _MAP)


def test_resolve_raises_when_map_set_but_record_lacks_contrast():
    with pytest.raises((KeyError, ValueError)):
        resolve_contrast_idx({"file_id": "sub-0001_T1w"}, _MAP)


# --- dataset emission -------------------------------------------------------


def _stub_dataset(monkeypatch, index, contrast_map):
    ds = UniversalMRIDataset(
        index=index,
        io_strategy="nifti",
        paired_io_strategy="nifti",
        load_sensitivity=False,
        contrast_map=contrast_map,
    )

    def _fake_build(_record):
        return tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 4)))

    monkeypatch.setattr(ds.subject_builder, "build", _fake_build)
    return ds


def test_getitem_stamps_contrast_idx_when_map_configured(monkeypatch):
    ds = _stub_dataset(
        monkeypatch,
        [{"primary_path": "/x/sub-0001_T2w.nii.gz", "contrast": "T2w"}],
        _MAP,
    )
    subject = ds[0]
    assert "contrast_idx" in subject
    assert int(subject["contrast_idx"]) == 1


def test_getitem_also_stamps_contrast_id_alias(monkeypatch):
    # Contrast-conditioned strategies ported from mrixfields (field_cold_diffusion,
    # generative_refiner) read batch['contrast_id']; the paired path must emit that
    # alias (same value) so they condition instead of defaulting to contrast 0.
    ds = _stub_dataset(
        monkeypatch,
        [{"primary_path": "/x/sub-0001_FLAIR.nii.gz", "contrast": "FLAIR"}],
        _MAP,
    )
    subject = ds[0]
    assert "contrast_id" in subject
    assert int(subject["contrast_id"]) == 2
    assert int(subject["contrast_id"]) == int(subject["contrast_idx"])


def test_getitem_omits_contrast_id_alias_without_map(monkeypatch):
    ds = _stub_dataset(
        monkeypatch,
        [{"primary_path": "/x/sub-0001_T2w.nii.gz", "contrast": "T2w"}],
        None,
    )
    subject = ds[0]
    assert "contrast_id" not in subject


def test_getitem_omits_contrast_idx_without_map(monkeypatch):
    ds = _stub_dataset(
        monkeypatch,
        [{"primary_path": "/x/sub-0001_T2w.nii.gz", "contrast": "T2w"}],
        None,
    )
    subject = ds[0]
    assert "contrast_idx" not in subject
