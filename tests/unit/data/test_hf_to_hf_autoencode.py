"""End-to-end acceptance: bidirectional_mode=hf_to_hf autoencodes HF (input≡target).

Drives the real ``_create_nifti_universal`` → ``UniversalMRIDataset`` path over
two tiny NIfTI files. Proves the stage-1 fix: with ``hf_to_hf`` the model INPUT
and reconstruction TARGET are the SAME HF tensor (a genuine autoencoder), not the
ULF→HF translation that the former vae.py fallback masked.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mriforge.data.builders.dataset_instantiator import DatasetInstantiator
from tests.utils.data_config_stub import DataConfigStub


def _write_nifti(path: Path, ramp: str) -> None:
    """Write an 8x8x2 volume whose in-plane values ramp up or down along H, so
    minmax normalization (monotonic) preserves which file it came from."""
    h = np.arange(8, dtype=np.float32)
    col = h if ramp == "up" else h[::-1]
    vol = np.broadcast_to(col[:, None, None], (8, 8, 2)).astype(np.float32)
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(path))


def test_hf_to_hf_input_equals_target_and_is_hf(tmp_path) -> None:
    ulf = tmp_path / "sub-01_ulf.nii.gz"
    hf = tmp_path / "sub-01_hf.nii.gz"
    _write_nifti(ulf, ramp="down")  # ULF: descending ramp
    _write_nifti(hf, ramp="up")  # HF:  ascending ramp
    idx = [{"primary_path": str(ulf), "target_path": str(hf), "file_id": "sub-01"}]


    cfg = DataConfigStub(dataset_type="nifti_paired", bidirectional_mode="hf_to_hf")
    train_ds, _ = DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)

    subject = train_ds[0]
    inp = subject["input"].data.float()
    tgt = subject["target"].data.float()

    # Acceptance: autoencoder objective — input and target are the SAME tensor.
    assert torch.equal(inp, tgt)

    # ...and it is the HF file (ascending ramp along H), not the ULF one
    # (descending). Subject layout is [C, H, W, D]; average out C, W and D to
    # get the per-row H profile. minmax normalization is monotonic → order holds.
    assert inp.dim() == 4 and inp.shape[1] == 8, f"unexpected layout {tuple(inp.shape)}"
    h_profile = inp.mean(dim=(0, 2, 3))  # → [H]
    assert h_profile[0] < h_profile[-1], "input must be the HF ascending-ramp file, not ULF"
