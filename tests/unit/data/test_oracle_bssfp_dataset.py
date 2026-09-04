"""oracle_bssfp dataset — phase-cycled bSSFP stack + real Hz B0 (vf_29 Path A).

Loads the EXTRACTED form (real-interleaved phase-cycled stack NIfTI [H,W,2N] +
Hz B0 NIfTI [H,W]) via the SSOT NiftiStrategy, emitting the stack as the model
input + the real B0 as b0_map for the field-scoring seam. The raw oracle_bssfp is
cluster-only Siemens TWIX; a reader now exists (``spectramr.data.twix`` /
``TwixStrategy``) to decode the raw k-space, but this loader consumes the
post-extraction NIfTI form and RAISES clearly when it is absent (no speculative
no-op).
"""
from __future__ import annotations

import numpy as np
import pytest

from spectramr.data.datasets.oracle_bssfp_dataset import (
    OracleBssfpDataset,
    build_oracle_bssfp_index,
)


def _nii(path, arr):
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), np.eye(4)), str(path))


def _make_oracle_like(root, n_cycles=4, h=8, w=8):
    # stack: real-interleaved [H, W, 2N] (N real + N imag channels, NIfTI last-axis)
    _nii(root / "subj1" / "bssfp_stack.nii", np.random.rand(h, w, 2 * n_cycles))
    _nii(root / "subj2" / "bssfp_stack.nii", np.random.rand(h, w, 2 * n_cycles))
    _nii(root / "maps" / "b0_Hz.nii", np.random.rand(h, w) * 40.0)


def test_build_index_pairs_each_stack_with_the_b0(tmp_path):
    _make_oracle_like(tmp_path)
    recs = build_oracle_bssfp_index(
        tmp_path, stack_glob="**/bssfp_stack.nii", b0_map="maps/b0_Hz.nii"
    )
    assert len(recs) == 2
    assert all(r["b0_map"].endswith("b0_Hz.nii") for r in recs)


def test_missing_b0_raises(tmp_path):
    _make_oracle_like(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_oracle_bssfp_index(
            tmp_path, stack_glob="**/bssfp_stack.nii", b0_map="maps/absent.nii"
        )


def test_no_stack_match_raises(tmp_path):
    _make_oracle_like(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_oracle_bssfp_index(
            tmp_path, stack_glob="**/does_not_exist.nii", b0_map="maps/b0_Hz.nii"
        )


def test_getitem_emits_stack_channels_and_b0(tmp_path):
    import torchio as tio

    _make_oracle_like(tmp_path, n_cycles=4, h=8, w=8)
    recs = build_oracle_bssfp_index(
        tmp_path, stack_glob="**/bssfp_stack.nii", b0_map="maps/b0_Hz.nii"
    )
    subj = OracleBssfpDataset(recs)[0]
    assert isinstance(subj, tio.Subject)
    # stack reshaped [H,W,2N] → [2N, H, W] channels-first (the model's input)
    assert subj["input"].tensor.shape[0] == 8  # 2N = 2*4
    assert "b0_map" in subj
    assert subj["b0_map"].tensor.shape[-2:] == subj["input"].tensor.shape[-2:]


def test_dry_iter_returns_one_stub_subject_per_record_without_voxel_read(tmp_path):
    """``dry_iter`` must yield cheap Subject shells for ``tio.Queue`` length counting.

    Regression for the cluster crash ``'OracleBssfpDataset' object has no attribute
    'dry_iter'`` (vf_29 Path-A oracle arms, diagnostics 2026-06-26). The Queue only
    needs an indexable Sequence of the right length; the stubs must NOT read the
    NIfTI payload (1x1x1x1 tensors).
    """
    import torchio as tio

    _make_oracle_like(tmp_path, n_cycles=4, h=8, w=8)
    recs = build_oracle_bssfp_index(
        tmp_path, stack_glob="**/bssfp_stack.nii", b0_map="maps/b0_Hz.nii"
    )
    ds = OracleBssfpDataset(recs)
    stubs = ds.dry_iter()
    assert len(stubs) == len(recs) == 2
    for s in stubs:
        assert isinstance(s, tio.Subject)
        # 1x1x1x1 stub — NOT the [2N, H, W] real payload
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)


def test_schema_defaults_and_allow_list():
    from spectramr.config.schemas.data import DataConfigSchema, OracleBssfpConfigSchema

    assert DataConfigSchema().oracle_bssfp.enabled is False
    assert (
        DataConfigSchema(dataset_type="oracle_bssfp").dataset_type == "oracle_bssfp"
    )
    # enabled without a b0_map must raise (the run grades against it)
    with pytest.raises(Exception):
        OracleBssfpConfigSchema(enabled=True)
    c = OracleBssfpConfigSchema(enabled=True, b0_map="maps/b0_Hz.nii")
    assert c.stack_glob  # has a default glob


def test_odd_channel_stack_raises(tmp_path):
    # a stack whose channel axis is not 2N (real-interleaved) must fail loudly.
    _nii(tmp_path / "s" / "bad_stack.nii", np.random.rand(8, 8, 5))  # 5 = odd
    _nii(tmp_path / "maps" / "b0_Hz.nii", np.random.rand(8, 8))
    recs = build_oracle_bssfp_index(
        tmp_path, stack_glob="**/bad_stack.nii", b0_map="maps/b0_Hz.nii"
    )
    with pytest.raises(ValueError, match="real-interleaved"):
        OracleBssfpDataset(recs)[0]


def test_twix_reader_referenced_by_docstring_actually_exists():
    """The raw-TWIX reader the loader's docs point at must be real + registered.

    Guards against the docstring referencing a reader that was never wired (an
    anti-facade check — pitfall #16). The loader itself consumes NIfTI, but the
    cluster TWIX->NIfTI conversion relies on this reader existing.
    """
    from spectramr.data.io_strategies import IOStrategyFactory, TwixStrategy
    from spectramr.data.twix import parse_twix_header, read_twix

    assert callable(parse_twix_header) and callable(read_twix)
    assert isinstance(IOStrategyFactory.get("twix"), TwixStrategy)
