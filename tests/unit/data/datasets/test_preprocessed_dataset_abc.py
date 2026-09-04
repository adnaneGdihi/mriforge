"""Regression test for the Dataset ABC inheritance of ``PreprocessedMRIDataset``.

Locks the fix from ``TODO/audit/16_data_layer.md`` F5. Lives in its own
file (separate from ``tests/unit/test_preprocessed_dataset.py``) so it
runs even when the parent test file is skipped because optional deps
(e.g. ``hypothesis``) are missing.
"""

from __future__ import annotations

import torch.utils.data as _torch_data

from spectramr.data.datasets.preprocessed_dataset import PreprocessedMRIDataset


def test_preprocessed_mri_dataset_inherits_from_torch_dataset() -> None:
    """``PreprocessedMRIDataset`` must subclass ``torch.utils.data.Dataset``.

    Without this, PyTorch's ``DataLoader`` silently mis-handles batch
    construction for some sampler combinations and does not flag a
    missing ``__getitem__`` contract at construction time.
    """
    assert issubclass(PreprocessedMRIDataset, _torch_data.Dataset)


def test_preprocessed_mri_dataset_declares_getitem_and_len() -> None:
    """The dataset honours the canonical map-style ``Dataset`` contract."""
    assert hasattr(PreprocessedMRIDataset, "__getitem__")
    assert hasattr(PreprocessedMRIDataset, "__len__")
