"""Shared contrast-conditioning helper for field-FiLM models.

The helper centralises the ``sequence_features`` construction that every
field-aware FiLM generator needs when it opts into multi-contrast
conditioning: widen the :class:`FieldFiLMBlock` ``sequence_dim`` by
``num_contrasts`` and concatenate a per-sample contrast one-hot. It enforces
the two always-on invariants (CLAUDE.md #15 raise-on-missing, #9
raise-on-out-of-range) in one place instead of inlining them per model.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)

# --- contrast_sequence_dim ------------------------------------------------


def test_seq_dim_no_base_disabled_is_scalar_placeholder() -> None:
    # No intrinsic sequence features + conditioning off => the FieldFiLMBlock
    # sequence_dim>=1 invariant is met by a single zero placeholder.
    assert contrast_sequence_dim(0, num_contrasts=3, enabled=False) == 1


def test_seq_dim_no_base_enabled_is_num_contrasts() -> None:
    # No base features + conditioning on => the one-hot REPLACES the placeholder
    # (matches FieldVelocityUNet's original seq_dim = num_contrasts).
    assert contrast_sequence_dim(0, num_contrasts=3, enabled=True) == 3


def test_seq_dim_with_base_disabled_is_base() -> None:
    assert contrast_sequence_dim(2, num_contrasts=3, enabled=False) == 2


def test_seq_dim_with_base_enabled_appends() -> None:
    # Real base features (e.g. a time embedding) are PRESERVED; contrast appends.
    assert contrast_sequence_dim(2, num_contrasts=3, enabled=True) == 5


# --- build_contrast_sequence: disabled path -------------------------------


def test_build_disabled_no_base_returns_zero_placeholder() -> None:
    seq = build_contrast_sequence(
        None, None, num_contrasts=3, enabled=False,
        batch_size=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert seq.shape == (4, 1)
    assert torch.equal(seq, torch.zeros(4, 1))


def test_build_disabled_with_base_is_identity() -> None:
    base = torch.randn(4, 2)
    seq = build_contrast_sequence(
        base, torch.tensor([0, 1, 2, 0]), num_contrasts=3, enabled=False,
        batch_size=4, device=base.device, dtype=base.dtype,
    )
    assert seq is base  # untouched passthrough


# --- build_contrast_sequence: enabled path --------------------------------


def test_build_enabled_no_base_is_one_hot() -> None:
    cid = torch.tensor([0, 2, 1])
    seq = build_contrast_sequence(
        None, cid, num_contrasts=3, enabled=True,
        batch_size=3, device=cid.device, dtype=torch.float32,
    )
    assert seq.shape == (3, 3)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    assert torch.equal(seq, expected)


def test_build_enabled_with_base_concatenates() -> None:
    base = torch.randn(2, 4)  # e.g. a 4-dim time embedding
    cid = torch.tensor([1, 0])
    seq = build_contrast_sequence(
        base, cid, num_contrasts=3, enabled=True,
        batch_size=2, device=base.device, dtype=base.dtype,
    )
    assert seq.shape == (2, 7)  # 4 base + 3 contrast
    assert torch.equal(seq[:, :4], base)  # base preserved
    assert torch.equal(
        seq[:, 4:], torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )


def test_build_enabled_missing_contrast_id_raises() -> None:
    # #15: a wired knob with no value must fail loud, not silently drop.
    with pytest.raises(ValueError, match="contrast_id"):
        build_contrast_sequence(
            None, None, num_contrasts=3, enabled=True,
            batch_size=2, device=torch.device("cpu"), dtype=torch.float32,
        )


def test_build_enabled_out_of_range_id_raises() -> None:
    # #9: no silent clamp — an id >= num_contrasts must raise (via one_hot).
    with pytest.raises(RuntimeError):
        build_contrast_sequence(
            None, torch.tensor([0, 5]), num_contrasts=3, enabled=True,
            batch_size=2, device=torch.device("cpu"), dtype=torch.float32,
        )


def test_build_enabled_respects_dtype() -> None:
    cid = torch.tensor([0, 1])
    seq = build_contrast_sequence(
        None, cid, num_contrasts=2, enabled=True,
        batch_size=2, device=cid.device, dtype=torch.float64,
    )
    assert seq.dtype == torch.float64
