"""Regression tests for 2026-05-31 audit MUST-FIX strategies (referenced by
live experiments/campaigns):

* multi_contrast_contrastive — InfoNCE positive labels must be offset by
  ``rank*B`` for the DDP-gathered tensor (was correct only on rank 0).
* synthetic_pathology_aug — the composited lesion input/target must reach the
  parent (was forwarding the original, discarding the augmentation).
* qsm_pipeline — compute_field_map must be called with its real 4-arg
  signature (two single-echo phase maps + two scalar TEs), not 2 args.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.multi_contrast_contrastive_strategy import (
    MultiContrastContrastiveStrategy,
)
from spectramr.infrastructure.training.strategies.qsm_pipeline_strategy import (
    QSMPipelineStrategy,
)
import spectramr.infrastructure.training.strategies.qsm_pipeline_strategy as qsm_mod
from spectramr.infrastructure.training.strategies.synthetic_pathology_aug_strategy import (
    SyntheticPathologyAugStrategy,
)


def test_info_nce_positive_offset_for_gathered_block():
    B, D = 3, 8
    z1 = torch.randn(B, D)
    # z2 has z1's positives at rows B:2B (simulating local block at rank 1).
    z2 = torch.cat([torch.randn(B, D), z1.clone()], dim=0)
    loss_correct = MultiContrastContrastiveStrategy._info_nce(z1, z2, 0.07, pos_offset=B)
    loss_wrong = MultiContrastContrastiveStrategy._info_nce(z1, z2, 0.07, pos_offset=0)
    # With the right offset the positive is the self-match -> much lower loss.
    assert loss_correct < loss_wrong


def test_synthetic_pathology_forwards_composited_tensors(monkeypatch):
    s = object.__new__(SyntheticPathologyAugStrategy)
    s.p_augment = 1.0  # always apply -> patched == lesion
    # deterministic sampler: lesion = ones, mask = ones
    monkeypatch.setattr(
        SyntheticPathologyAugStrategy,
        "_ensure_sampler",
        lambda self, device: (lambda x: (torch.ones_like(x), torch.ones_like(x[:, :1]))),
    )
    captured = {}
    parent = type(s).__mro__[1]
    monkeypatch.setattr(
        parent,
        "_compute_losses_impl",
        lambda self, input_batch=None, target_batch=None, epoch=0, **kw: captured.update(
            inp=input_batch
        )
        or {"loss_total": torch.zeros(())},
        raising=False,
    )
    batch = {"input": torch.zeros(2, 1, 4, 4), "target": torch.zeros(2, 1, 4, 4)}
    s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)
    # gate=1 (p_augment=1) so the composite == lesion == ones, NOT the zeros input.
    assert torch.allclose(captured["inp"], torch.ones(2, 1, 4, 4))


class _Stop(Exception):
    pass


def test_qsm_compute_field_map_called_with_four_args(monkeypatch):
    rec = {}

    def _fake_cfm(phase_te1, phase_te2, te1, te2):
        rec.update(p1=phase_te1, p2=phase_te2, te1=te1, te2=te2)
        raise _Stop()  # halt run_pipeline right after the field-map step

    monkeypatch.setattr(qsm_mod, "compute_field_map", _fake_cfm)
    s = object.__new__(QSMPipelineStrategy)
    batch = {
        "phase_echoes": torch.randn(2, 2, 4, 4, 4),  # [B, n_echo, D, H, W]
        "TE": torch.tensor([0.005, 0.010]),
    }
    with pytest.raises(_Stop):
        s.run_pipeline(batch)
    assert rec["te1"] == pytest.approx(0.005) and rec["te2"] == pytest.approx(0.010)
    assert rec["p1"].shape[1] == 1 and rec["p2"].shape[1] == 1  # single-echo maps
