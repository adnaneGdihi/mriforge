"""Tests for the frame_coherence loss (A-6.5 SCO-Frame, the one-knob penalty)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.tight_frame_learner import TightFrameLearner
from mriforge.models.losses.frame_coherence_loss import (
    FrameCoherenceLoss,
    gram_offdiagonal_penalty,
)
from mriforge.models.losses.registry import create_loss, list_available


def test_reads_frame_module() -> None:
    m = TightFrameLearner(num_atoms=8, kernel_size=3)
    loss = FrameCoherenceLoss()
    assert float(loss(frame=m).detach()) == pytest.approx(float(m.coherence_penalty().detach()), rel=1e-6)


def test_reads_atoms_tensor() -> None:
    atoms = torch.randn(6, 9)
    assert float(FrameCoherenceLoss()(atoms=atoms).detach()) == pytest.approx(
        float(gram_offdiagonal_penalty(atoms).detach()), rel=1e-6
    )


def test_orthonormal_atoms_zero() -> None:
    q = torch.linalg.qr(torch.randn(9, 4))[0].t()  # [4, 9] orthonormal rows
    assert float(FrameCoherenceLoss()(atoms=q).detach()) == pytest.approx(0.0, abs=1e-5)


def test_differentiable_in_atoms() -> None:
    atoms = torch.randn(6, 9, requires_grad=True)
    pen = FrameCoherenceLoss()(atoms=atoms)
    pen.backward()
    assert atoms.grad is not None and torch.isfinite(atoms.grad).all()


def test_raises_without_frame_or_atoms() -> None:
    with pytest.raises(ValueError, match="PARAMETER penalty"):
        FrameCoherenceLoss()(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))


def test_gram_penalty_accepts_conv_weight_shape() -> None:
    # Contract: axis 0 = atoms; trailing axes flatten. A raw conv weight [m,C,k,k]
    # is accepted and equals the penalty on its [m, C*k*k] flatten.
    w = torch.randn(5, 1, 3, 3)
    assert float(gram_offdiagonal_penalty(w).detach()) == pytest.approx(
        float(gram_offdiagonal_penalty(w.reshape(5, -1)).detach()), rel=1e-6
    )


def test_gram_penalty_raises_on_rank_below_2() -> None:
    # A rank-<2 tensor has no atom/feature split -> raise, not silently reshape (#9).
    with pytest.raises(ValueError, match=">=2 dims"):
        gram_offdiagonal_penalty(torch.randn(9))


def test_accepts_positional_frame() -> None:
    m = TightFrameLearner(num_atoms=8, kernel_size=3)
    assert float(FrameCoherenceLoss()(m).detach()) == pytest.approx(
        float(m.coherence_penalty().detach()), rel=1e-6
    )


def test_registered_and_constructible() -> None:
    assert "frame_coherence" in list_available()
    assert isinstance(create_loss("mutual_coherence_penalty"), FrameCoherenceLoss)  # via alias
