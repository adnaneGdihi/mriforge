"""Oracle tests for end-to-end acquisition co-design (A-7.1 E2E-AcqCo)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.acquisition_codesign import (
    LearnableSamplingMask,
    codesign_sampling,
)


def _low_freq_dataset(n: int = 32, w: int = 16) -> torch.Tensor:
    # Rows whose energy concentrates in low frequencies -> the optimal sampling
    # budget should migrate to the low-frequency columns.
    torch.manual_seed(0)
    x_k = torch.zeros(n, w, dtype=torch.complex64)
    x_k[:, :4] = torch.randn(n, 4, dtype=torch.complex64) * 3.0  # strong low-freq
    x_k[:, 4:] = torch.randn(n, w - 4, dtype=torch.complex64) * 0.1  # weak high-freq
    return torch.fft.ifft(x_k, dim=-1).real


# --------------------------------------------------------------------------- #
# Mask module
# --------------------------------------------------------------------------- #


def test_soft_mask_in_unit_interval() -> None:
    m = LearnableSamplingMask(16, budget=4)
    sm = m.soft_mask()
    assert sm.shape == (16,)
    assert float(sm.min()) >= 0.0 and float(sm.max()) <= 1.0


def test_hard_mask_respects_budget() -> None:
    m = LearnableSamplingMask(16, budget=5)
    assert int(m.hard_mask().sum()) == 5


def test_budget_must_be_valid() -> None:
    with pytest.raises(ValueError, match="budget"):
        LearnableSamplingMask(8, budget=0)
    with pytest.raises(ValueError, match="budget"):
        LearnableSamplingMask(8, budget=9)


# --------------------------------------------------------------------------- #
# THE co-design oracles: beats matched-budget random; migrates to signal
# --------------------------------------------------------------------------- #


def test_codesign_beats_matched_budget_random() -> None:
    images = _low_freq_dataset()
    out = codesign_sampling(images, budget=4, n_steps=300, lr=0.1)
    assert out["learned_mse"] < out["random_mse"]


def test_codesign_concentrates_budget_on_signal_lines() -> None:
    # The learned budget-4 mask should land on the HIGHEST-ENERGY columns. (A real
    # image's spectrum is Hermitian-symmetric, so the high-energy lines come in
    # conjugate pairs at both ends, not only {0,1,2,3} — compute them directly.)
    images = _low_freq_dataset()
    energy = (torch.fft.fft(images, dim=-1).abs() ** 2).mean(0)
    top_energy = set(torch.topk(energy, 6).indices.tolist())  # generous top set
    out = codesign_sampling(images, budget=4, n_steps=300, lr=0.1)
    chosen = set(torch.nonzero(out["mask"]).reshape(-1).tolist())
    assert len(chosen & top_energy) >= 3  # most of the budget lands on signal lines


def test_codesign_reduces_mse() -> None:
    images = _low_freq_dataset()
    out = codesign_sampling(images, budget=4, n_steps=200, lr=0.1)
    hist = out["mse_history"]
    assert hist[-1] < hist[0]  # the co-design loop actually reduces recon error


def test_returns_expected_fields() -> None:
    out = codesign_sampling(_low_freq_dataset(), budget=4, n_steps=20)
    assert {"mask", "gain", "mse_history", "learned_mse", "random_mse"} <= set(out)
    assert int(out["mask"].sum()) == 4
