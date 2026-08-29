"""Oracle tests for operator-estimating blind-motion recon (A-4.1 OER)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import ifft2c
from mriforge.infrastructure.physics.blind_motion_recon import (
    estimate_blind_motion,
    gradient_entropy,
    simulate_segmented_motion,
)


def _phantom(h: int = 24, w: int = 24) -> torch.Tensor:
    # A sharp structured phantom (a bright square) -> clear ghosting under motion.
    x = torch.zeros(h, w)
    x[6:12, 8:16] = 1.0
    x[14:18, 5:10] = 0.6
    return x


def _two_segments(h: int, w: int) -> list[torch.Tensor]:
    # Disjoint k-space halves (top rows vs bottom rows), each a motion state.
    m0 = torch.zeros(h, w)
    m1 = torch.zeros(h, w)
    m0[: h // 2] = 1.0
    m1[h // 2 :] = 1.0
    return [m0, m1]


# --------------------------------------------------------------------------- #
# Autofocus metric: sharp < ghosted
# --------------------------------------------------------------------------- #


def test_gradient_entropy_lower_for_sharp_than_ghosted() -> None:
    clean = _phantom()
    masks = _two_segments(24, 24)
    corrupt_k = simulate_segmented_motion(clean, masks, [(0, 0), (0, 5)])  # seg1 shifted
    ghosted = ifft2c(corrupt_k)
    assert gradient_entropy(clean.to(torch.complex64)) < gradient_entropy(ghosted)


# --------------------------------------------------------------------------- #
# THE OER oracle: recover the relative motion + de-ghost the image
# --------------------------------------------------------------------------- #


def test_recovers_relative_translation_and_deghosts() -> None:
    clean = _phantom()
    masks = _two_segments(24, 24)
    true_shift = (0, 3)  # segment 1 shifted by 3 px in x relative to segment 0
    corrupt_k = simulate_segmented_motion(clean, masks, [(0, 0), true_shift])

    out = estimate_blind_motion(corrupt_k, masks, search_radius=4, n_outer=3)

    # (1) recovers segment 1's translation relative to the segment-0 reference.
    assert out["translations"][0] == (0, 0)
    assert out["translations"][1] == true_shift

    # (2) the de-ghosted reconstruction matches the clean phantom.
    recon = out["image"].abs()
    assert float((recon - clean).abs().max()) < 1e-3

    # (3) the corrected image is sharper (lower entropy) than the uncorrected.
    uncorrected = ifft2c(corrupt_k)
    assert gradient_entropy(out["image"]) < gradient_entropy(uncorrected)


def test_entropy_history_non_increasing() -> None:
    clean = _phantom()
    masks = _two_segments(24, 24)
    corrupt_k = simulate_segmented_motion(clean, masks, [(0, 0), (2, 0)])
    out = estimate_blind_motion(corrupt_k, masks, search_radius=4, n_outer=3)
    hist = out["entropy_history"]
    assert all(hist[i + 1] <= hist[i] + 1e-6 for i in range(len(hist) - 1))


def test_zero_motion_recovers_identity() -> None:
    clean = _phantom()
    masks = _two_segments(24, 24)
    corrupt_k = simulate_segmented_motion(clean, masks, [(0, 0), (0, 0)])  # no motion
    out = estimate_blind_motion(corrupt_k, masks, search_radius=3, n_outer=2)
    assert out["translations"][1] == (0, 0)
    assert float((out["image"].abs() - clean).abs().max()) < 1e-3


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="equal length"):
        simulate_segmented_motion(_phantom(), _two_segments(24, 24), [(0, 0)])
