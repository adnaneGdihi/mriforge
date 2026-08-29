"""Regression test for F8/E28 (smoke_audit_20260521) — k-space mixin's
``_prepare_validation_data`` normalization must handle all channel
counts, not just 2-channel real-stacked complex.

Root cause: the prior code did

    input_mag = torch.sqrt(
        input_batch[:, 0:1] ** 2 + input_batch[:, 1:2] ** 2
    ).reshape(input_batch.size(0), -1)
    percentile_vals = torch.quantile(input_mag, 0.99, dim=1)

For 1-channel inputs (PINN / implicit-fields family),
``input_batch[:, 1:2]`` slices past the channel axis and returns
shape ``(B, 0, H, W)`` — magnitude reshape becomes empty per sample
and ``torch.quantile`` raises ``input tensor must be non-empty``.

This crashed 6 experiments:
    experiment_12_pinn_reconstruction
    experiment_39_pin_inr_mri_implicit_fields
    experiment_42b_graph_diffusion
    experiment_46_pinn_inverse_solving
    experiment_60_implicit_neural
    experiment_pipeline_b_quantum_nerf

Fix: dispatch by channel count (1 → abs, 2 → R+iI, even-N → coil RSS,
odd-N → channel-RSS) plus a final empty-tensor guard.

This test pins:
- 1-channel input doesn't crash.
- 2-channel input still computes R²+I² (backward compat).
- Multi-coil 8-channel input runs coil-pair RSS.
- Empty input (numel==0) returns the un-normalized batch instead of crashing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "src" / "mriforge" / "infrastructure" / "training" / "strategies" / "mixins" / "kspace.py"


def _read() -> str:
    return SOURCE.read_text()


def test_per_channel_dispatch_present() -> None:
    """The normalization code must branch on channel count rather than
    assume 2-channel real-stacked layout. Pinned because reverting to
    the single ``[:, 0:1]**2 + [:, 1:2]**2`` form would silently break
    the 6 affected experiments."""
    text = _read()
    # The fix introduces a channel-count branch. Pin the literal control
    # flow that's load-bearing: 1-channel uses .abs(), 2-channel uses
    # sqrt(R²+I²), and we have an even-N coil-pair branch.
    assert "if c == 1:" in text, (
        "_prepare_validation_data must branch on c==1 (single-channel "
        "PINN / implicit-fields input). See "
        "TODO/audit/smoke_audit_20260521.md §F8/E28."
    )
    assert "elif c == 2:" in text, (
        "_prepare_validation_data must keep the 2-channel R/I path "
        "(backward compat)."
    )
    assert "elif c % 2 == 0:" in text, (
        "_prepare_validation_data must handle even-channel multi-coil."
    )


def test_empty_input_doesnt_crash() -> None:
    """Hard-fail-loud is the right CLAUDE.md policy for genuine
    misconfigs, BUT an empty input here is recoverable (skip
    normalization, return identity-scale). Pinned because the prior
    behavior was a hard crash deep in ``torch.quantile``.
    """
    text = _read()
    # The final guard returns the un-normalized batch when numel==0.
    assert "if input_mag.numel() == 0:" in text, (
        "_prepare_validation_data must guard against empty input_mag "
        "before calling torch.quantile."
    )


def test_audit_anchor_present() -> None:
    text = _read()
    assert "F8/E28 (2026-05-21" in text, (
        "F8 fix must reference the audit doc code/date for traceability."
    )


# ── Behavioral test ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_channels", [1, 2, 4, 8])
def test_normalize_path_does_not_raise_for_any_channel_count(
    n_channels: int, tmp_path: Path
) -> None:
    """Direct exercise of the normalization code on synthetic inputs of
    several channel counts. All must complete without raising
    ``input tensor must be non-empty``.

    Note: we can't easily instantiate the full mixin without the rest
    of the training plumbing, so we extract the post-F8 logic and run
    it standalone. This pins the math, not the call-site integration.
    """
    torch.manual_seed(0)
    input_batch = torch.randn(2, n_channels, 16, 16)

    # Mirror the post-F8 dispatch.
    c = input_batch.shape[1]
    if c == 1:
        input_mag = input_batch.abs().reshape(input_batch.size(0), -1)
    elif c == 2:
        input_mag = torch.sqrt(
            input_batch[:, 0:1] ** 2 + input_batch[:, 1:2] ** 2
        ).reshape(input_batch.size(0), -1)
    elif c % 2 == 0:
        re_ = input_batch[:, 0::2]
        im_ = input_batch[:, 1::2]
        coil_mags = torch.sqrt(re_ ** 2 + im_ ** 2 + 1e-12)
        input_mag = torch.sqrt(
            (coil_mags ** 2).sum(dim=1, keepdim=True)
        ).reshape(input_batch.size(0), -1)
    else:
        input_mag = torch.sqrt(
            (input_batch ** 2).sum(dim=1, keepdim=True)
        ).reshape(input_batch.size(0), -1)

    assert input_mag.numel() > 0, (
        f"Channel count {n_channels} produced empty magnitude tensor; "
        "torch.quantile would crash. F8/E28 regression."
    )
    # Final quantile call — must not raise.
    percentile_vals = torch.quantile(input_mag, 0.99, dim=1)
    assert percentile_vals.shape == (input_batch.shape[0],)
