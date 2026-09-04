r"""Tests for ``BlochSignalSynthesisConsistencyLoss``.

Targets ``spectramr.models.losses.bloch_signal_synthesis_consistency_loss``
(idea 10 §10.2 / §10.6).

Plan acceptance criterion (§10.6):

- *"Loss zero for self-consistent triple; positive for an injected T2
  corruption"* — covered by the two property tests below.

Other contract tests:

- Construction: invalid weight / eps rejected
- Shape mismatches → ``ValueError``
- ``acquisition_params`` length must equal channel count
- Missing TR/TE/FA key in a params dict → ``ValueError``
- Gradient flows back to ``t1_ms``, ``t2_ms``, ``pd``
- Registered in the loss registry under the canonical name
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.losses.bloch_signal_synthesis_consistency_loss import (
    BlochSignalSynthesisConsistencyLoss,
    split_t1_t2_pd,
)


# ---------------------------------------------------------------------------
# Helper: synthesise a self-consistent contrast under the SPGR model.
# ---------------------------------------------------------------------------


def _synth_spgr(
    t1_ms: torch.Tensor,
    t2_ms: torch.Tensor,
    pd: torch.Tensor,
    tr_ms: float,
    te_ms: float,
    fa_deg: float,
) -> torch.Tensor:
    fa_rad = fa_deg * math.pi / 180.0
    sin_a = math.sin(fa_rad)
    cos_a = math.cos(fa_rad)
    e1 = torch.exp(-tr_ms / t1_ms)
    spgr = (sin_a * (1.0 - e1)) / (1.0 - cos_a * e1).clamp_min(1e-6)
    return pd * spgr * torch.exp(-te_ms / t2_ms)


# ---------------------------------------------------------------------------
# split_t1_t2_pd helper (keeps callers off the positional-arg foot-gun that
# caused the two mis-call bugs in the strategies)
# ---------------------------------------------------------------------------


def test_split_t1_t2_pd_resolves_by_alias() -> None:
    t1 = torch.ones(1, 1, 2, 2)
    t2 = torch.ones(1, 1, 2, 2) * 2
    pd = torch.ones(1, 1, 2, 2) * 3
    # Mixed-case + a decoy segmentation head that must be ignored.
    maps = {"T1": t1, "t2_ms": t2, "PD": pd, "segmentation": torch.zeros(1, 1, 2, 2)}
    out_t1, out_t2, out_pd = split_t1_t2_pd(maps)
    assert torch.equal(out_t1, t1)
    assert torch.equal(out_t2, t2)
    assert torch.equal(out_pd, pd)


def test_split_t1_t2_pd_raises_on_missing_map() -> None:
    with pytest.raises(KeyError, match="PD"):
        split_t1_t2_pd({"t1": torch.ones(1), "t2": torch.ones(1)})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_weight_rejected() -> None:
    """``weight < 0`` raises."""
    with pytest.raises(ValueError, match="weight"):
        BlochSignalSynthesisConsistencyLoss(weight=-0.1)


def test_non_positive_eps_rejected() -> None:
    """``eps ≤ 0`` raises."""
    with pytest.raises(ValueError, match="eps"):
        BlochSignalSynthesisConsistencyLoss(eps=0.0)


# ---------------------------------------------------------------------------
# Plan acceptance: zero loss for self-consistent triple
# ---------------------------------------------------------------------------


def test_zero_loss_for_self_consistent_triple() -> None:
    """L = 0 when observed contrasts equal the resynthesised SPGR signal."""
    H, W = 4, 4
    t1 = torch.full((1, 1, H, W), 800.0)
    t2 = torch.full((1, 1, H, W), 80.0)
    pd = torch.full((1, 1, H, W), 1.0)

    params = [
        {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0},
        {"TR_ms": 4000.0, "TE_ms": 80.0, "FA_deg": 90.0},
        {"TR_ms": 4000.0, "TE_ms": 12.0, "FA_deg": 90.0},
    ]
    observed = torch.cat(
        [
            _synth_spgr(t1, t2, pd, p["TR_ms"], p["TE_ms"], p["FA_deg"])
            for p in params
        ],
        dim=1,
    )

    loss = BlochSignalSynthesisConsistencyLoss(weight=1.0)
    val = loss(t1, t2, pd, observed, params).item()
    assert val < 1e-7


def test_loss_positive_under_t2_corruption() -> None:
    """Corrupting T2 raises the loss above zero."""
    H, W = 4, 4
    t1 = torch.full((1, 1, H, W), 800.0)
    t2 = torch.full((1, 1, H, W), 80.0)
    pd = torch.full((1, 1, H, W), 1.0)

    params = [
        {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0},
        {"TR_ms": 4000.0, "TE_ms": 80.0, "FA_deg": 90.0},
    ]
    observed = torch.cat(
        [
            _synth_spgr(t1, t2, pd, p["TR_ms"], p["TE_ms"], p["FA_deg"])
            for p in params
        ],
        dim=1,
    )
    # Corrupt T2 in the prediction.
    t2_bad = torch.full((1, 1, H, W), 200.0)

    loss = BlochSignalSynthesisConsistencyLoss(weight=1.0)
    val_clean = loss(t1, t2, pd, observed, params).item()
    val_corrupt = loss(t1, t2_bad, pd, observed, params).item()
    assert val_corrupt > val_clean
    assert val_corrupt > 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_observed_contrasts_3d_rejected() -> None:
    """``observed_contrasts`` must be 4-D."""
    t1 = torch.full((1, 1, 4, 4), 800.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    pd = torch.full((1, 1, 4, 4), 1.0)
    obs = torch.zeros(2, 4, 4)
    loss = BlochSignalSynthesisConsistencyLoss()
    with pytest.raises(ValueError, match="observed_contrasts"):
        loss(t1, t2, pd, obs, [{"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0}] * 2)


def test_acquisition_params_length_mismatch_rejected() -> None:
    """N_c channels must match params count."""
    t1 = torch.full((1, 1, 4, 4), 800.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    pd = torch.full((1, 1, 4, 4), 1.0)
    obs = torch.zeros(1, 3, 4, 4)
    loss = BlochSignalSynthesisConsistencyLoss()
    with pytest.raises(ValueError, match="acquisition_params"):
        loss(t1, t2, pd, obs, [{"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0}])


def test_missing_acquisition_key_rejected() -> None:
    """Missing TR/TE/FA in a params entry raises with the missing key name."""
    t1 = torch.full((1, 1, 4, 4), 800.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    pd = torch.full((1, 1, 4, 4), 1.0)
    obs = torch.zeros(1, 1, 4, 4)
    loss = BlochSignalSynthesisConsistencyLoss()
    with pytest.raises(ValueError, match="missing key"):
        loss(t1, t2, pd, obs, [{"TR_ms": 600.0, "TE_ms": 12.0}])


def test_t1_wrong_channel_count_rejected() -> None:
    """T1 must be ``[B, 1, H, W]``."""
    t1 = torch.full((1, 2, 4, 4), 800.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    pd = torch.full((1, 1, 4, 4), 1.0)
    obs = torch.zeros(1, 1, 4, 4)
    loss = BlochSignalSynthesisConsistencyLoss()
    with pytest.raises(ValueError, match="t1_ms"):
        loss(t1, t2, pd, obs, [{"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0}])


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_flows_to_all_three_maps() -> None:
    """Gradient reaches T1, T2, and PD."""
    H, W = 4, 4
    t1 = torch.full((1, 1, H, W), 800.0, requires_grad=True)
    t2 = torch.full((1, 1, H, W), 80.0, requires_grad=True)
    pd = torch.full((1, 1, H, W), 1.0, requires_grad=True)
    params = [{"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0}]
    obs = torch.zeros(1, 1, H, W)

    loss = BlochSignalSynthesisConsistencyLoss(weight=1.0)
    loss(t1, t2, pd, obs, params).backward()

    for name, t in (("t1", t1), ("t2", t2), ("pd", pd)):
        assert t.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(t.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``bloch_signal_synthesis_consistency`` resolvable in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "bloch_signal_synthesis_consistency" in list_available()
