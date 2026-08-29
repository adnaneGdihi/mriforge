r"""Unit tests for the Gromov-Wasserstein cross-field alignment loss (GW-CFA).

Targets ``mriforge.models.losses.gw_cross_field_loss``.

GW-CFA aligns ultra-low-field and high-field representations with *no*
correspondence, exploiting the physical premise that the *relative* geometric
structure of anatomy (intra-domain distances) is approximately field-invariant
even when absolute intensities are not. The loss is the entropic Gromov-
Wasserstein value between the two domains' intra-domain distance matrices, so
minimising it aligns relational structure.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _emb(n: int, d: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def _random_orthogonal(d: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d))
    return q


def test_gw_value_lower_for_isometric_copy_than_random() -> None:
    from mriforge.models.losses.gw_cross_field_loss import (
        GromovWassersteinCrossFieldLoss,
    )

    src = _emb(8, 4, 0)
    q = _random_orthogonal(4, 7)
    perm = torch.randperm(8)
    tgt_iso = (src @ q)[perm]  # relational structure preserved (isometry + relabel)
    tgt_rand = _emb(8, 4, 99)

    loss = GromovWassersteinCrossFieldLoss(reg=0.02)
    v_iso = loss(prediction=src, target=tgt_iso)
    v_rand = loss(prediction=src, target=tgt_rand)
    assert float(v_iso) < float(v_rand), "relationally-aligned domains cost less"


def test_gw_value_is_differentiable() -> None:
    from mriforge.models.losses.gw_cross_field_loss import (
        GromovWassersteinCrossFieldLoss,
    )

    src = _emb(6, 4, 1).requires_grad_(True)
    tgt = _emb(6, 4, 2)
    v = GromovWassersteinCrossFieldLoss(reg=0.05)(prediction=src, target=tgt)
    v.backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()
    assert v.ndim == 0


def test_recovers_correspondence_on_relabeled_copy() -> None:
    from mriforge.models.losses.gw_cross_field_loss import (
        GromovWassersteinCrossFieldLoss,
    )

    # A 1-D ramp with all-distinct pairwise gaps gives a fully asymmetric intra-
    # distance structure (no isometric symmetry), so the GW optimum is the unique
    # relabeling permutation. (Symmetric structure would only resolve up to an
    # isometry -- the proposal's stated indeterminacy, fixed by anchors.)
    pos = torch.tensor([0.0, 1.0, 3.0, 7.0, 15.0]).reshape(5, 1)
    perm = torch.tensor([2, 0, 3, 1, 4])
    tgt = pos[perm]  # pure relabeling (identity isometry)

    loss = GromovWassersteinCrossFieldLoss(reg=0.01, gw_iter=20)
    coupling = loss.gw_coupling(pos, tgt)
    matched = coupling.argmax(dim=1)  # src i -> best tgt j
    inv = torch.argsort(perm)  # correct: src i matches tgt position inv[i]
    purity = (matched == inv).float().mean().item()
    assert purity >= 0.8, f"GW correspondence purity too low: {purity:.2f}"


def test_loss_is_registered() -> None:
    from mriforge.models.losses import create_loss
    from mriforge.models.losses.gw_cross_field_loss import (
        GromovWassersteinCrossFieldLoss,
    )

    assert isinstance(create_loss("gw_cross_field"), GromovWassersteinCrossFieldLoss)
