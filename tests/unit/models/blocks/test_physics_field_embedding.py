"""Tests for PhysicsFieldEmbedding (A-5.2 physics-grounded field representation)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.relaxation_priors import TissueClass, bottomley_t1
from spectramr.models.blocks.physics_field_embedding import PhysicsFieldEmbedding


def test_embed_dim_and_shape() -> None:
    emb = PhysicsFieldEmbedding()
    out = emb(torch.tensor([0.1, 1.5, 7.0]))
    assert out.shape == (3, PhysicsFieldEmbedding.embed_dim)


def test_t1_dispersion_matches_bottomley() -> None:
    # The embedding's T1 features are the genuine Bottomley T1(B0) dispersion (not raw b0 / a
    # black-box) — feature[1] == T1_wm(b0)/1500 for each field.
    emb = PhysicsFieldEmbedding(enable_physics=True)
    for b in (0.1, 1.5, 7.0):
        e = emb(torch.tensor([b]))[0]
        assert abs(float(e[1]) * 1500.0 - bottomley_t1(b, TissueClass.WHITE_MATTER)) < 1.0


def test_dispersion_is_monotone_and_generalises() -> None:
    # T1 rises monotonically with field -> the embedding interpolates sensibly to an UNSEEN field
    # (0.55T between 0.3T and 1.5T), the property that lets the prior generalise across fields.
    emb = PhysicsFieldEmbedding(enable_physics=True)
    t03 = float(emb(torch.tensor([0.3]))[0, 1])
    t055 = float(emb(torch.tensor([0.55]))[0, 1])
    t15 = float(emb(torch.tensor([1.5]))[0, 1])
    assert t03 < t055 < t15


def test_ablation_zeroes_dispersion_keeps_field() -> None:
    on = PhysicsFieldEmbedding(enable_physics=True)(torch.tensor([1.5]))[0]
    off = PhysicsFieldEmbedding(enable_physics=False)(torch.tensor([1.5]))[0]
    assert float(off[1:].abs().sum()) == 0.0  # dispersion features zeroed
    assert abs(float(off[0]) - float(on[0])) < 1e-6  # log-field kept (param-matched ablation)


def test_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="enable_physics"):
        PhysicsFieldEmbedding(enable_physics="yes")  # type: ignore[arg-type]


def test_rejects_nonpositive_field() -> None:
    # b0 <= 0 is unphysical -> raise (not silently clamp, pitfall #9; matches scalar bottomley_t1).
    with pytest.raises(ValueError, match="field_strength must be > 0"):
        PhysicsFieldEmbedding()(torch.tensor([1.5, 0.0]))
