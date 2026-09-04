"""Unit tests for the physics-derived prompt embedding (PR-7).

This is the framework's differentiator versus published foundation-recon models:
the conditioning prompt is a deterministic function of physical acquisition
parameters (TR, TE, flip angle, field strength) rather than empirically learned
prompt tokens.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.physics_prompt import PhysicsPromptEmbedding


@pytest.fixture
def params():
    # [TR, TE, flip_angle_deg, field_strength_T]
    return torch.tensor(
        [
            [2000.0, 80.0, 90.0, 3.0],
            [500.0, 10.0, 15.0, 0.3],
        ]
    )


class TestPhysicsPromptShape:
    def test_output_shape(self, params):
        emb = PhysicsPromptEmbedding(in_features=4, embed_dim=32)
        out = emb(params)
        assert out.shape == (2, 32)

    def test_default_in_features_is_four(self, params):
        emb = PhysicsPromptEmbedding(embed_dim=16)
        assert emb.in_features == 4
        assert emb(params).shape == (2, 16)

    def test_wrong_feature_count_raises(self):
        emb = PhysicsPromptEmbedding(in_features=4, embed_dim=8)
        with pytest.raises(ValueError):
            emb(torch.randn(2, 3))


class TestPhysicsPromptDeterminism:
    def test_deterministic(self, params):
        emb = PhysicsPromptEmbedding(in_features=4, embed_dim=16).eval()
        a = emb(params)
        b = emb(params)
        torch.testing.assert_close(a, b)


class TestPhysicsPromptDifferentiable:
    def test_gradient_flows(self, params):
        emb = PhysicsPromptEmbedding(in_features=4, embed_dim=16)
        x = params.clone().requires_grad_(True)
        out = emb(x).sum()
        out.backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in emb.parameters())


class TestPhysicsPromptDistinctness:
    def test_distinct_params_distinct_embeddings(self, params):
        emb = PhysicsPromptEmbedding(in_features=4, embed_dim=16).eval()
        out = emb(params)
        # The two rows have very different physics -> embeddings must differ.
        assert not torch.allclose(out[0], out[1])
