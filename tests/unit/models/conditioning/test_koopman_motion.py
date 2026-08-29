"""Tests for the Koopman temporal motion encoder (2026-05-26).

Encodes a per-line pose sequence θ(τ) into a latent advanced by a *linear*
Koopman operator — natural for periodic/respiratory motion.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.conditioning import ConditioningContext  # noqa: E402
from mriforge.models.conditioning.encoders import get_conditioner  # noqa: E402
from mriforge.models.conditioning.koopman_motion import (  # noqa: E402
    KoopmanMotionEncoder,
)


def test_forward_pools_pose_to_embedding() -> None:
    enc = KoopmanMotionEncoder(embed_dim=16)
    out = enc(ConditioningContext(motion_pose=torch.randn(2, 3, 20)))
    assert out.shape == (2, 16)


def test_koopman_advance_is_linear() -> None:
    enc = KoopmanMotionEncoder(embed_dim=8)
    z = torch.randn(2, 8)
    assert torch.allclose(enc.advance(2.0 * z), 2.0 * enc.advance(z), atol=1e-5)


def test_rollout_shape() -> None:
    enc = KoopmanMotionEncoder(embed_dim=8)
    assert enc.rollout(torch.randn(2, 8), steps=5).shape == (2, 5, 8)


def test_registered_under_motion_koopman() -> None:
    assert isinstance(get_conditioner("motion_koopman", embed_dim=8), KoopmanMotionEncoder)


def test_requires_motion_pose() -> None:
    with pytest.raises(ValueError, match="motion_pose"):
        KoopmanMotionEncoder(embed_dim=8)(ConditioningContext())
