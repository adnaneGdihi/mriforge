"""Regression test for F11 (smoke_audit_20260521 round 6) —
``LatentDiffusionHierarchical`` must expose identity
``encode_to_latent`` / ``decode_from_latent`` so the DiffusionTrainingStrategy's
latent-diffusion path can route through it as a no-op.

Root cause: ``baseline_latent_diffusion_3d`` uses
``model_type='latent_diffusion_hierarchical'`` which matches the
``"latent" in mt and "diffusion" in mt`` substring heuristic in
``diffusion.py::_is_latent_diffusion``. The strategy then expects an
``encode_to_latent`` method on the generator, but the model didn't
implement one (despite operating in pixel space — the final
``F.interpolate`` restores input spatial size). Crashed with
``[Latent Diffusion] Generator LatentDiffusionHierarchical does not
implement encode_to_latent()``.

Fix: identity encode/decode pair on the model, same pattern as
``x_diffusion_latent.py``.

This test pins:
- The model has both `encode_to_latent` and `decode_from_latent`.
- Both are identity on a sample tensor.
- The model is callable as a no-op encode → forward → decode chain.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def model():
    from mriforge.models.generators.generative_variants import (
        LatentDiffusionHierarchical,
    )
    # Use in_channels=2 to match the smoke-test config (complex
    # real-stacked single-coil).
    return LatentDiffusionHierarchical(in_channels=2, out_channels=2, base_features=16)


def test_encode_to_latent_is_identity(model) -> None:
    """The pixel-space model must implement ``encode_to_latent`` as
    identity so the latent-diffusion wrapper is a no-op."""
    assert hasattr(model, "encode_to_latent"), (
        "LatentDiffusionHierarchical must expose encode_to_latent — "
        "the latent-diffusion strategy calls it unconditionally."
    )
    x = torch.randn(2, 2, 64, 64)
    z = model.encode_to_latent(x)
    assert torch.equal(z, x), "encode_to_latent must be identity for this pixel-space model."


def test_decode_from_latent_is_identity(model) -> None:
    """Pair of encode_to_latent — identity on the latent tensor."""
    assert hasattr(model, "decode_from_latent"), (
        "LatentDiffusionHierarchical must expose decode_from_latent."
    )
    z = torch.randn(2, 2, 64, 64)
    out = model.decode_from_latent(z)
    assert torch.equal(out, z), "decode_from_latent must be identity."


def test_forward_still_works(model) -> None:
    """The forward pass must still produce the right output shape —
    the F.interpolate at the end restores spatial size. Pinned so
    the F11 fix doesn't change forward semantics."""
    x = torch.randn(2, 2, 64, 64)
    out = model(x)
    assert out.shape == x.shape, (
        f"Forward must preserve spatial size; got {out.shape}, expected {x.shape}"
    )
