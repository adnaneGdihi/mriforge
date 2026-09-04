"""Correctness tests for the Phase-3 Tier-2 invariant probes.

These assert each model satisfies its *defining mathematical invariant* on a
synthetic input (Glow round-trip, C_n rotation-invariant density, structural
divergence-freeness, vMF sphere constraint, VQ index range, heat-equation
high-frequency dissipation), and that the probe correctly *rejects* a model
that violates its invariant. This is the audit-time correctness gate added
with ``validation/phase3_probes.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.validation.phase3_probes import (  # noqa: E402
    has_invariant_probe,
    run_invariant_probe,
)


def _build(model_type):
    """Build a small instance of each probed model + a matching input."""
    if model_type == "glow":
        from spectramr.models.generative.glow import Glow

        return Glow(num_scales=2, num_steps=2, hidden_channels=16), torch.randn(2, 1, 16, 16)
    if model_type == "equivariant_flow":
        from spectramr.models.generative.equivariant_flow import EquivariantFlow

        return EquivariantFlow(), torch.randn(2, 1, 16, 16)
    if model_type == "divergence_free_flow":
        from spectramr.models.generative.divergence_free_flow import DivergenceFreeFlow

        return DivergenceFreeFlow(), torch.randn(2, 2, 16, 16)  # 2-component velocity state
    if model_type == "hyperspherical_vae":
        from spectramr.models.vae.hyperspherical_vae import HypersphericalVAE

        return HypersphericalVAE(latent_dim=8), torch.randn(2, 1, 32, 32)
    if model_type == "hierarchical_vq_vae":
        from spectramr.models.vq.hierarchical_vq_vae import HierarchicalVQVAE

        return (
            HierarchicalVQVAE(codebook_size_top=64, codebook_size_bot=64),
            torch.randn(2, 1, 32, 32),
        )
    if model_type == "blurring_diffusion":
        from spectramr.models.diffusion.blurring_diffusion import BlurringDiffusion

        return BlurringDiffusion(image_size=16, num_timesteps=100), torch.randn(2, 1, 16, 16)
    raise AssertionError(model_type)


_PROBED = [
    "glow",
    "equivariant_flow",
    "divergence_free_flow",
    "hyperspherical_vae",
    "hierarchical_vq_vae",
    "blurring_diffusion",
]


@pytest.mark.registry_contract
@pytest.mark.parametrize("model_type", _PROBED)
def test_invariant_holds(model_type: str) -> None:
    torch.manual_seed(0)
    model, x = _build(model_type)
    model.eval()
    result = run_invariant_probe(model_type, model, x, "cpu")
    assert result is not None, f"no probe registered for {model_type}"
    ok, msg = result
    assert ok, f"{model_type} invariant FAILED: {msg}"


@pytest.mark.registry_contract
def test_no_probe_for_unrelated_model() -> None:
    assert run_invariant_probe("unet", object(), torch.zeros(1), "cpu") is None
    assert not has_invariant_probe("unet")
    assert has_invariant_probe("glow")


@pytest.mark.registry_contract
def test_probe_rejects_broken_invariant() -> None:
    """A Glow whose inverse is corrupted must FAIL the round-trip probe —
    proving the probe detects violations, not just rubber-stamps."""
    from spectramr.models.generative.glow import Glow

    torch.manual_seed(0)
    model = Glow(num_scales=2, num_steps=2, hidden_channels=16).eval()
    # Corrupt the inverse so the round-trip no longer reconstructs the input.
    model.inverse = lambda z, shape: torch.zeros(shape)  # type: ignore[assignment]
    ok, msg = run_invariant_probe("glow", model, torch.randn(2, 1, 16, 16), "cpu")
    assert ok is False
    assert "round-trip" in msg


@pytest.mark.registry_contract
def test_probe_exception_becomes_clean_failure() -> None:
    """A probe that raises internally returns (False, msg), never propagates."""

    class _Boom(torch.nn.Module):
        def forward(self, x):  # pragma: no cover - exercised via probe
            raise RuntimeError("boom")

        def log_prob(self, x):
            raise RuntimeError("boom")

        def inverse(self, z, shape):
            raise RuntimeError("boom")

    ok, msg = run_invariant_probe("glow", _Boom(), torch.randn(1, 1, 16, 16), "cpu")
    assert ok is False
    assert "boom" in msg or "raised" in msg
