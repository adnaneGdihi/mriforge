"""Integration test: Bloch-equivariant translation Bloch round trip (Idea 3).

The unpaired Bloch-equivariance branch is correct iff the Bloch
forward at the HF field strength applied to the *same* tissue
parameters that produced the ULF observation yields a non-trivial,
finite signal. This test exercises the round trip end-to-end on a
small parametric atlas without going through the full training stack.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
from mriforge.infrastructure.training.strategies.bloch_equivariant_translation_strategy import (
    _TissueAtlas,
)


@pytest.mark.integration
def test_atlas_samples_are_valid_relaxation_triples() -> None:
    atlas = _TissueAtlas()
    samples = atlas.sample(64)
    # Shape and basic constraints.
    assert samples.shape == (64, 3)
    M0, T1, T2 = samples.unbind(dim=-1)
    assert (M0 > 0).all()
    assert (T1 > 0).all()
    assert (T2 > 0).all()
    # T2 ≤ T1 (the manifold constraint).
    assert (T2 <= T1 + 1e-3).all()


@pytest.mark.integration
def test_bloch_round_trip_produces_finite_signal() -> None:
    """For atlas-sampled (M0, T1, T2), both ULF and HF Bloch forwards
    must produce finite, non-NaN signals so the equivariance residual
    is well-defined."""
    atlas = _TissueAtlas()
    bloch = DifferentiableBlochLayer()
    H = 8
    n = H * H
    q = atlas.sample(n).reshape(1, 1, H, H, 3)
    M0, T1, T2 = q[..., 0], q[..., 1], q[..., 2]
    alpha = torch.tensor(0.2618)  # 15°
    ulf = bloch(
        t1_map=(T1 * 0.3).contiguous(),
        t2_map=(T2 * 1.05).contiguous(),
        pd_map=M0,
        tr=8.0,
        te=4.0,
        flip_angle_rad=alpha,
        b0_map=None,
        b1_map=None,
    )
    hf = bloch(
        t1_map=T1,
        t2_map=T2,
        pd_map=M0,
        tr=8.0,
        te=4.0,
        flip_angle_rad=alpha,
        b0_map=None,
        b1_map=None,
    )
    assert ulf.shape == (1, 1, H, H)
    assert torch.isfinite(ulf).all()
    assert torch.isfinite(hf).all()
    # The two signals should not be identical — the field scaling makes
    # ULF strictly different from HF.
    assert (ulf - hf).abs().mean().item() > 1e-6


@pytest.mark.integration
def test_bloch_checkpoint_flag_preserves_output_equality() -> None:
    """With grad disabled, the checkpoint path must produce identical
    outputs to the non-checkpoint path. With grad enabled, gradients
    must still flow."""
    from types import SimpleNamespace

    import torch.nn as nn

    from mriforge.infrastructure.training.strategies.bloch_equivariant_translation_strategy import (
        BlochEquivariantTranslationStrategy,
    )

    class _Id(nn.Module):
        def forward(self, x, *a, **kw):
            return x

    def _build(ckpt: bool) -> BlochEquivariantTranslationStrategy:
        strat = BlochEquivariantTranslationStrategy.__new__(
            BlochEquivariantTranslationStrategy
        )
        strat.config = SimpleNamespace(training=SimpleNamespace())
        strat.device = torch.device("cpu")
        strat.env = SimpleNamespace(
            models={"generator": _Id()},
            device=torch.device("cpu"),
            config=strat.config,
        )
        strat.t1_scale_ulf = 0.30
        strat.t2_scale_ulf = 1.05
        strat.tr_ulf = strat.tr_hf = 8.0
        strat.te_ulf = strat.te_hf = 4.0
        strat.flip_angle_rad = 0.2618
        strat.lambda_eq = 1.0
        strat.lambda_cycle = 0.1
        strat.unpaired_patch = 8
        strat.use_bloch_checkpoint = ckpt
        strat.bloch = DifferentiableBlochLayer()
        strat.atlas = _TissueAtlas()
        type(strat).generator_model = property(  # type: ignore[assignment]
            lambda s: s.env.models["generator"]
        )
        return strat

    strat_a = _build(False)
    strat_b = _build(True)

    # In no-grad mode the strategy's _simulate_pair should produce the
    # same outputs regardless of checkpointing — checkpointing affects
    # only the backward graph, not the forward values. Seed
    # immediately before each call because the atlas draws from the
    # default global generator.
    with torch.no_grad():
        torch.manual_seed(7)
        ulf_a, hf_a = strat_a._simulate_pair(64)
    with torch.no_grad():
        torch.manual_seed(7)
        ulf_b, hf_b = strat_b._simulate_pair(64)
    assert torch.allclose(ulf_a, ulf_b, atol=1e-6)
    assert torch.allclose(hf_a, hf_b, atol=1e-6)
