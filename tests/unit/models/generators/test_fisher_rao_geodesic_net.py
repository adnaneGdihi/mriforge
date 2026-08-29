"""Tests for FisherRaoGeodesicNet (B-3.4 Fisher-Rao geodesic cross-field translator)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.fisher_rao_geodesic_net import FisherRaoGeodesicNet


def _net(**kw) -> FisherRaoGeodesicNet:
    return FisherRaoGeodesicNet(width=24, n_blocks=2, **kw)


def _f(b_s, b_t, n=2):
    return torch.full((n,), float(b_s)), torch.full((n,), float(b_t))


def test_forward_shape_and_near_identity_init() -> None:
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    bs, bt = _f(0.1, 7.0)
    out = m(x, field_strength=bs, field_strength_target=bt).detach()
    assert out.shape == (2, 1, 16, 16)
    assert float((out - x).norm() / x.norm()) < 0.1  # zero-init head -> identity


def test_fisher_geometry_output_bounded_by_construction() -> None:
    # The geodesic output stays in [0,1] WITHOUT a clamp, even with a large angular velocity
    # (the distinctive Fisher-Rao norm-preserving property).
    torch.manual_seed(0)
    m = _net(use_fisher_rao_geometry=True).eval()
    with torch.no_grad():
        m.head.weight.normal_(0.0, 5.0)
        m.head.bias.normal_(0.0, 5.0)
    x = torch.rand(2, 1, 16, 16)
    bs, bt = _f(0.1, 7.0)
    out = m(x, field_strength=bs, field_strength_target=bt).detach()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_field_is_load_bearing_after_perturbation() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():  # escape the zero-init head + activate the field FiLM
        m.head.weight.normal_(0.0, 0.5)
        m.head.bias.normal_(0.0, 0.5)
        m.field_mlp[-1].weight.normal_(0.0, 0.5)
        m.field_mlp[-1].bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(o1.detach(), o2.detach())


def test_euclidean_ablation_builds_and_runs() -> None:
    bs, bt = _f(0.1, 7.0, n=1)
    out = _net(use_fisher_rao_geometry=False)(
        torch.rand(1, 1, 16, 16), field_strength=bs, field_strength_target=bt
    )
    assert out.shape == (1, 1, 16, 16)
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0


def test_both_fields_required_for_probe() -> None:
    import inspect

    params = inspect.signature(FisherRaoGeodesicNet.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty
    assert params["field_strength_target"].default is inspect.Parameter.empty


def test_rejects_complex() -> None:
    bs, bt = _f(0.1, 7.0, n=1)
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=bs, field_strength_target=bt)


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        FisherRaoGeodesicNet(in_channels=2)


def test_registered() -> None:
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "fisher_rao_geodesic_net" in MODEL_REGISTRY


def test_probe_skip_includes_dead_loss() -> None:
    """The near-identity init drives the configured placeholder l1 to ~0 on the
    probe phantom; the real loss is computed by the strategy, so the model must
    opt the dead_loss gate out (alongside identity_collapse) to avoid a Tier-2
    false positive on an arm that trains on the cluster."""
    assert FisherRaoGeodesicNet.synthetic_forward_probe_skip >= {
        "identity_collapse",
        "dead_loss",
    }


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_field_mlp_input() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    assert m.field_mlp[0].in_features == 2 + 3  # (b_s, b_t) + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    with pytest.raises(ValueError, match="contrast_id"):
        m(
            torch.rand(1, 1, 16, 16),
            field_strength=torch.tensor([0.1]),
            field_strength_target=torch.tensor([7.0]),
        )


def test_contrast_changes_output_after_perturbation() -> None:
    torch.manual_seed(0)
    m = _net(use_contrast_conditioning=True, num_contrasts=3).eval()
    with torch.no_grad():
        m.field_mlp[-1].weight.normal_(0.0, 0.5)
        m.field_mlp[-1].bias.normal_(0.0, 0.5)
        m.head.weight.normal_(0.0, 0.1)  # head zero-init pins delta=0; move it so FiLM matters
    x = torch.rand(2, 1, 16, 16)
    bs = torch.tensor([0.1, 0.1])
    bt = torch.tensor([7.0, 7.0])
    a = m(x, field_strength=bs, field_strength_target=bt, contrast_id=torch.tensor([0, 0]))
    b = m(x, field_strength=bs, field_strength_target=bt, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a.detach(), b.detach())
