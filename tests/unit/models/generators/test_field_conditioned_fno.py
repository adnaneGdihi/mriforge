"""Tests for FieldConditionedFNO (B-1.6/3.7 field-conditioned spectral operator)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.field_conditioned_fno import FieldConditionedFNO


def _net(enable: bool = True) -> FieldConditionedFNO:
    return FieldConditionedFNO(
        width=16, modes=8, n_layers=2, field_hidden=16, field_token_enable=enable
    )


def _perturb_field_mlp(m: FieldConditionedFNO) -> None:
    with torch.no_grad():
        for spec in m.spectral:
            spec.field_mlp[-1].weight.normal_(std=0.3)
            spec.field_mlp[-1].bias.normal_(std=0.3)


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        FieldConditionedFNO(in_channels=2)


def test_field_inert_at_identity_init() -> None:
    # The field-token MLP is zero-init -> per-mode gain = 1 -> field has no effect at
    # init (documents the contract the load-bearing test perturbs past).
    m = _net(enable=True).eval()
    x = torch.rand(1, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1]))
    b = m(x, field_strength=torch.tensor([7.0]))
    assert torch.allclose(a, b, atol=1e-6)


def test_spectral_multiplier_is_load_bearing_after_perturbation() -> None:
    # ANTI-FACADE (#16): once the field-token MLP moves off identity, the continuous
    # field changes the spectral response and hence the output.
    m = _net(enable=True)
    _perturb_field_mlp(m)
    x = torch.rand(2, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo, hi)


def test_ablation_is_field_agnostic() -> None:
    # field_token_enable=False bypasses the multiplier entirely, so the output is
    # invariant to the field even with a perturbed field-token MLP (one-knob control).
    m = _net(enable=False)
    _perturb_field_mlp(m)
    m.eval()
    x = torch.rand(1, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1]))
    b = m(x, field_strength=torch.tensor([7.0]))
    assert torch.allclose(a, b, atol=1e-6)


def test_low_and_high_corner_field_gains_are_independent() -> None:
    # Review #1/#2: the two H-corners (positive/negative freq) carry independent
    # spectral content, so they must get INDEPENDENT field gains (a single shared gain
    # mis-modulates half the spectrum). The MLP emits 2*modes1*modes2 -> (lo, hi).
    spec = _net().spectral[0]
    with torch.no_grad():
        spec.field_mlp[-1].weight.normal_(std=0.3)
        spec.field_mlp[-1].bias.normal_(std=0.3)
    tau = torch.log10(torch.tensor([[3.0]]))
    g = (1.0 + spec.field_mlp(tau)).view(1, 2, spec.modes1, spec.modes2)
    assert g.shape[1] == 2
    assert not torch.allclose(g[:, 0], g[:, 1])  # lo and hi gains differ


def test_ablation_knob_plumbs_through_factory() -> None:
    # Review #3: the one-knob ablation must reach the constructed model via the factory
    # signature filter (a silently-dropped/misspelled knob would break the ablation).
    import warnings

    from mriforge.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():  # ModelFactory is the canonical (deprecated) build path
        warnings.simplefilter("ignore", DeprecationWarning)
        from mriforge.models.factories.model_factory import ModelFactory

        f = ModelFactory()
    on = f.create_generator(model_type="field_conditioned_fno", in_channels=1,
                            out_channels=1, field_token_enable=True, modes=8, width=16)
    off = f.create_generator(model_type="field_conditioned_fno", in_channels=1,
                             out_channels=1, field_token_enable=False, modes=8, width=16)
    assert on.spectral[0].field_token_enable is True
    assert off.spectral[0].field_token_enable is False


def test_registered() -> None:
    from mriforge.models.registry import MODEL_REGISTRY

    assert "field_conditioned_fno" in MODEL_REGISTRY
