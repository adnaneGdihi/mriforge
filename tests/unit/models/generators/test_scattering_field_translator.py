"""Tests for ScatteringFieldTranslator (B-1.3 ->7T host backbone)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.scattering_field_translator import ScatteringFieldTranslator


def _net(**kw) -> ScatteringFieldTranslator:
    return ScatteringFieldTranslator(width=24, n_blocks=2, **kw)


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 3.0]))
    assert out.shape == (2, 1, 16, 16)


def test_field_is_load_bearing_after_perturbation() -> None:
    # FiLM is zero-init (identity) at start; after perturbing the FiLM MLPs the source field
    # changes the output (the conditioning is wired, not inert).
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        for film in m.films:
            film.mlp[-1].weight.normal_(0.0, 0.5)
            film.mlp[-1].bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]))
    assert not torch.allclose(o1.detach(), o2.detach())


def test_field_batch_guard() -> None:
    m = _net()
    with pytest.raises(ValueError, match="field_strength batch"):
        m(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([1.5]))


def test_field_required_for_probe() -> None:
    import inspect

    params = inspect.signature(ScatteringFieldTranslator.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty


def test_rejects_complex_and_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.5]))
    with pytest.raises(ValueError, match="magnitude-only"):
        ScatteringFieldTranslator(in_channels=2)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "scattering_field_translator" in MODEL_REGISTRY


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_film_sequence_dim() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    assert m.films[0].sequence_dim == 1 + 3  # log-field + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    with pytest.raises(ValueError, match="contrast_id"):
        m(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 3.0]))


def test_contrast_changes_output_after_perturbation() -> None:
    torch.manual_seed(0)
    m = _net(use_contrast_conditioning=True, num_contrasts=3).eval()
    with torch.no_grad():
        for film in m.films:
            film.mlp[-1].weight.normal_(0.0, 0.5)
            film.mlp[-1].bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    fs = torch.tensor([3.0, 3.0])
    a = m(x, field_strength=fs, contrast_id=torch.tensor([0, 0]))
    b = m(x, field_strength=fs, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a.detach(), b.detach())
