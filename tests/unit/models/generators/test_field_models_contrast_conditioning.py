"""The three field-conditioned models declare and exercise contrast conditioning.

``check_multi_contrast_model_support`` trusts the ``supports_contrast_conditioning``
flag on ``@register_model``; ten ``ulf_paired_restoration`` arms failed it because
the flag was absent while the models read ``contrast_id`` (2026-09-02 review). A
declaration is only as good as the mechanism behind it: two contrast ids must
produce two outputs when conditioning is on.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.field_conditioned_inr import FieldConditionedINR
from spectramr.models.generators.field_guided_score_unet import FieldGuidedScoreUNet
from spectramr.models.generators.field_velocity_unet import FieldVelocityUNet
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import model_supports

MODELS = ("field_conditioned_inr", "field_velocity_unet", "field_guided_score_unet")


@pytest.mark.parametrize("name", MODELS)
def test_the_registration_declares_contrast_conditioning(name: str) -> None:
    populate_model_registry()
    assert model_supports(name, "supports_contrast_conditioning")


def _run(model: torch.nn.Module, name: str, contrast_id: torch.Tensor | None) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 16)
    field = torch.tensor([0.064, 0.064])
    if name == "field_guided_score_unet":
        return model(
            x,
            timesteps=torch.tensor([3, 3]),
            field_strength=field,
            cond_image=torch.rand(2, 1, 16, 16),
            contrast_id=contrast_id,
        )
    return model(x, field_strength=field, contrast_id=contrast_id)


def _build(name: str, conditioning: bool) -> torch.nn.Module:
    torch.manual_seed(0)
    if name == "field_conditioned_inr":
        return FieldConditionedINR(
            feat_dim=4,
            hidden_features=16,
            hidden_layers=1,
            style_dim=8,
            use_contrast_conditioning=conditioning,
            num_contrasts=3,
        )
    if name == "field_velocity_unet":
        return FieldVelocityUNet(width=8, use_contrast_conditioning=conditioning, num_contrasts=3)
    return FieldGuidedScoreUNet(
        width=8, time_dim=8, use_contrast_conditioning=conditioning, num_contrasts=3
    )


def _leave_identity_init(model: torch.nn.Module) -> None:
    """The FiLM heads are zero-initialised so modulation starts as the identity
    (``FieldFiLMBlock``); at init no conditioning input can change the output.
    Perturbing every parameter asks whether the pathway is wired, not trained."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))


@pytest.mark.parametrize("name", MODELS)
def test_two_contrast_ids_give_two_outputs(name: str) -> None:
    model = _build(name, conditioning=True).eval()
    _leave_identity_init(model)
    with torch.no_grad():
        a = _run(model, name, torch.tensor([0, 0]))
        b = _run(model, name, torch.tensor([1, 1]))
    assert a.shape == b.shape
    assert not torch.allclose(a, b), f"{name}: contrast_id has no effect on the output"


@pytest.mark.parametrize("name", MODELS)
def test_conditioning_off_ignores_the_id(name: str) -> None:
    """Planted control: with conditioning off the id must not change the output."""
    model = _build(name, conditioning=False).eval()
    _leave_identity_init(model)
    with torch.no_grad():
        a = _run(model, name, None)
        b = _run(model, name, torch.tensor([1, 1]))
    assert torch.allclose(a, b)
