
import torch
import torch.nn as nn

from spectramr.models.interfaces import IDiscriminator, IGenerator
from spectramr.models.registry import MODEL_REGISTRY, get_model_class, register_model


# Create dummy models for testing
class DummyGenerator(nn.Module, IGenerator):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class DummyDiscriminator(nn.Module, IDiscriminator):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x, target=None):
        return self.conv(x).mean()

# Test ModelRegistry registration
def test_model_registry_registration():
    """Test that @register_model correctly adds models to the registry."""
    # Temporarily clean up registry if 'test_gen' exists
    if 'test_gen' in MODEL_REGISTRY:
        del MODEL_REGISTRY['test_gen']

    @register_model("test_gen", "test_mode")
    class TestGen(DummyGenerator):
        pass

    assert 'test_gen' in MODEL_REGISTRY
    assert MODEL_REGISTRY['test_gen']['mode'] == 'test_mode'
    assert get_model_class("test_gen") is TestGen

from spectramr.models.losses.registry import create_loss, register_loss


def test_loss_registry_and_creation():
    """Test loss registration and creation via registry."""
    @register_loss("dummy_loss")
    class DummyLoss(nn.Module):
        def __init__(self, weight=1.0):
            super().__init__()
            self.weight = weight

        def forward(self, pred, target):
            return torch.nn.functional.mse_loss(pred, target) * self.weight

    loss_fn = create_loss("dummy_loss", weight=2.5)
    assert isinstance(loss_fn, nn.Module)
    assert loss_fn.weight == 2.5


def test_model_creator_refuses_list_to_int_silent_coercion() -> None:
    """F-COERCE-LOUD / 2026-05-20 — ModelCreator must NOT silently
    narrow a multi-element list to its first element when the
    constructor declares the param as ``int``.

    Smoke 20260519 logged "Coerced list param 'features' [32, 64, 128,
    256, 512] → 32 for NeuralODEGenerator (constructor expects int)" —
    a single warning followed by an *architecturally* very different
    model. The user's intent of a 5-level UNet became a 32-channel
    residual block; the bad scale propagated to training loss /
    gradient magnitude / mosaic outputs.

    The fixed factory raises TypeError so the YAML bug is impossible
    to ignore.
    """
    import pytest
    from spectramr.models.factories.model_factory import (
        ActivationManager,
        ModelCreator,
        ModelRegistry,
        ModelValidator,
        ParameterMapper,
    )

    @register_model("loud_coerce_target", "test_mode")
    class LoudCoerceTarget(DummyGenerator):
        def __init__(self, in_channels: int = 1, features: int = 64):
            super().__init__(in_channels=in_channels)
            self.features = features

    registry = ModelRegistry()
    creator = ModelCreator(
        registry=registry,
        parameter_mapper=ParameterMapper(),
        validator=ModelValidator(registry),
        activation_manager=ActivationManager(),
    )
    with pytest.raises(TypeError, match=r"discard|silent fallback|CLAUDE.md"):
        creator.create_generator(
            "loud_coerce_target",
            in_channels=1,
            features=[32, 64, 128, 256, 512],
        )


def test_model_creator_accepts_single_element_list_narrowing() -> None:
    """Single-element ``[X]`` → ``X`` remains allowed (no architecture
    loss); F-COERCE-LOUD only fires when ≥2 elements would be discarded.
    """
    from spectramr.models.factories.model_factory import (
        ActivationManager,
        ModelCreator,
        ModelRegistry,
        ModelValidator,
        ParameterMapper,
    )

    @register_model("single_element_coerce_target", "test_mode")
    class SingleElementCoerceTarget(DummyGenerator):
        def __init__(self, in_channels: int = 1, features: int = 64):
            super().__init__(in_channels=in_channels)
            self.features = features

    registry = ModelRegistry()
    creator = ModelCreator(
        registry=registry,
        parameter_mapper=ParameterMapper(),
        validator=ModelValidator(registry),
        activation_manager=ActivationManager(),
    )
    out = creator.create_generator(
        "single_element_coerce_target",
        in_channels=1,
        features=[128],
    )
    inner = out.module if hasattr(out, "module") else out
    assert getattr(inner, "features", 128) == 128
