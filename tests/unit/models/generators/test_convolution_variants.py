"""Registry identity for the thin convolution variants (#977).

These classes are small, and smallness is why the collision went unnoticed:
``EnhancedUNet`` is 26 lines, so nobody reading a registry dump asked whether
the name it claimed was already taken by a 657-line class elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.factories.model_factory import ModelFactory
from spectramr.models.generators.convolution_variants import EnhancedUNet
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY


@pytest.fixture(scope="module", autouse=True)
def _registry():
    populate_model_registry()


def _factory_class(name: str):
    return ModelFactory()._registry._generators.get(name)


class TestSEResidualUNetName:
    """``EnhancedUNet`` is registered as ``se_residual_unet``, not ``enhanced_unet``."""

    def test_both_registries_bind_se_residual_unet_to_this_class(self):
        assert MODEL_REGISTRY["se_residual_unet"]["class"] is EnhancedUNet
        assert _factory_class("se_residual_unet") is EnhancedUNet

    def test_it_is_buildable_at_all(self):
        """The regression that motivated the rename.

        While this class claimed ``enhanced_unet`` in ``MODEL_REGISTRY``, the
        factory bound that name to the configurable ``UNet`` -- and the factory
        is the only build path (``create_generator`` raises for a name it does
        not hold rather than consulting ``MODEL_REGISTRY``). So the class was
        registered and reachable from nothing.
        """
        model = EnhancedUNet(in_channels=2, out_channels=2, base_features=8)
        out = model(torch.zeros(1, 2, 32, 32))
        assert out.shape == (1, 2, 32, 32)

    def test_it_no_longer_claims_enhanced_unet(self):
        assert MODEL_REGISTRY["enhanced_unet"]["class"] is not EnhancedUNet


class TestEnhancedUNetNameIsUndivided:
    """The 11 arms declaring ``enhanced_unet`` must keep building what they built."""

    def test_the_two_registries_agree(self):
        assert MODEL_REGISTRY["enhanced_unet"]["class"] is _factory_class("enhanced_unet")

    def test_it_resolves_to_the_configurable_unet(self):
        from spectramr.models.reconstruction.unet import UNet

        assert MODEL_REGISTRY["enhanced_unet"]["class"] is UNet
        assert _factory_class("enhanced_unet") is UNet

    def test_the_registry_entry_still_exists(self):
        """Renaming the decorator would otherwise leave a hole.

        ``inference.py`` indexes ``MODEL_REGISTRY[model_type]`` directly, and
        schema validation would fall through to the ``VALID_MODEL_TYPES``
        whitelist -- making a fallback that is provably unreachable today (#978)
        silently load-bearing. The binding in ``stubs.py`` is a plain assignment
        rather than ``setdefault`` so it cannot become import-order dependent.
        """
        assert "enhanced_unet" in MODEL_REGISTRY
