"""Regression test for the audit-18a-F3 fix.

Adds ``spectramr.models.gans`` to the model-registry walk list so that the
@register_model decorators in [stylegan_variants.py, swin_kan_gan.py,
swin_transformer_kan.py, stylegan_kan.py] actually fire. Pre-fix, only
the 3 models that ``src/models/generators/__init__.py`` imported
transitively (``kan_gan``, ``kan_discriminator``, ``vit_kan_generator``)
made it into the registry; the other 6 were silent ghosts.

Re-removing ``spectramr.models.gans`` from the walk list would put us back in
CLAUDE.md pitfall #9 territory (advertised contract that does not fire).
"""

from __future__ import annotations

import pytest


EXPECTED_GANS_MODELS = (
    "kan_gan",
    "kan_discriminator",
    "vit_kan_generator",
    "stylegan2",
    "stylegan_discriminator",
    "stylegan_kan_gen",
    "swin_kan_transformer",
    "swin_kan_generator",
    "style_kan_gan",
)


@pytest.fixture(scope="module")
def populated_registry():
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    return MODEL_REGISTRY


def test_all_viable_gans_decorators_fire(populated_registry) -> None:
    missing = [m for m in EXPECTED_GANS_MODELS if m not in populated_registry]
    assert not missing, (
        f"spectramr.models.gans decorators no longer fire: {missing}. "
        "Check that 'spectramr.models.gans' is still in the walk list in "
        "src/spectramr/models/init_registry.py — see audit 18a F3."
    )


def test_standard_gan_dc_dependency_is_wired(populated_registry) -> None:
    """``physics_informed_unet`` historically referenced a missing
    ``PhysicsInformedDataConsistency`` class. The module was kept and
    the dependency rewired to ``AdaptiveDataConsistency``; this test
    pins that the alias remains and the model is registered.
    """
    from spectramr.models.gans.standard_gan import (
        AdaptiveDataConsistency,
        PhysicsInformedDataConsistency,
        PhysicsInformedUNet,
    )

    assert PhysicsInformedDataConsistency is AdaptiveDataConsistency
    assert PhysicsInformedUNet is not None
    assert "physics_informed_unet" in populated_registry
