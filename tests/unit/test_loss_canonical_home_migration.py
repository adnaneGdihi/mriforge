"""Phase-5 canonical-home migration regression tests.

Locks ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5 + CLAUDE.md
pitfall #12: ``@register_loss`` decorators live exclusively under
``src/models/losses/``. The four classes that previously lived in
``src/infrastructure/physics/integration.py`` and
``src/models/ot/optimal_transport.py`` were moved 2026-05-14.
"""

from __future__ import annotations

import inspect

from mriforge.models.losses import list_available
from mriforge.models.losses.optimal_transport_losses import (
    DynamicOTFlow,
    KIDOTLoss,
    SinkhornDistance,
)
from mriforge.models.losses.physics_informed_integration_loss import PhysicsInformedLoss


def test_physics_informed_integration_canonical_home() -> None:
    """``PhysicsInformedLoss`` lives in ``mriforge.models.losses``."""
    assert PhysicsInformedLoss.__module__ == (
        "mriforge.models.losses.physics_informed_integration_loss"
    )


def test_sinkhorn_canonical_home() -> None:
    assert SinkhornDistance.__module__ == (
        "mriforge.models.losses.optimal_transport_losses"
    )


def test_dynamic_ot_canonical_home() -> None:
    assert DynamicOTFlow.__module__ == (
        "mriforge.models.losses.optimal_transport_losses"
    )


def test_kidot_canonical_home() -> None:
    assert KIDOTLoss.__module__ == (
        "mriforge.models.losses.optimal_transport_losses"
    )


def test_all_four_migrated_losses_are_registered() -> None:
    """The decorator fires on package import; registry must include all four keys."""
    names = list_available()
    assert "physics_informed_integration" in names
    assert "sinkhorn" in names
    assert "dynamic_ot" in names
    assert "kidot" in names


def test_physics_integration_module_no_longer_imports_register_loss() -> None:
    """``physics/integration.py`` no longer pulls in the loss registry.

    The circular-import that fired during the Phase-5 migration was caused
    by ``physics/integration.py`` importing ``register_loss`` while also
    being a dependency of the losses package. Removing that import broke
    the cycle.
    """
    from mriforge.infrastructure.physics import integration

    src = inspect.getsource(integration)
    assert "from mriforge.models.losses.registry import register_loss" not in src
    # The class itself is gone from this module too.
    assert "class PhysicsInformedLoss" not in src


def test_ot_module_no_longer_decorates_losses() -> None:
    """``models/ot/optimal_transport.py`` no longer carries ``@register_loss``."""
    from mriforge.models.ot import optimal_transport

    src = inspect.getsource(optimal_transport)
    assert "@register_loss(" not in src
    assert "class SinkhornDistance" not in src
    assert "class DynamicOTFlow" not in src
    assert "class KIDOTLoss" not in src
    # The non-loss primitives stay.
    assert "class VelocityFieldNetwork" in src
    assert "def sinkhorn_knopp" in src


def test_ot_package_init_does_not_re_export_losses() -> None:
    """``src/models/ot/__init__.py`` no longer re-exports the moved loss classes.

    The re-export caused a circular import: losses depend on
    ``VelocityFieldNetwork`` (here), and the ``__init__`` re-exporting
    those losses formed a cycle.
    """
    from mriforge.models import ot

    assert not hasattr(ot, "SinkhornDistance")
    assert not hasattr(ot, "DynamicOTFlow")
    assert not hasattr(ot, "KIDOTLoss")
    assert hasattr(ot, "VelocityFieldNetwork")
    assert hasattr(ot, "sinkhorn_knopp")
