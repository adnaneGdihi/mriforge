"""Regression tests for the strict duplicate-registration guards.

These tests pin down the behaviour added 2026-05-09 in response to
``TODO/audit/00_implementation_tracker.md`` items A5/A9. The three
registries (model / loss / metric) now refuse to silently overwrite an
existing registration with a different class, and refuse to silently
remap an alias from one canonical name to another.

Same-class re-registration is still idempotent so test reloads and
``importlib.reload`` are safe.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from mriforge.core.metrics.registry import MetricsRegistry, register_metric
from mriforge.models.losses.registry import LossRegistry, register_loss
from mriforge.models.registry import MODEL_REGISTRY, register_model


# ---------------------------------------------------------------------------
# LossRegistry
# ---------------------------------------------------------------------------


class _LossA(nn.Module):
    pass


class _LossB(nn.Module):
    pass


def test_loss_registry_idempotent_same_class() -> None:
    """Re-registering the same class with the same name is a silent no-op."""
    register_loss(name="_test_idempotent_loss")(_LossA)
    # Second registration of the same class must not raise.
    register_loss(name="_test_idempotent_loss")(_LossA)
    assert LossRegistry._custom_losses["_test_idempotent_loss"] is _LossA


def test_loss_registry_rejects_different_class_same_name() -> None:
    """Registering a different class under an existing canonical name raises."""
    register_loss(name="_test_dup_loss")(_LossA)
    with pytest.raises(ValueError, match="already registered"):
        register_loss(name="_test_dup_loss")(_LossB)


def test_loss_registry_rejects_alias_remap() -> None:
    """An alias may not silently flip from one canonical to another."""
    register_loss(name="_test_alias_owner", aliases=["_shared_alias"])(_LossA)
    with pytest.raises(ValueError, match="already maps to canonical"):
        register_loss(name="_test_other_owner", aliases=["_shared_alias"])(_LossB)


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


class _MetricA:
    pass


class _MetricB:
    pass


def test_metric_registry_idempotent_same_class() -> None:
    """Re-registering the same metric class is a silent no-op."""
    register_metric("_test_idempotent_metric")(_MetricA)
    register_metric("_test_idempotent_metric")(_MetricA)
    assert MetricsRegistry._metrics["_test_idempotent_metric"] is _MetricA


def test_metric_registry_rejects_different_class_same_name() -> None:
    register_metric("_test_dup_metric")(_MetricA)
    with pytest.raises(ValueError, match="already registered"):
        register_metric("_test_dup_metric")(_MetricB)


def test_metric_registry_rejects_alias_remap() -> None:
    register_metric("_test_metric_owner", aliases=["_shared_metric_alias"])(_MetricA)
    with pytest.raises(ValueError, match="already maps to canonical"):
        register_metric(
            "_test_metric_other_owner", aliases=["_shared_metric_alias"]
        )(_MetricB)


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------


class _ModelA(nn.Module):
    pass


class _ModelB(nn.Module):
    pass


def test_model_registry_idempotent_same_class() -> None:
    register_model(name="_test_idempotent_model", training_mode="reconstruction")(
        _ModelA
    )
    # Second registration with same class must not raise.
    register_model(name="_test_idempotent_model", training_mode="reconstruction")(
        _ModelA
    )
    assert MODEL_REGISTRY["_test_idempotent_model"]["class"] is _ModelA


def test_model_registry_rejects_different_class_same_name() -> None:
    register_model(name="_test_dup_model", training_mode="reconstruction")(_ModelA)
    with pytest.raises(ValueError, match="already registered"):
        register_model(name="_test_dup_model", training_mode="diffusion")(_ModelB)


# ---------------------------------------------------------------------------
# Capability-downgrade guard (the bloch_mamba_v2 scar)
# ---------------------------------------------------------------------------


class _CapModel(nn.Module):
    pass


def test_model_registry_rejects_capability_downgrade() -> None:
    """A same-class re-registration with EMPTY caps must not clobber declared
    caps (a bare ``register_model(name, mode)(Cls)`` after a decorator with
    full ``ModelCapabilities`` silently disabled the audit checks)."""
    register_model(
        name="_test_caps_downgrade",
        training_mode="reconstruction",
        output_domain="kspace",
        spatial_dims=(2,),
    )(_CapModel)
    with pytest.raises(ValueError, match="EMPTY capabilities"):
        register_model(name="_test_caps_downgrade", training_mode="reconstruction")(
            _CapModel
        )


def test_model_registry_downgrade_guard_allows_override() -> None:
    """``override=True`` is the explicit escape hatch for the downgrade guard."""
    register_model(
        name="_test_caps_override",
        training_mode="reconstruction",
        output_domain="kspace",
    )(_CapModel)
    # Must not raise with override=True.
    register_model(
        name="_test_caps_override", training_mode="reconstruction", override=True
    )(_CapModel)
    assert MODEL_REGISTRY["_test_caps_override"]["capabilities"].output_domain is None


def test_model_registry_reregister_with_same_caps_is_idempotent() -> None:
    """Re-registering with the SAME (non-empty) caps is fine — not a downgrade."""
    register_model(
        name="_test_caps_same",
        training_mode="reconstruction",
        output_domain="image",
    )(_CapModel)
    register_model(
        name="_test_caps_same",
        training_mode="reconstruction",
        output_domain="image",
    )(_CapModel)
    assert MODEL_REGISTRY["_test_caps_same"]["capabilities"].output_domain == "image"
