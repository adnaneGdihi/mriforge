"""Tests for the public scripting surface: ``mriforge.api`` + top-level re-exports.

The framework is pip-installable, so a user's script does
``from mriforge import register_model, fit, settings_from_dict`` (or
``from mriforge.api import ...``). The top-level re-exports are LAZY (PEP 562
``__getattr__``) so ``import mriforge`` does not eagerly drag in the whole torch
+ models import chain — see gotcha G1. ``mriforge.api`` is a leaf facade that may
import eagerly because it is only imported on demand.
"""

from __future__ import annotations

import pytest


def test_top_level_register_model_resolves_to_real_object():
    import mriforge
    from mriforge.models.registry import register_model as canonical

    assert mriforge.register_model is canonical


def test_top_level_public_callables_present():
    import mriforge

    for name in (
        "register_loss",
        "register_metric",
        "settings_from_dict",
        "make_model",
        "make_optimizer",
        "make_dataset",
        "make_dataloader",
        "fit",
        "Trainer",
    ):
        assert callable(getattr(mriforge, name)), name


def test_unknown_top_level_attribute_raises():
    import mriforge

    with pytest.raises(AttributeError):
        _ = mriforge.does_not_exist_xyz


def test_dunder_all_advertises_public_names():
    import mriforge

    for name in (
        "register_model",
        "register_loss",
        "register_metric",
        "settings_from_dict",
        "make_model",
    ):
        assert name in mriforge.__all__, name


def test_api_facade_reexports():
    from mriforge import api

    for name in (
        "register_model",
        "register_loss",
        "register_metric",
        "settings_from_dict",
        "make_model",
        "make_optimizer",
        "make_dataset",
        "make_dataloader",
        "fit",
        "Trainer",
    ):
        assert callable(getattr(api, name)), name


def test_settings_from_dict_via_public_surface():
    import mriforge

    settings = mriforge.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {},
            "optimization": {},
            "logging": {},
        }
    )
    assert settings.model.model_type == "unet"
