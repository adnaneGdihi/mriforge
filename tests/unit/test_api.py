"""Tests for the public scripting surface: ``spectramr.api`` + top-level re-exports.

The framework is pip-installable, so a user's script does
``from spectramr import register_model, fit, settings_from_dict`` (or
``from spectramr.api import ...``). The top-level re-exports are LAZY (PEP 562
``__getattr__``) so ``import spectramr`` does not eagerly drag in the whole torch
+ models import chain — see gotcha G1. ``spectramr.api`` is a leaf facade that may
import eagerly because it is only imported on demand.
"""

from __future__ import annotations

import pytest


def test_top_level_register_model_resolves_to_real_object():
    import spectramr
    from spectramr.models.registry import register_model as canonical

    assert spectramr.register_model is canonical


def test_top_level_public_callables_present():
    import spectramr

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
        assert callable(getattr(spectramr, name)), name


def test_unknown_top_level_attribute_raises():
    import spectramr

    with pytest.raises(AttributeError):
        _ = spectramr.does_not_exist_xyz


def test_dunder_all_advertises_public_names():
    import spectramr

    for name in (
        "register_model",
        "register_loss",
        "register_metric",
        "settings_from_dict",
        "make_model",
    ):
        assert name in spectramr.__all__, name


def test_api_facade_reexports():
    from spectramr import api

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
    import spectramr

    settings = spectramr.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {},
            "optimization": {},
            "logging": {},
        }
    )
    assert settings.model.model_type == "unet"


def test_api_and_root_export_the_identical_objects():
    """``spectramr.api.X is spectramr.X`` — the identity the docstring promises.

    Both surfaces now resolve through the single ``_LAZY_EXPORTS`` table, so a
    name added to one cannot silently miss the other.
    """
    import spectramr
    import spectramr.api as api

    for name in api.__all__:
        assert getattr(api, name) is getattr(spectramr, name), name


def test_api_unknown_attribute_raises_naming_this_module():
    """An unknown export raises, and the message names the import the user wrote."""
    import pytest

    import spectramr.api as api

    with pytest.raises(AttributeError, match=r"spectramr\.api.*not_a_real_export"):
        _ = api.not_a_real_export


def test_documented_register_model_snippet_executes():
    """The ``docs/scripting_api.rst`` / ``api.py`` docstring snippet must RUN.

    It shipped for a long time as ``@register_model("my_unet")`` — one argument
    — which raises ``TypeError`` because ``training_mode`` is required. A prose
    example nobody executes is a docs bug that reads like working code, so pin
    it with a test that actually calls the decorator.
    """
    import torch.nn as nn

    from spectramr.api import register_model
    from spectramr.models.registry import MODEL_REGISTRY

    name = "docs_snippet_probe_unet"
    MODEL_REGISTRY.pop(name, None)

    @register_model(name, "reconstruction")
    class DocsSnippetProbeUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, **kwargs):
            super().__init__()
            self.c = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        def forward(self, x):
            return self.c(x)

    try:
        assert name in MODEL_REGISTRY
    finally:
        MODEL_REGISTRY.pop(name, None)
