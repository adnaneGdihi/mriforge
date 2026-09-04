"""Tests for :class:`spectramr.config.schemas.plugins.PluginsConfigSchema`.

The ``plugins`` block is the YAML-facing wire for out-of-tree component
discovery: a list of dotted module paths the framework imports at registry
population so a user's ``@register_model`` / ``@register_loss`` decorators fire.
Because the parent :class:`TrainingSettings` is ``extra="forbid"``, the block
must be a *typed* field (an untyped ``plugins:`` key would be rejected outright);
and because it is a top-level opt-in subsystem it follows the
``optimization``/``logging`` precedent of ``extra="forbid"`` (a misspelled knob
must raise, not silently no-op — pitfall #15).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.plugins import PluginsConfigSchema


class TestPluginsConfigSchema:
    def test_defaults_construct_empty_and_disabled(self) -> None:
        cfg = PluginsConfigSchema()
        assert cfg.enabled is False
        assert cfg.paths == []

    def test_hooks_key_is_rejected(self) -> None:
        # ``plugins.hooks`` was advertised but never read (pitfall #15 — an
        # unwired knob). It was removed; ``extra="forbid"`` must now reject it
        # so it cannot silently re-creep as a no-op.
        with pytest.raises(ValidationError):
            PluginsConfigSchema(hooks={"sparsity": {"lambda": 1.0}})  # type: ignore[call-arg]

    def test_accepts_dotted_import_paths(self) -> None:
        cfg = PluginsConfigSchema(
            enabled=True,
            paths=["mypkg.models.my_unet", "mypkg.losses.my_loss"],
        )
        assert cfg.enabled is True
        assert cfg.paths == ["mypkg.models.my_unet", "mypkg.losses.my_loss"]

    def test_unknown_key_is_rejected(self) -> None:
        # A misspelled knob must raise, never silently vanish (pitfall #15).
        with pytest.raises(ValidationError):
            PluginsConfigSchema(enabld=True)  # type: ignore[call-arg]

    def test_is_frozen(self) -> None:
        cfg = PluginsConfigSchema()
        with pytest.raises(ValidationError):
            cfg.enabled = True  # type: ignore[misc]
