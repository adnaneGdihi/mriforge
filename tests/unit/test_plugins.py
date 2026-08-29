"""Tests for :mod:`mriforge.plugins` — out-of-tree component discovery.

Three layers feed the registries with components defined OUTSIDE the source
tree (a user's pip-installed package or a standalone script):

1. ``importlib.metadata`` entry-points (groups ``mriforge.models`` etc.) — for
   shareable, installed plugins. A broken third-party plugin must *warn*, not
   crash someone else's training.
2. the ``MRIFORGE_PLUGINS`` env var — dotted module paths; a user-declared knob,
   so an unimportable token must *raise* (pitfall #15: no silent skip).
3. ``config.plugins.paths`` — same raise-on-failure contract (handled by
   :func:`import_plugin_paths`).

The module is stdlib-only so it can be imported from every layer
(``models/``, ``core/metrics/``) without a leftward dependency violation.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

import mriforge.plugins as plugins


@pytest.fixture(autouse=True)
def _reset_discovery_cache():
    """Each test starts from a clean discovery cache."""
    plugins._reset_discovery_cache()
    yield
    plugins._reset_discovery_cache()


def _write_plugin(tmp_path, name: str) -> str:
    """Write an importable module ``name`` under ``tmp_path`` (on sys.path)."""
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            # A standalone plugin module. Importing it is the side effect the
            # discovery layers rely on (it would carry @register_* decorators).
            LOADED = True
            """
        )
    )
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return name


def _cleanup_module(name: str) -> None:
    sys.modules.pop(name, None)


def test_import_plugin_paths_imports_the_module(tmp_path):
    name = _write_plugin(tmp_path, "gm_plugin_a")
    assert name not in sys.modules
    try:
        plugins.import_plugin_paths([name])
        assert name in sys.modules
    finally:
        _cleanup_module(name)


def test_import_plugin_paths_raises_on_bad_path():
    with pytest.raises(plugins.PluginImportError, match="no_such_plugin_xyz"):
        plugins.import_plugin_paths(["no_such_plugin_xyz"])


def test_env_var_paths_are_imported(tmp_path, monkeypatch):
    name = _write_plugin(tmp_path, "gm_plugin_env")
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, name)
    try:
        plugins.discover_plugins("mriforge.models")
        assert name in sys.modules
    finally:
        _cleanup_module(name)


def test_env_var_bad_token_raises(monkeypatch):
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, "definitely.not.a.module_xyz")
    with pytest.raises(plugins.PluginImportError, match="module_xyz"):
        plugins.discover_plugins("mriforge.models")


def test_entry_points_are_loaded(monkeypatch):
    loaded: list[str] = []

    class _FakeEP:
        name = "my_unet"

        def load(self):
            loaded.append(self.name)
            return object()

    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda group=None: [_FakeEP()] if group == "mriforge.models" else [],
    )
    plugins.discover_plugins("mriforge.models")
    assert loaded == ["my_unet"]


def test_entry_point_failure_warns_but_does_not_raise(monkeypatch, caplog):
    class _BrokenEP:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda group=None: [_BrokenEP()] if group == "mriforge.models" else [],
    )
    import logging

    with caplog.at_level(logging.WARNING):
        plugins.discover_plugins("mriforge.models")  # must not raise
    assert any("broken" in r.message for r in caplog.records)


def test_env_plugin_model_resolves_in_config(tmp_path, monkeypatch):
    """End-to-end: a model defined OUTSIDE the tree, named via MRIFORGE_PLUGINS,
    fires its @register_model on discovery and is accepted as a config
    ``model_type`` — the headline out-of-tree capability."""
    name = "gm_plugin_int_model"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from mriforge.models.registry import register_model

            @register_model("gm_plugin_int_unet", "reconstruction")
            class GmPluginIntUNet(nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.conv = nn.Conv2d(1, 1, 3, padding=1)

                def forward(self, x):
                    return self.conv(x)
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, name)
    plugins._reset_discovery_cache()

    from mriforge.config.settings import TrainingSettings
    from mriforge.models.registry import list_models

    try:
        # settings_from_dict -> _finalize_from_dict -> populate_model_registry()
        # -> discover_plugins("mriforge.models") imports the env plugin module,
        # firing @register_model; the model_type check then accepts the name.
        settings = TrainingSettings.settings_from_dict(
            {
                "model": {"model_type": "gm_plugin_int_unet"},
                "data": {},
                "optimization": {},
                "logging": {},
            }
        )
        assert settings.model.model_type == "gm_plugin_int_unet"
        assert "gm_plugin_int_unet" in list_models()
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        # Leave the module in sys.modules so a later re-import is a no-op (avoids
        # a register_model collision on the same name across tests).


def test_env_plugin_loss_resolves_by_name(tmp_path, monkeypatch):
    """An out-of-tree @register_loss, named via MRIFORGE_PLUGINS, joins the loss
    registry on discovery (the loss path, previously only proven for models)."""
    name = "gm_plugin_loss_mod"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from mriforge.models.losses.registry import register_loss

            @register_loss("gm_plugin_test_loss", domain="image")
            class GmPluginTestLoss(nn.Module):
                def forward(self, pred, target):
                    return (pred - target).abs().mean()
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, name)
    plugins._reset_discovery_cache()

    from mriforge.models.losses.registry import list_available

    try:
        plugins.discover_plugins("mriforge.losses")
        assert "gm_plugin_test_loss" in list_available()
    finally:
        sys.path.remove(str(tmp_path)) if str(tmp_path) in sys.path else None


def test_env_plugin_metric_resolves_by_name(tmp_path, monkeypatch):
    """An out-of-tree @register_metric, named via MRIFORGE_PLUGINS, resolves on
    discovery (the metric path)."""
    name = "gm_plugin_metric_mod"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            from mriforge.core.metrics.registry import register_metric

            @register_metric("gm_plugin_test_metric")
            class GmPluginTestMetric:
                higher_is_better = True

                def __call__(self, pred, target, **kwargs):
                    return 0.0
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, name)
    plugins._reset_discovery_cache()

    from mriforge.core.metrics.registry import get_metric

    try:
        plugins.discover_plugins("mriforge.metrics")
        assert get_metric("gm_plugin_test_metric") is not None
    finally:
        sys.path.remove(str(tmp_path)) if str(tmp_path) in sys.path else None


def test_entry_point_strategy_short_name_resolves(monkeypatch):
    """A plugin strategy declared via the mriforge.strategies entry-point group
    resolves by short name through the factory (the strategy path)."""
    from mriforge.infrastructure.training import strategy_factory as sf

    # An existing, importable strategy class to point the entry-point at.
    target = (
        "mriforge.infrastructure.training.strategies.reconstruction."
        "ReconstructionTrainingStrategy"
    )

    class _FakeEP:
        name = "my_plugin_strategy"
        value = target

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [_FakeEP()] if group == "mriforge.strategies" else [],
    )
    # Reset the factory's lazy plugin-strategy cache so the fake EP is read.
    monkeypatch.setattr(sf, "_plugin_strategy_paths", None)

    resolved = sf._load_plugin_strategy_paths()
    assert resolved.get("my_plugin_strategy") == target

    # And the factory resolves the short name to the real class.
    cls = sf.TrainingStrategyFactory()._load_strategy_class("my_plugin_strategy")
    assert cls.__name__ == "ReconstructionTrainingStrategy"


def test_config_plugin_cannot_shadow_intree_model_name(tmp_path, monkeypatch):
    """A config plugin (``plugins.paths``) re-registering an in-tree model name
    must RAISE, not silently shadow it (spec §6.1, pitfall #9).

    This is guaranteed by ``_finalize_from_dict`` populating the in-tree registry
    BEFORE importing config plugins: with the reverse order the plugin registers
    first and the in-tree discovery walk's duplicate ``ValueError`` is swallowed
    at ``logger.debug`` — so the plugin wins silently. The test discriminates the
    two orderings (it FAILS under the old plugin-first order).
    """
    from mriforge.config.settings import TrainingSettings
    from mriforge.models import init_registry
    from mriforge.models.registry import MODEL_REGISTRY, register_model

    sentinel = "shadow_probe_model_xyz"
    MODEL_REGISTRY.pop(sentinel, None)

    # Cheap stand-in for the real in-tree walk (which imports the whole model
    # zoo): registers `sentinel` once. Idempotent like the real one.
    def fake_populate() -> None:
        if sentinel not in MODEL_REGISTRY:
            register_model(sentinel, "reconstruction")(
                type("InTreeSentinel", (object,), {})
            )

    monkeypatch.setattr(init_registry, "populate_model_registry", fake_populate)

    # A config plugin re-registering the SAME name with a DIFFERENT class.
    (tmp_path / "shadow_plugin.py").write_text(
        textwrap.dedent(
            f"""
            from mriforge.models.registry import register_model

            @register_model("{sentinel}", "reconstruction")
            class ShadowPlugin:
                pass
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    plugins._reset_discovery_cache()
    try:
        with pytest.raises(
            (ValueError, plugins.PluginImportError), match=sentinel
        ):
            TrainingSettings.settings_from_dict(
                {
                    "model": {"model_type": sentinel},
                    "data": {},
                    "optimization": {},
                    "logging": {},
                    "plugins": {"enabled": True, "paths": ["shadow_plugin"]},
                }
            )
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        MODEL_REGISTRY.pop(sentinel, None)
        sys.modules.pop("shadow_plugin", None)


def test_discovery_is_idempotent(tmp_path, monkeypatch):
    name = _write_plugin(tmp_path, "gm_plugin_once")
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, name)
    try:
        plugins.discover_plugins("mriforge.models")
        # Drop the module record; a second call must NOT re-import it
        # (the env layer runs once), so it stays absent.
        sys.modules.pop(name, None)
        plugins.discover_plugins("mriforge.losses")
        assert name not in sys.modules
    finally:
        _cleanup_module(name)
