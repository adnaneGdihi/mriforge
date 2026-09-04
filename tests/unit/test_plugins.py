"""Tests for :mod:`spectramr.plugins` — out-of-tree component discovery.

Three layers feed the registries with components defined OUTSIDE the source
tree (a user's pip-installed package or a standalone script):

1. ``importlib.metadata`` entry-points (groups ``spectramr.models`` etc.) — for
   shareable, installed plugins. A broken third-party plugin must *warn*, not
   crash someone else's training.
2. the ``SPECTRAMR_PLUGINS`` env var — dotted module paths; a user-declared knob,
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

import spectramr.plugins as plugins


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
        plugins.discover_plugins("spectramr.models")
        assert name in sys.modules
    finally:
        _cleanup_module(name)


def test_env_var_bad_token_raises(monkeypatch):
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, "definitely.not.a.module_xyz")
    with pytest.raises(plugins.PluginImportError, match="module_xyz"):
        plugins.discover_plugins("spectramr.models")


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
        lambda group=None: [_FakeEP()] if group == "spectramr.models" else [],
    )
    plugins.discover_plugins("spectramr.models")
    assert loaded == ["my_unet"]


def test_entry_point_failure_warns_but_does_not_raise(monkeypatch, caplog):
    class _BrokenEP:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda group=None: [_BrokenEP()] if group == "spectramr.models" else [],
    )
    import logging

    with caplog.at_level(logging.WARNING):
        plugins.discover_plugins("spectramr.models")  # must not raise
    assert any("broken" in r.message for r in caplog.records)


def test_env_plugin_model_resolves_in_config(tmp_path, monkeypatch):
    """End-to-end: a model defined OUTSIDE the tree, named via SPECTRAMR_PLUGINS,
    fires its @register_model on discovery and is accepted as a config
    ``model_type`` — the headline out-of-tree capability."""
    name = "gm_plugin_int_model"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from spectramr.models.registry import register_model

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

    from spectramr.config.settings import TrainingSettings
    from spectramr.models.registry import list_models

    try:
        # settings_from_dict -> _finalize_from_dict -> populate_model_registry()
        # -> discover_plugins("spectramr.models") imports the env plugin module,
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
    """An out-of-tree @register_loss, named via SPECTRAMR_PLUGINS, joins the loss
    registry on discovery (the loss path, previously only proven for models)."""
    name = "gm_plugin_loss_mod"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from spectramr.models.losses.registry import register_loss

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

    from spectramr.models.losses.registry import list_available

    try:
        plugins.discover_plugins("spectramr.losses")
        assert "gm_plugin_test_loss" in list_available()
    finally:
        sys.path.remove(str(tmp_path)) if str(tmp_path) in sys.path else None


def test_env_plugin_metric_resolves_by_name(tmp_path, monkeypatch):
    """An out-of-tree @register_metric, named via SPECTRAMR_PLUGINS, resolves on
    discovery (the metric path)."""
    name = "gm_plugin_metric_mod"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """
            from spectramr.core.metrics.registry import register_metric

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

    from spectramr.core.metrics.registry import get_metric

    try:
        plugins.discover_plugins("spectramr.metrics")
        assert get_metric("gm_plugin_test_metric") is not None
    finally:
        sys.path.remove(str(tmp_path)) if str(tmp_path) in sys.path else None


def test_entry_point_strategy_short_name_resolves(monkeypatch):
    """A plugin strategy declared via the spectramr.strategies entry-point group
    resolves by short name through the factory (the strategy path)."""
    from spectramr.infrastructure.training import strategy_factory as sf

    # An existing, importable strategy class to point the entry-point at.
    target = (
        "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
    )

    class _FakeEP:
        name = "my_plugin_strategy"
        value = target

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [_FakeEP()] if group == "spectramr.strategies" else [],
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
    from spectramr.config.settings import TrainingSettings
    from spectramr.models import init_registry
    from spectramr.models.registry import MODEL_REGISTRY, register_model

    sentinel = "shadow_probe_model_xyz"
    MODEL_REGISTRY.pop(sentinel, None)

    # Cheap stand-in for the real in-tree walk (which imports the whole model
    # zoo): registers `sentinel` once. Idempotent like the real one.
    def fake_populate() -> None:
        if sentinel not in MODEL_REGISTRY:
            register_model(sentinel, "reconstruction")(type("InTreeSentinel", (object,), {}))

    monkeypatch.setattr(init_registry, "populate_model_registry", fake_populate)

    # A config plugin re-registering the SAME name with a DIFFERENT class.
    (tmp_path / "shadow_plugin.py").write_text(
        textwrap.dedent(
            f"""
            from spectramr.models.registry import register_model

            @register_model("{sentinel}", "reconstruction")
            class ShadowPlugin:
                pass
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    plugins._reset_discovery_cache()
    try:
        with pytest.raises((ValueError, plugins.PluginImportError), match=sentinel):
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
        plugins.discover_plugins("spectramr.models")
        # Drop the module record; a second call must NOT re-import it
        # (the env layer runs once), so it stays absent.
        sys.modules.pop(name, None)
        plugins.discover_plugins("spectramr.losses")
        assert name not in sys.modules
    finally:
        _cleanup_module(name)


# ---------------------------------------------------------------------------
# Re-entrancy: discovery runs at MODULE-IMPORT time from
# ``core/metrics/__init__`` and ``models/losses/__init__``, so a plugin that
# imports anything from ``spectramr`` re-enters discovery while its own body is
# still executing. These are the shapes that regression took (CLAUDE.md
# non-negotiable 15: one planted violation per shape, not just the easy one).
# ---------------------------------------------------------------------------

#: The three documented ways a plugin may reach the registration decorators.
#: All three MUST register. ``spectramr.api`` is the one ``api.py`` and
#: ``docs/scripting_api.rst`` tell users to write, and it was the one that
#: silently failed -- its eager ``pipelines.fit`` chain hit a half-built module
#: mid-initialisation, the plugin body was abandoned, and nothing was raised.
_IMPORT_SPELLINGS = {
    "root_lazy": "from spectramr import register_model",
    "canonical": "from spectramr.models.registry import register_model",
    "api_facade": "from spectramr.api import register_model",
}


@pytest.mark.parametrize("spelling", sorted(_IMPORT_SPELLINGS))
def test_plugin_registers_under_every_documented_import_spelling(tmp_path, spelling):
    """A SPECTRAMR_PLUGINS module registers regardless of how it imports the decorator.

    Runs in a SUBPROCESS on purpose: the defect is an import-ordering one, and
    the registries plus ``populate_model_registry``'s idempotency flag are
    process-global, so an in-process test cannot reproduce the first-import
    window that broke.
    """
    import os
    import subprocess

    (tmp_path / "spelling_plugin.py").write_text(
        textwrap.dedent(
            f"""
            import torch.nn as nn
            {_IMPORT_SPELLINGS[spelling]}

            @register_model("spelling_probe_net", "reconstruction")
            class SpellingProbeNet(nn.Module):
                def __init__(self, in_channels=1, out_channels=1, **kw):
                    super().__init__()
                    self.c = nn.Conv2d(in_channels, out_channels, 3, padding=1)

                def forward(self, x):
                    return self.c(x)
            """
        )
    )
    probe = textwrap.dedent(
        """
        from spectramr.models.init_registry import populate_model_registry
        populate_model_registry()
        from spectramr.models.registry import MODEL_REGISTRY
        print("REGISTERED:", "spelling_probe_net" in MODEL_REGISTRY)
        """
    )
    env = {
        **os.environ,
        "SPECTRAMR_PLUGINS": "spelling_plugin",
        "PYTHONPATH": str(tmp_path),
        "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=600
    )
    assert "REGISTERED: True" in out.stdout, (
        f"plugin using {_IMPORT_SPELLINGS[spelling]!r} did not register.\n"
        f"stdout={out.stdout[-2000:]}\nstderr={out.stderr[-2000:]}"
    )


def test_reentrant_import_does_not_cache_an_unfinished_module(tmp_path, monkeypatch):
    """``_imported_paths`` must never name a module absent from ``sys.modules``.

    The poisoning shape: a re-entrant ``_import_path`` found the half-built
    module in ``sys.modules``, returned it without executing the body, and
    cached it as done -- so every later ``discover_plugins`` short-circuited and
    the plugin could never register, silently.

    Both halves below are load-bearing. The re-entrant call is what wrongly
    cached the path; the LATER failure is what makes the cache a lie, because
    Python then drops the half-built module from ``sys.modules``. A module that
    re-enters and then *succeeds* leaves a consistent cache and cannot reproduce
    the defect -- that is the easy shape this test must not settle for
    (non-negotiable 15).
    """
    (tmp_path / "reentrant_plugin.py").write_text(
        textwrap.dedent(
            """
            import spectramr.plugins as _p
            # Re-enter discovery from inside our own module body — exactly what
            # `from spectramr.api import ...` does via the metrics/losses seams.
            _p.discover_plugins("spectramr.models")
            # ...and then fail, as the real plugin did when the eager api chain
            # hit a partially-initialised module.
            raise ImportError("cannot import name 'x' (most likely a circular import)")
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(plugins.PLUGIN_ENV_VAR, "reentrant_plugin")
    try:
        with pytest.raises(plugins.PluginImportError):
            plugins.discover_plugins("spectramr.models")
        assert plugins._imported_paths <= set(sys.modules), (
            "discovery cached a path that is not in sys.modules — the module was "
            "never fully executed, so its @register_* decorators never fired, yet "
            "every later discover_plugins() will now short-circuit. "
            f"cached={plugins._imported_paths}"
        )
        assert "reentrant_plugin" not in plugins._imported_paths, (
            "a plugin whose import FAILED was cached as imported"
        )
    finally:
        _cleanup_module("reentrant_plugin")


def test_broken_env_token_is_not_blamed_on_an_in_tree_package(tmp_path):
    """A bad SPECTRAMR_PLUGINS token must name the PLUGIN, not ``spectramr.models.*``.

    The walk's per-package ``except`` used to swallow the ``PluginImportError``
    raised from the import-time discovery seam nested inside it, and report
    "every @register_model in spectramr.models.generators is MISSING" — sending
    the reader to fix an in-tree package that was never broken.
    """
    import os
    import subprocess

    (tmp_path / "broken_plugin.py").write_text('raise RuntimeError("deliberately broken")\n')
    probe = textwrap.dedent(
        """
        from spectramr.models.init_registry import populate_model_registry
        try:
            populate_model_registry()
            print("NO_RAISE")
        except Exception as exc:
            print("RAISED:", type(exc).__name__, exc)
        """
    )
    env = {
        **os.environ,
        "SPECTRAMR_PLUGINS": "broken_plugin",
        "PYTHONPATH": str(tmp_path),
        "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=600
    )
    combined = out.stdout + out.stderr
    assert "RAISED: PluginImportError" in out.stdout, (
        f"expected a typed raise, got {combined[-2000:]}"
    )
    assert "broken_plugin" in out.stdout
    assert "Model discovery could not import spectramr." not in combined, (
        "a broken user plugin was misreported as an in-tree package import failure:\n"
        f"{combined[-2000:]}"
    )


def test_a_plugin_declared_after_the_first_config_load_is_still_discovered(tmp_path):
    """Discovery must not depend on what ran before it.

    TWO independent latches made ``SPECTRAMR_PLUGINS`` a once-per-process read,
    and each alone was enough to lose the plugin:

    * ``populate_model_registry`` returned early on ``_REGISTRY_POPULATED``, and
      its ``discover_plugins`` call sat BELOW that return -- so discovery was
      reachable on the first call and never again;
    * ``_discover_env_paths`` latched ``_env_imported`` after its first clean
      pass -- so even reaching it a second time re-read nothing.

    Loading any config triggers the first population (``settings_from_dict``
    validates ``model_type`` through the registry), so a long-lived process --
    notebook, server, campaign runner -- that set the env var afterwards never
    picked the plugin up. It is also what made
    ``test_env_plugin_model_resolves_in_config`` pass alone and fail in the suite
    (#1637).

    Subprocess: both latches are process-global module state.
    """
    import os
    import subprocess

    (tmp_path / "late_plugin.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from spectramr.models.registry import register_model

            @register_model("late_probe_net", "reconstruction")
            class LateProbeNet(nn.Module):
                def __init__(self, in_channels=1, out_channels=1, **kw):
                    super().__init__()
                    self.c = nn.Conv2d(in_channels, out_channels, 3, padding=1)

                def forward(self, x):
                    return self.c(x)
            """
        )
    )
    probe = textwrap.dedent(
        """
        import os
        from spectramr.config.settings import TrainingSettings as TS

        base = {"model": {"model_type": "unet"}, "data": {}, "optimization": {}, "logging": {}}
        TS.settings_from_dict(base)                      # populate the registry FIRST
        os.environ["SPECTRAMR_PLUGINS"] = "late_plugin"   # ...declare the plugin AFTER

        from spectramr.models.init_registry import populate_model_registry
        populate_model_registry()
        from spectramr.models.registry import MODEL_REGISTRY
        print("DISCOVERED:", "late_probe_net" in MODEL_REGISTRY)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
        },
        timeout=600,
    )
    assert "DISCOVERED: True" in out.stdout, (
        "a plugin declared after the first config load was never discovered — "
        f"discovery is still order-dependent.\nstdout={out.stdout[-1500:]}\n"
        f"stderr={out.stderr[-1500:]}"
    )


def test_a_bad_env_token_still_raises_on_every_call():
    """Removing the latch must not weaken fail-fast.

    ``_env_imported`` was documented as giving "consistent fail-fast" — a bad
    token re-raising rather than being skipped on retry. Per-path tracking keeps
    that: a token that failed never enters ``_imported_paths``.
    """
    import os

    os.environ[plugins.PLUGIN_ENV_VAR] = "definitely_not_a_module_xyz"
    try:
        for _ in range(3):
            with pytest.raises(plugins.PluginImportError, match="definitely_not_a_module_xyz"):
                plugins.discover_plugins("spectramr.models")
    finally:
        os.environ.pop(plugins.PLUGIN_ENV_VAR, None)


def test_out_of_tree_dataset_type_is_accepted_by_a_config(tmp_path):
    """Datasets are pluggable, like models / losses / metrics.

    ``dataset_type`` is a dispatch key resolved through ``DATASET_REGISTRY``,
    but it was validated against a frozen 21-name literal, so a dataset
    registered out of tree was rejected at load time however correctly it was
    registered. Models, losses and metrics were pluggable; datasets alone were
    not.

    Subprocess: ``DATASET_REGISTRY`` and ``DatasetInstantiator._REGISTERED`` are
    process-global, so an in-process test cannot reproduce a clean first load.
    """
    import os
    import subprocess

    (tmp_path / "ds_plugin.py").write_text(
        textwrap.dedent(
            """
            import torch
            from torch.utils.data import Dataset
            from spectramr.data.datasets.registry import register_dataset

            class _DS(Dataset):
                def __len__(self): return 4
                def __getitem__(self, i):
                    return {"input": torch.randn(1, 8, 8), "target": torch.randn(1, 8, 8)}

            def _make(config, train_tfm=None, val_tfm=None):
                return _DS(), _DS()

            register_dataset("byoc_probe_dataset", _make, indexed=False, serves="probe")
            """
        )
    )
    probe = textwrap.dedent(
        """
        from spectramr.config.settings import TrainingSettings as TS
        base = {"model": {"model_type": "unet"}, "optimization": {}, "logging": {}}
        cfg = TS.settings_from_dict({**base, "data": {"dataset_type": "byoc_probe_dataset"}})
        print("ACCEPTED:", cfg.data.dataset_type)
        try:
            TS.settings_from_dict({**base, "data": {"dataset_type": "not_a_dataset"}})
            print("BOGUS_ACCEPTED")
        except ValueError:
            print("BOGUS_REJECTED")
        """
    )
    env = {
        **os.environ,
        "SPECTRAMR_PLUGINS": "ds_plugin",
        "PYTHONPATH": str(tmp_path),
        "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=600
    )
    assert "ACCEPTED: byoc_probe_dataset" in out.stdout, (
        f"out-of-tree dataset_type was rejected.\nstdout={out.stdout[-1500:]}\n"
        f"stderr={out.stderr[-1500:]}"
    )
    # The gate must still be a gate: opening it to the registry must not open it
    # to anything at all.
    assert "BOGUS_REJECTED" in out.stdout, out.stdout[-800:]


def test_datasets_is_a_declared_entry_point_group():
    """The group must be declared AND consumed, not merely listed.

    Adding a name to ``ENTRY_POINT_GROUPS`` without a ``discover_plugins`` call
    site is the registered-but-unwired shape non-negotiable 16 is about.
    """
    import inspect

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

    assert "spectramr.datasets" in plugins.ENTRY_POINT_GROUPS
    source = inspect.getsource(DatasetInstantiator._ensure_registered)
    assert 'discover_plugins("spectramr.datasets")' in source, (
        "the datasets entry-point group is declared but nothing discovers it"
    )
