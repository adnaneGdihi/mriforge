"""Tests for the ``make_*`` config-driven factory helpers.

Historically these took ``config_path: Path`` only and called
``TrainingSettings.from_yaml`` internally, forcing a pure-Python caller to
round-trip an in-memory config through a temp file. The scripting-API work
adds an optional ``config: TrainingSettings`` parameter so the helpers accept
an already-built (in-memory) SSOT directly, while keeping the path form
backward-compatible. Passing neither must raise (no silent default).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.config.settings import TrainingSettings
from mriforge.pipelines.make import make_optimizer


def _in_memory_settings() -> TrainingSettings:
    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
        }
    )


def test_make_optimizer_accepts_in_memory_config():
    """An in-memory ``TrainingSettings`` builds an optimizer with no YAML file."""
    settings = _in_memory_settings()
    model = torch.nn.Linear(4, 4)

    optimizer, metadata = make_optimizer(config=settings, model=model)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metadata["learning_rate"] == 1e-4
    # Provenance reflects the in-memory origin (no path).
    assert metadata["config_path"] == "<in-memory>"


def test_make_optimizer_requires_path_or_config():
    """Passing neither ``config_path`` nor ``config`` must raise (no default)."""
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="config_path or config"):
        make_optimizer(model=model)


def test_make_optimizer_still_accepts_path(tmp_path):
    """Backward compat: the original ``config_path`` form keeps working."""
    import yaml as _yaml

    cfg = {
        "config_version": "1.0",
        "model": {"model_type": "unet"},
        "data": {},
        "optimization": {"learning_rate": 2e-4},
        "logging": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    model = torch.nn.Linear(4, 4)

    optimizer, metadata = make_optimizer(config_path=path, model=model)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metadata["learning_rate"] == 2e-4
    assert metadata["config_path"] == str(path)


def test_make_reads_a_declared_dataset_field() -> None:
    """`make_dataset` / `make_dataloader` read `config.data.dataset_name` on
    their first line. It was never a DataConfigSchema field, so both public
    entry points raised AttributeError for every config; no test covered them.
    They now read `dataset_type`, and the emitted metadata key matches."""
    import inspect

    from mriforge.config.schemas.data import DataConfigSchema
    from mriforge.pipelines import make

    assert DataConfigSchema().dataset_type  # readable, unlike dataset_name
    src = inspect.getsource(make)
    assert "config.data.dataset_name" not in src
    assert '"dataset_type": config.data.dataset_type' in src


class TestMakeBuildsThroughTheCanonicalBuilder:
    """``make`` advertises itself as "SSOT: TrainingSettings" and did not honour it.

    It called ``ModelFactory.create_model(config.model)`` -- passing only the
    model sub-schema, which structurally cannot reach
    ``config.data.processing`` or ``config.undersampling``. So the helper whose
    whole contract is "the config is the single source of truth" built a model
    from a strict subset of it.

    It was also the narrowest checkpoint reader in the repository: it knew only
    ``model_state_dict`` and never stripped wrapper prefixes at all, so a DDP or
    ``torch.compile`` checkpoint loaded nothing and reported success.
    """

    @staticmethod
    def _make_src() -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[3]
            / "src"
            / "mriforge"
            / "pipelines"
            / "make.py"
        ).read_text()

    def test_no_longer_calls_the_config_sniffing_factory(self) -> None:
        import ast

        tree = ast.parse(self._make_src())
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute | ast.Name)
        }
        assert "create_model" not in called
        assert "get_model_factory" not in called

    def test_builds_through_model_builder(self) -> None:
        assert "ModelBuilder(config, device_obj)" in self._make_src()

    def test_binds_the_shared_checkpoint_reader(self) -> None:
        import mriforge.pipelines.make as make_mod

        assert hasattr(make_mod, "resolve_state_dict"), (
            "make.py still knows only `model_state_dict` and never strips "
            "wrapper prefixes; a DDP/compile checkpoint loads nothing"
        )
