"""Unit tests for TrainingEnvironmentDirector.

Focus: the WS-D calling-convention migration. ``__init__`` was moved onto the
canonical :class:`BuilderContext` shape behind the ``accepts_builder_context``
back-compat shim, so both the legacy ``Director(config)`` call and the canonical
``Director(BuilderContext(config=config))`` call must construct an equivalent
object (same stored ``._config``).
"""

from unittest.mock import MagicMock, patch

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.training.builders.director import (
    TrainingEnvironmentDirector,
)


def _make_config() -> MagicMock:
    """Minimal stub TrainingSettings (device wiring is patched out)."""
    config = MagicMock(spec=TrainingSettings)
    config.device = "cpu"
    return config


class TestDirectorBuilderContextMigration:
    """Both construction shapes are accepted and produce equivalent state."""

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_legacy_config_shape(self, _mock_resolve) -> None:
        """Legacy ``Director(config)`` still works via the compat shim."""
        config = _make_config()

        director = TrainingEnvironmentDirector(config)  # type: ignore[arg-type]

        assert director._config is config
        assert director._device == torch.device("cpu")

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_builder_context_shape(self, _mock_resolve) -> None:
        """Canonical ``Director(BuilderContext(config=config))`` works."""
        config = _make_config()

        director = TrainingEnvironmentDirector(BuilderContext(config=config))

        assert director._config is config
        assert director._device == torch.device("cpu")

    @patch.object(
        TrainingEnvironmentDirector,
        "_resolve_device",
        return_value=torch.device("cpu"),
    )
    def test_both_shapes_equivalent(self, _mock_resolve) -> None:
        """Legacy and context construction store the same config + device."""
        config = _make_config()

        legacy = TrainingEnvironmentDirector(config)  # type: ignore[arg-type]
        canonical = TrainingEnvironmentDirector(BuilderContext(config=config))

        assert legacy._config is canonical._config is config
        assert legacy._device == canonical._device

    def test_init_marked_as_accepting_builder_context(self) -> None:
        """The shim records its marker so the fitness check sees the migration."""
        assert (
            getattr(
                TrainingEnvironmentDirector.__init__,
                "__accepts_builder_context__",
                False,
            )
            is True
        )
