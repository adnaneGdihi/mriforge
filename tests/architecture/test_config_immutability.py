"""
TASK III.2 – Config immutability fitness function.

Enforces CLAUDE.md pitfall #4:

    Mutating a loaded ``TrainingSettings`` is forbidden — it's frozen
    (``model_config = SettingsConfigDict(frozen=True, ...)``) and raises
    ``ValidationError`` on any field assignment after construction.

Tests:

1. ``test_training_settings_schema_is_frozen`` — asserts the class-level
   ``model_config`` has ``frozen=True`` without instantiating anything
   heavy.  Fast: pure Python, no YAML loaded.

2. ``test_training_settings_mutation_raises`` — constructs a minimal
   ``TrainingSettings`` from a dict (same dict pattern as existing
   e2e tests) and asserts that mutating any top-level field raises
   ``pydantic.ValidationError`` or ``TypeError`` (both are expected from
   frozen Pydantic v2 models).

3. ``test_nested_schema_classes_are_frozen`` — iterates over the
   config schema sub-models (``ModelConfigSchema``, ``DataConfigSchema``,
   etc.) and confirms each carries ``frozen=True`` in its ``model_config``.
   Pure class inspection, no instantiation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.architecture


# ---------------------------------------------------------------------------
# 1. Schema-class frozen flag (no instantiation)
# ---------------------------------------------------------------------------


def test_training_settings_schema_is_frozen() -> None:
    """TrainingSettings.model_config must declare frozen=True."""
    from spectramr.config.settings import TrainingSettings

    cfg = TrainingSettings.model_config
    assert cfg.get("frozen") is True, (
        "TrainingSettings.model_config does not have frozen=True. "
        "CLAUDE.md pitfall #4 requires the config to be immutable after load."
    )


# ---------------------------------------------------------------------------
# 2. Runtime mutation raises
# ---------------------------------------------------------------------------


def test_training_settings_mutation_raises() -> None:
    """Assigning to a field of a constructed TrainingSettings must raise."""
    import pydantic

    from spectramr.config.settings import TrainingSettings

    # NOTE: no ``config_version`` here. The loader validates it and then DELETES
    # it (``TrainingSettings.from_yaml``), so the model itself forbids the key
    # under extra="forbid" — passing it is a construction error, not a version
    # declaration.
    minimal: dict = {
        "model": {
            "model_type": "standard_unet",
            "in_channels": 1,
            "out_channels": 1,
        },
        "training": {
            "training_mode": "reconstruction",
        },
        "data": {
            "batch_size": 2,
        },
        "optimization": {},
        "logging": {},
    }
    # Construction must succeed. This used to pytest.skip on ValidationError,
    # which meant schema drift silently disabled the frozen-config assertion
    # below — non-negotiable #1 went untested from whenever the dict rotted
    # until someone read the skip reason (#629). Fail loudly instead: a broken
    # fixture is a defect to fix, not a reason to stop checking.
    try:
        settings = TrainingSettings(**minimal)
    except pydantic.ValidationError as exc:
        pytest.fail(
            "The minimal TrainingSettings fixture no longer validates — update it "
            f"to match the current schema rather than skipping this test:\n{exc}"
        )

    # Mutation must raise
    with pytest.raises((pydantic.ValidationError, TypeError)):
        settings.seed = 999  # type: ignore[misc]

    with pytest.raises((pydantic.ValidationError, TypeError)):
        settings.device = "cpu"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Nested sub-schema classes are also frozen
# ---------------------------------------------------------------------------


_SUB_SCHEMA_IMPORTS: list[tuple[str, str]] = [
    ("spectramr.config.schemas.model", "ModelConfigSchema"),
    ("spectramr.config.schemas.data", "DataConfigSchema"),
    ("spectramr.config.schemas.optimization", "OptimizationConfigSchema"),
    ("spectramr.config.schemas.training.base", "TrainingStrategyConfigSchema"),
    ("spectramr.config.schemas.early_stopping", "EarlyStoppingConfigSchema"),
    ("spectramr.config.schemas.ema", "EMAConfigSchema"),
    ("spectramr.config.schemas.logging", "LoggingConfigSchema"),
    ("spectramr.config.schemas.acceleration", "AccelerationConfigSchema"),
]


@pytest.mark.parametrize("module_path,class_name", _SUB_SCHEMA_IMPORTS)
def test_sub_schema_class_is_frozen(module_path: str, class_name: str) -> None:
    """Each named sub-schema class must carry frozen=True in model_config."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    cfg = getattr(cls, "model_config", {})
    assert cfg.get("frozen") is True, (
        f"{module_path}.{class_name}.model_config does not have frozen=True. "
        "All config sub-schemas must be immutable (CLAUDE.md pitfall #4)."
    )
