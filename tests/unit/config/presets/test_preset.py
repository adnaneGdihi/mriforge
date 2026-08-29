"""Tests for ``config/presets/preset.py``.

Created 2026-08-03 — this module had no test partner, which is how its
``version`` default and its version *comparison* both stayed at a legacy
spelling through the 1.0 rename.

``ConfigPreset.load_config`` is the **only** place in the codebase that compares two
schema versions. It reads the raw YAML with ``yaml.safe_load``, so the loader's
legacy fold never runs on it: a preset pointing at a migrated ``1.0`` file while
its own ``version`` still said ``6.0`` would raise. It is currently unreachable
(no ``presets/`` directory exists and no preset YAML ships), which is exactly why
it needs pinning — dead code with a hard-coded version is a trap armed for
whoever adds the first preset.
"""

from __future__ import annotations

import pytest
import yaml

from mriforge.config.schemas.base import (
    CANONICAL_CONFIG_VERSION,
    LEGACY_CONFIG_VERSIONS,
)
from mriforge.config.presets.preset import ConfigPreset


def _preset(tmp_path, declared: str, expected: str | None = None) -> ConfigPreset:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"config_version": declared, "model": {}}))
    kwargs = {} if expected is None else {"version": expected}
    return ConfigPreset(
        name="p",
        category="test",
        description="fixture",
        yaml_path=str(path),
        **kwargs,
    )


def test_the_default_version_is_canonical() -> None:
    """Not a hard-coded legacy string. A preset that defaults to 6.0 demands
    6.0 from its YAML, so it would reject every migrated config."""
    import inspect

    default = inspect.signature(ConfigPreset.__init__).parameters["version"].default
    assert default == CANONICAL_CONFIG_VERSION
    assert default not in LEGACY_CONFIG_VERSIONS


def test_a_canonical_yaml_loads_against_the_default(tmp_path) -> None:
    assert _preset(tmp_path, CANONICAL_CONFIG_VERSION).load_config()


def test_a_legacy_yaml_is_refused_by_a_canonical_preset(tmp_path) -> None:
    """The comparison is on the RAW document, so the loader's fold never runs.

    This is the behaviour to know about, not a bug to paper over: a preset
    pins an exact schema version on purpose. Migrate the preset's YAML and its
    ``version`` together.
    """
    # `LEGACY_CONFIG_VERSIONS` is empty since 2026-08-08, so this is a literal:
    # what is under test is a version the preset does not expect, and a retired
    # one is the sharpest case.
    with pytest.raises(ValueError, match="expected"):
        _preset(tmp_path, "6.0").load_config()


def test_an_explicit_version_still_wins(tmp_path) -> None:
    """The default moved; the parameter must still be honoured, or pinning a
    preset to a specific schema stops working."""
    # A preset may pin ANY version it likes -- the comparison is on the raw
    # document, before the loader's version gate -- so pinning a retired one
    # still works. That is the property under test: the parameter wins.
    assert _preset(tmp_path, "6.0", expected="6.0").load_config()
