"""Tests for ``spectramr.config.io`` — raw YAML I/O for build-time tools.

This helper is the SSOT-compliant escape valve: legitimate build-time
code paths (ablation generation, HPO winner-config materialisation)
that need a mutable raw dict route through here instead of calling
``yaml.safe_load`` directly. The pre-commit guard
``tests/unit/test_no_yaml_reparse.py`` still passes because
``src/config/`` is the allowed subtree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.io import read_yaml_dict, write_yaml_dict


def test_read_yaml_dict_returns_dict(tmp_path: Path) -> None:
    """A simple YAML file round-trips through ``read_yaml_dict`` as a dict."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n")
    out = read_yaml_dict(p)
    assert out == {"a": 1, "b": {"c": 2}}


def test_read_yaml_dict_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    """An empty YAML file yields an empty dict, never ``None``."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert read_yaml_dict(p) == {}


def test_read_yaml_dict_raises_on_missing_file(tmp_path: Path) -> None:
    """Missing file raises ``FileNotFoundError`` (not a silent empty result)."""
    with pytest.raises(FileNotFoundError, match="YAML file not found"):
        read_yaml_dict(tmp_path / "does_not_exist.yaml")


def test_write_yaml_dict_creates_parent_dirs(tmp_path: Path) -> None:
    """Nested output directories are created on demand."""
    target = tmp_path / "deep" / "nested" / "config.yaml"
    write_yaml_dict({"x": 1}, target)
    assert target.exists()


def test_round_trip_preserves_structure(tmp_path: Path) -> None:
    """``read_yaml_dict ∘ write_yaml_dict`` preserves dict structure."""
    data = {"model": {"type": "unet", "depth": 4}, "lr": 1e-4, "seeds": [0, 1, 42]}
    target = tmp_path / "config.yaml"
    write_yaml_dict(data, target)
    assert read_yaml_dict(target) == data
