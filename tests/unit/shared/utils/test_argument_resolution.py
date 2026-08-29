"""Tests for argument-resolution helpers.

Targets ``mriforge.shared.utils.argument_resolution``. Provides the canonical
"CLI > config > default" precedence pattern used everywhere CLI flags
override YAML.

Categories:

- ``resolve_argument`` — ordered fallback chain
- ``resolve_argument_pair`` — CLI primary + fallback + config primary + fallback
- ``resolve_boolean_argument`` — boolean-aware fallback
- ``resolve_with_getattr`` — object-attribute lookup with config fallback
- ``ArgumentResolver`` — context manager wrapper, ``required=True`` enforcement
"""

from __future__ import annotations

import argparse

import pytest

from mriforge.shared.utils.argument_resolution import (
    ArgumentResolver,
    resolve_argument,
    resolve_argument_pair,
    resolve_boolean_argument,
    resolve_with_getattr,
)


# ---------------------------------------------------------------------------
# resolve_argument
# ---------------------------------------------------------------------------


def test_resolve_argument_returns_first_non_none() -> None:
    """First non-None source is returned."""
    assert resolve_argument(None, "value", "fallback") == "value"


def test_resolve_argument_skips_empty_string() -> None:
    """Empty string is treated as 'no value' and skipped."""
    assert resolve_argument("", "value") == "value"


def test_resolve_argument_returns_default_when_all_empty() -> None:
    """When every source is None / empty, the default is returned."""
    assert resolve_argument(None, None, default="fallback") == "fallback"


def test_resolve_argument_returns_none_with_no_default() -> None:
    """Without a default, returns None when all sources are empty."""
    assert resolve_argument(None, "") is None


# ---------------------------------------------------------------------------
# resolve_argument_pair
# ---------------------------------------------------------------------------


def test_resolve_argument_pair_cli_wins() -> None:
    """CLI value takes precedence over both config keys."""
    out = resolve_argument_pair(
        cli_attr="cli_value",
        cli_fallback_attr=None,
        config_key="primary",
        config_dict={"primary": "config_value"},
    )
    assert out == "cli_value"


def test_resolve_argument_pair_cli_fallback_used() -> None:
    """When primary CLI is None, fallback CLI is used."""
    out = resolve_argument_pair(
        cli_attr=None,
        cli_fallback_attr="cli_fallback",
        config_key="primary",
        config_dict={"primary": "config_value"},
    )
    assert out == "cli_fallback"


def test_resolve_argument_pair_config_used() -> None:
    """When CLI sources are absent, primary config key is used."""
    out = resolve_argument_pair(
        cli_attr=None,
        cli_fallback_attr=None,
        config_key="primary",
        config_dict={"primary": "config_value"},
    )
    assert out == "config_value"


def test_resolve_argument_pair_config_fallback_used() -> None:
    """When primary config is missing, fallback config is used."""
    out = resolve_argument_pair(
        cli_attr=None,
        cli_fallback_attr=None,
        config_key="primary",
        config_dict={"fallback": "fb_value"},
        config_fallback_key="fallback",
    )
    assert out == "fb_value"


def test_resolve_argument_pair_default_returned_when_all_empty() -> None:
    """All sources empty → default is returned."""
    out = resolve_argument_pair(
        cli_attr=None,
        cli_fallback_attr=None,
        config_key="primary",
        config_dict={},
        default="DEFAULT",
    )
    assert out == "DEFAULT"


# ---------------------------------------------------------------------------
# resolve_boolean_argument
# ---------------------------------------------------------------------------


def test_resolve_boolean_cli_true_wins() -> None:
    """CLI True takes precedence."""
    assert resolve_boolean_argument(True, "use_ema", {"use_ema": False}, default=False)


def test_resolve_boolean_cli_false_treated_as_unset() -> None:
    """CLI False is treated as 'unset' (so config can override)."""
    out = resolve_boolean_argument(False, "use_ema", {"use_ema": True}, default=False)
    assert out is True


def test_resolve_boolean_default_when_neither() -> None:
    """No CLI, no config → default."""
    out = resolve_boolean_argument(None, "use_ema", {}, default=False)
    assert out is False


# ---------------------------------------------------------------------------
# resolve_with_getattr
# ---------------------------------------------------------------------------


def test_resolve_with_getattr_picks_first_attribute() -> None:
    """First non-empty attribute is returned."""

    class Args:
        input_dir = "/cli/path"
        input = None

    out = resolve_with_getattr(Args(), "input_dir", "input")
    assert out == "/cli/path"


def test_resolve_with_getattr_falls_back_to_second_attribute() -> None:
    """First attribute None → second attribute is used."""

    class Args:
        input_dir = None
        input = "/fallback/path"

    out = resolve_with_getattr(Args(), "input_dir", "input")
    assert out == "/fallback/path"


def test_resolve_with_getattr_uses_config_when_attrs_missing() -> None:
    """All attrs empty → config dict consulted."""

    class Args:
        input_dir = None

    out = resolve_with_getattr(
        Args(), "input_dir",
        config_dict={"input": "/config/path"},
        config_keys=["input"],
    )
    assert out == "/config/path"


# ---------------------------------------------------------------------------
# ArgumentResolver
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Helper: build a Namespace with the given attributes."""
    return argparse.Namespace(**kwargs)


def test_argument_resolver_resolve_path_cli_wins() -> None:
    """``resolve_path`` returns the CLI value when present."""
    args = _make_args(input_dir="/cli/path")
    resolver = ArgumentResolver(args, config_dict={"input_dir": "/config/path"})
    assert resolver.resolve_path("input_dir") == "/cli/path"


def test_argument_resolver_resolve_path_config_fallback() -> None:
    """``resolve_path`` falls back to config when CLI is empty."""
    args = _make_args(input_dir=None)
    resolver = ArgumentResolver(args, config_dict={"input_dir": "/config/path"})
    assert resolver.resolve_path("input_dir", config_key="input_dir") == "/config/path"


def test_argument_resolver_required_path_missing_raises() -> None:
    """``required=True`` with no value → ``ValueError``."""
    args = _make_args(input_dir=None)
    resolver = ArgumentResolver(args, config_dict={})
    with pytest.raises(ValueError, match="Required path argument"):
        resolver.resolve_path("input_dir", required=True)


def test_argument_resolver_resolve_value_default_used() -> None:
    """``resolve_value`` returns default when no source is provided."""
    args = _make_args()
    resolver = ArgumentResolver(args, config_dict={})
    assert resolver.resolve_value("ghost", default="DEFAULT") == "DEFAULT"


def test_argument_resolver_resolve_device_default_auto() -> None:
    """``resolve_device`` defaults to 'auto'."""
    args = _make_args()
    resolver = ArgumentResolver(args, config_dict={})
    assert resolver.resolve_device() == "auto"


def test_argument_resolver_resolve_device_picks_cli() -> None:
    """``resolve_device`` picks the CLI ``device`` when present."""
    args = _make_args(device="cuda:0")
    resolver = ArgumentResolver(args, config_dict={"device": "cpu"})
    assert resolver.resolve_device() == "cuda:0"
