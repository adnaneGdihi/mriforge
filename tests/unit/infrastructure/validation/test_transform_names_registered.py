"""Tier-1 check: ``data.processing.transforms`` names must be registered.

Mirrors ``check_metric_names_are_registered``. Before it, an unregistered name
validated at load and was silently discarded by the builder, so the arm trained
without the mechanism it was named for and reported success (pitfall #16).
"""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _config(transforms):
    return SimpleNamespace(
        data=SimpleNamespace(processing=SimpleNamespace(transforms=transforms))
    )


def test_empty_list_skips():
    r = ConfigHealthChecker().check_transform_names_are_registered(_config([]))
    assert r.passed is True
    assert r.severity == "info"


def test_registered_names_pass():
    r = ConfigHealthChecker().check_transform_names_are_registered(
        _config([{"name": "phase_residual"}, {"name": "scout_acquisition"}])
    )
    assert r.passed is True, r.message


def test_unregistered_name_is_an_error():
    r = ConfigHealthChecker().check_transform_names_are_registered(
        _config([{"name": "no_such_transform"}])
    )
    assert r.passed is False
    assert r.severity == "error"
    assert "no_such_transform" in r.message


def test_message_lists_the_registered_names():
    """An error the user cannot act on is barely better than silence."""
    r = ConfigHealthChecker().check_transform_names_are_registered(
        _config([{"name": "nope"}])
    )
    assert "phase_residual" in r.message


def test_entry_without_a_name_is_an_error():
    r = ConfigHealthChecker().check_transform_names_are_registered(
        _config([{"type": "scout_acquisition"}])
    )
    assert r.passed is False
    assert "no 'name'" in r.message


def test_the_check_is_registered_in_the_report():
    """A check nobody calls is the exact failure mode this audit keeps finding."""
    import inspect

    src = inspect.getsource(ConfigHealthChecker)
    assert "self.check_transform_names_are_registered(config)" in src
