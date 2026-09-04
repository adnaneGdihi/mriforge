"""Logging-service de-duplication (2026-06-19 audit).

Pins the fixes that consolidated redundant root-logger configuration in
:mod:`spectramr.infrastructure.services.logging_service`:

* #1/#2 — the "upgrade-or-install a colored console handler" routine, once
  copy-pasted in ``bootstrap_console_logging`` and ``LoggingService.setup``,
  now lives in a single ``_install_or_upgrade_colored_console`` helper.
* #3 — the file-handler format string is a single ``_FILE_LOG_FORMAT`` constant
  (was duplicated verbatim across the try/except branches of ``setup``).
* #4 — ``ComprehensiveLoggingService.log`` no longer re-throttles on the base
  class's ``_throttle_counts`` (which double-counted info/debug AND wrongly
  throttled warnings/errors); the base ``LoggingService.log`` is the single
  throttle authority.

The helper tests run on throwaway *named* loggers — no ``LoggingService`` is
constructed, so the global ``getLogger`` monkey-patch is never installed. The
``#4`` checks are AST-based because constructing ``ComprehensiveLoggingService``
pulls TensorBoard / file-handler side effects that don't belong in a unit test.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from spectramr.infrastructure.services.logging_service import (
    _FILE_LOG_FORMAT,
    ColoredConsoleFormatter,
    _install_or_upgrade_colored_console,
)

_SRC = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "services"
    / "logging_service.py"
)


def _fresh_logger(name: str) -> logging.Logger:
    """Return a uniquely-named logger with no handlers (no cross-test bleed)."""
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.setLevel(logging.NOTSET)
    return lg


def _stream_handlers(lg: logging.Logger) -> list[logging.Handler]:
    return [
        h
        for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]


# --------------------------------------------------------------------------- #
#  #1 / #2 — shared colored-console helper
# --------------------------------------------------------------------------- #


def test_helper_installs_fresh_colored_handler_when_none() -> None:
    lg = _fresh_logger("test_dedup.install_fresh")
    changed = _install_or_upgrade_colored_console(lg, logging.INFO)
    sh = _stream_handlers(lg)
    assert changed is True
    assert len(sh) == 1
    assert isinstance(sh[0].formatter, ColoredConsoleFormatter)
    assert sh[0].level == logging.INFO


def test_helper_upgrades_plain_handler_in_place() -> None:
    """The load-bearing contract: upgrade in place, preserve handler identity."""
    lg = _fresh_logger("test_dedup.upgrade_in_place")
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(plain)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO)

    sh = _stream_handlers(lg)
    assert changed is True
    assert len(sh) == 1, "must upgrade in place, not stack a second handler"
    assert sh[0] is plain, "handler identity must be preserved"
    assert isinstance(plain.formatter, ColoredConsoleFormatter)


def test_helper_is_noop_when_already_colored() -> None:
    lg = _fresh_logger("test_dedup.already_colored")
    h = logging.StreamHandler()
    h.setFormatter(ColoredConsoleFormatter())
    lg.addHandler(h)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO)

    assert changed is False, "already-colored handler needs no change"
    assert len(_stream_handlers(lg)) == 1, "must not add a second handler"


def test_helper_install_if_missing_false_adds_nothing() -> None:
    """Silent-mode gate: setup passes install_if_missing=log_to_console."""
    lg = _fresh_logger("test_dedup.no_install")
    changed = _install_or_upgrade_colored_console(lg, logging.INFO, install_if_missing=False)
    assert changed is False
    assert _stream_handlers(lg) == [], "silent mode must not add a console sink"


def test_helper_install_if_missing_false_still_upgrades_existing() -> None:
    """install_if_missing=False suppresses fresh installs but still upgrades."""
    lg = _fresh_logger("test_dedup.upgrade_no_install")
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(plain)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO, install_if_missing=False)
    assert changed is True
    assert isinstance(plain.formatter, ColoredConsoleFormatter)
    assert len(_stream_handlers(lg)) == 1


# --------------------------------------------------------------------------- #
#  #3 — single file-format constant
# --------------------------------------------------------------------------- #


def test_file_log_format_literal_defined_once() -> None:
    src = _SRC.read_text()
    literal = '"%(asctime)s - %(name)s - %(levelname)s - %(message)s"'
    assert _FILE_LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert src.count(literal) == 1, (
        "the file-handler format string must appear once (in _FILE_LOG_FORMAT); "
        "the setup() try/except branches must reference the constant (#3)."
    )
    assert src.count("_FILE_LOG_FORMAT") >= 3, (
        "_FILE_LOG_FORMAT must be defined and used in both file-handler branches."
    )


# --------------------------------------------------------------------------- #
#  #4 — Comprehensive.log must not re-throttle (single authority = base class)
# --------------------------------------------------------------------------- #


def _comprehensive_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(_SRC.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "ComprehensiveLoggingService"
    )
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_comprehensive_log_does_not_touch_throttle_counts() -> None:
    fn = _comprehensive_method("log")
    touches = [
        n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == "_throttle_counts"
    ]
    assert not touches, (
        "ComprehensiveLoggingService.log must NOT re-throttle on _throttle_counts "
        "— that double-counted info/debug and throttled warnings/errors (#4). "
        "The base LoggingService.log is the single throttle authority."
    )


def test_comprehensive_init_does_not_reinit_throttle_state() -> None:
    fn = _comprehensive_method("__init__")
    reinits = [
        t
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and t.attr in ("_throttle_counts", "_throttle_limit")
    ]
    assert not reinits, (
        "ComprehensiveLoggingService.__init__ must not re-initialise throttle "
        "state — LoggingService.__init__ already owns it (#4)."
    )
