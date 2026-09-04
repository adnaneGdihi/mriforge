"""Regression: ``spectramr`` main() now has a top-level error boundary.

2026-06-11 CLI enrichment. Previously ``main()`` ended with a bare
``return args.func(args)`` — a Ctrl-C during a long cluster run dumped a raw
``KeyboardInterrupt`` traceback, and any handler exception printed a full
traceback regardless of verbosity. main() now maps Ctrl-C → 130, preserves a
handler's own ``sys.exit`` code, and turns other exceptions into a concise
message + exit 1 (full traceback only with ``-v`` / ``SPECTRAMR_DEBUG``).
"""

from __future__ import annotations

import sys

import pytest

from spectramr.cli import app, diagnostics


def _run_with_doctor_handler(monkeypatch, handler, argv):
    # main() does ``from spectramr.cli.diagnostics import run_doctor`` and binds it
    # to the ``doctor`` subcommand — patch the module attr so our handler is used.
    monkeypatch.setattr(diagnostics, "run_doctor", handler)
    monkeypatch.setattr(sys, "argv", argv)
    return app.main()


def test_keyboard_interrupt_maps_to_130(monkeypatch):
    def boom(_args):
        raise KeyboardInterrupt

    assert _run_with_doctor_handler(monkeypatch, boom, ["spectramr", "doctor"]) == 130


def test_generic_exception_maps_to_1(monkeypatch):
    def boom(_args):
        raise RuntimeError("kaboom")

    assert _run_with_doctor_handler(monkeypatch, boom, ["spectramr", "doctor"]) == 1


def test_handler_systemexit_code_is_preserved(monkeypatch):
    def boom(_args):
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exc:
        _run_with_doctor_handler(monkeypatch, boom, ["spectramr", "doctor"])
    assert exc.value.code == 2


def test_clean_handler_return_passes_through(monkeypatch):
    assert _run_with_doctor_handler(monkeypatch, lambda _a: 0, ["spectramr", "doctor"]) == 0
