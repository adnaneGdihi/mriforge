"""Regression: LoggingService must not leak process-global logging state.

Each ``LoggingService`` instance monkeypatched ``logging.getLogger`` and added a
``_GlobalLevelFilter`` to the root logger, and nothing ever undid either. A
second instantiation (tests, HPO workers, nested pipelines) therefore:

* captured the FIRST instance's patch as its "original", chaining wrappers so
  every ``getLogger`` call got progressively slower and could never be restored;
* stacked another root filter, over-aggressively filtering and leaking memory.

The patch now always delegates to the pristine stdlib ``getLogger``, stale
``_GlobalLevelFilter``s are dropped before a new one is added, and ``close()``
restores both.
"""

from __future__ import annotations

import logging as std_logging

from spectramr.infrastructure.services.logging_service import (
    _TRUE_GETLOGGER,
    LoggingService,
    _GlobalLevelFilter,
)


def _root_level_filters() -> list:
    return [
        f
        for f in std_logging.getLogger().filters
        if isinstance(f, _GlobalLevelFilter)
    ]


def test_repeated_instantiation_does_not_stack_root_filters():
    a = LoggingService()
    b = LoggingService()
    c = LoggingService()
    try:
        # Exactly one global filter regardless of how many services exist (each
        # __init__ drops stale ones before adding its own).
        assert len(_root_level_filters()) == 1
    finally:
        for svc in (c, b, a):
            svc.close()


def test_getlogger_patch_does_not_chain():
    a = LoggingService()
    b = LoggingService()
    try:
        # Both instances delegate to the SAME pristine stdlib function, so no
        # wrapper chains (which would slow every getLogger call).
        assert a._original_getLogger is _TRUE_GETLOGGER
        assert b._original_getLogger is _TRUE_GETLOGGER
    finally:
        b.close()
        a.close()


def test_close_restores_pristine_getlogger():
    svc = LoggingService()
    # Compare against the STORED bound-method ref (a fresh ``svc._patched_getLogger``
    # access would not be ``is``-identical).
    assert std_logging.getLogger is svc._patched_getLogger_ref  # patched while alive
    svc.close()
    assert std_logging.getLogger is _TRUE_GETLOGGER  # restored on close
    assert not _root_level_filters()  # and its filter removed
