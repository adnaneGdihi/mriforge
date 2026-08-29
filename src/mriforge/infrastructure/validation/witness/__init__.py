"""Witness package: registry, subject, gate, and the registered checks.

Discovery is a ``pkgutil`` walk started **at each leaf subpackage's own**
``__path__``, not one level up.

An in-repo ``ImportError`` is **re-raised**. A witness module that fails to import
is a detector that does not exist, and swallowing that would leave a clean report
for a class nothing was watching. Only a genuinely missing third-party dependency
is downgraded to a warning.

That re-raise needs ``onerror`` to cover a sub-package, and this docstring used to
say otherwise -- that a walk "started above a subpackage skips it (``is_pkg``
continue)". Measured 2026-08-28: false. ``walk_packages`` recurses INTO each
sub-package by importing it, so ``is_pkg`` skips only that sub-package's own
``__init__`` and every module beneath it is still yielded. The import pkgutil does
to recurse is the one the loop below cannot see, and pkgutil's default silently
discards its failure along with the entire sub-tree.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

from mriforge.infrastructure.validation.witness.gate import (
    WitnessGateError,
    assert_no_errors,
    run_witnesses,
    scheduled_witnesses,
)
from mriforge.infrastructure.validation.witness.registry import (
    Applicability,
    Severity,
    Stage,
    Subject,
    Tier,
    Witness,
    WitnessRegistry,
    WitnessVerdict,
    get_witness_registry,
    register_witness,
)
from mriforge.infrastructure.validation.witness.subject import (
    WitnessSubject,
    WitnessSubjectUnavailableError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Applicability",
    "Severity",
    "Stage",
    "Subject",
    "Tier",
    "Witness",
    "WitnessGateError",
    "WitnessRegistry",
    "WitnessSubject",
    "WitnessSubjectUnavailableError",
    "WitnessVerdict",
    "assert_no_errors",
    "get_witness_registry",
    "register_witness",
    "run_witnesses",
    "scheduled_witnesses",
]


def _classify_witness_import_error(module_name: str, exc: BaseException) -> None:
    """Re-raise an in-repo failure; downgrade a missing third-party dep.

    Shared by the per-module handler and the ``onerror`` hook so a sub-package
    and a module get the same verdict from one owner.
    """
    if not isinstance(exc, ImportError):
        raise exc
    root = getattr(exc, "name", "") or ""
    if not root or root.startswith("mriforge"):
        raise exc  # a missing detector must never be silent
    logger.warning(
        "witness module %s skipped: optional dependency %s is absent",
        module_name,
        root,
    )


def _discover(package) -> None:
    def _on_package_error(name: str) -> None:
        _classify_witness_import_error(
            name, sys.exc_info()[1] or RuntimeError("unknown walk error")
        )

    for _finder, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + ".", onerror=_on_package_error
    ):
        if is_pkg:
            continue
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            _classify_witness_import_error(module_name, exc)


def _discover_all() -> None:
    from mriforge.infrastructure.validation.witness import checks as _checks

    _discover(_checks)


_discover_all()
