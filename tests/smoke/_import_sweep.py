"""Import every named module in a pristine interpreter and record what failed.

Executed as a subprocess by :mod:`tests.smoke.test_all_imports`; the leading
underscore keeps pytest from collecting it. The separate process is the point:
by the time the smoke suite runs, the conftest chain has already imported most
of ``spectramr``, so an in-process ``importlib.import_module`` is a
``sys.modules`` dict hit that passes over a module which cannot actually load.

Usage: ``python _import_sweep.py <modules.json> <failures.json>``
"""

from __future__ import annotations

import importlib
import json
import sys
import tomllib
import warnings
from pathlib import Path

# The entries live under `[tool.pytest.ini_options]`, so pytest owns their
# grammar and is the only correct parser for them. Imported at module scope with
# no fallback: if the symbol moves, the sweep must fail loudly rather than
# degrade to a partial filter set (pitfall #9). `escape=False` is what pytest
# itself applies to ini entries -- `_pytest/warnings.py:53` is this same call.
from _pytest.config import parse_warning_filter

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def apply_session_warning_filters(pyproject: Path) -> None:
    """Mirror pytest's ``filterwarnings`` so the sweep grades warnings identically.

    Read from ``pyproject.toml`` rather than restated here: a second copy of the
    list is the drift this repo keeps paying for. The *grammar* is delegated for
    the same reason -- a hand-rolled parser here understood only the
    ``action::Category:module`` form, so it dropped ``message`` and ``lineno``
    from every entry and raised outright on the message-keyed
    ``action:message:Category`` form.
    """
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    for spec in config["tool"]["pytest"]["ini_options"]["filterwarnings"]:
        warnings.filterwarnings(*parse_warning_filter(spec, escape=False))


def sweep(module_names: list[str]) -> dict[str, dict[str, str | None]]:
    """Import each module once, returning ``{module: failure}`` for the ones that raised."""
    failures: dict[str, dict[str, str | None]] = {}
    for name in module_names:
        try:
            importlib.import_module(name)
        except (Exception, SystemExit) as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                # ModuleNotFoundError.name distinguishes an absent third-party
                # dependency (an environment gap) from a broken import inside
                # the package (a source defect).
                "missing_module": (
                    getattr(exc, "name", None)
                    if isinstance(exc, ModuleNotFoundError)
                    else None
                ),
            }
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <modules.json> <failures.json>", file=sys.stderr)
        return 2
    module_names = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    apply_session_warning_filters(PYPROJECT)
    failures = sweep(module_names)
    Path(argv[2]).write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
