"""Shared :class:`rich.console.Console` for the CLI (round 11, 2026-05-17).

Rich auto-detects TTY vs non-TTY and strips ANSI codes when stdout is
piped — this means:

- ``python -m spectramr.cli audit foo.yaml`` in an interactive terminal renders
  green ✅ / red ❌ / yellow ⚠ with cyan check-name tags.
- ``python -m spectramr.cli audit foo.yaml > out.txt`` writes plain text
  identical to the pre-round-11 ``print(...)`` output. CI / grep /
  downstream parsers are unaffected.
- ``python -m spectramr.cli audit foo.yaml --json`` is unaffected — the JSON
  paths use ``json.dumps`` directly, not the console.

``soft_wrap=True`` prevents rich from inserting hard line breaks to
fit the terminal width — long messages stay on one line so grep keeps
working even in a narrow window.

``highlight=False`` prevents rich's default auto-highlighting of
numbers / paths / quoted strings, which would otherwise inject styles
into substrings that downstream tools may inspect.

Audit anchor: TODO/audit/smoke_audit_20260516.md §F-COLOR.
"""

from __future__ import annotations

from rich.console import Console

# Public module-level console. Importers use this directly:
#
#     from spectramr.cli._rendering import console
#     console.print(some_health_check_result)
#
# Rich's auto-detect handles TTY vs non-TTY without any caller code.
console: Console = Console(soft_wrap=True, highlight=True)

__all__ = ["console"]
