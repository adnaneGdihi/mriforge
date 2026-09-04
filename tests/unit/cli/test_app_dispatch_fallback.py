"""Regression test for the audit-15-F5 fix.

``src/cli/app.py:main`` historically did ``return args.func(args)``
with no explicit guard. If any subcommand parser forgot
``set_defaults(func=...)``, the call site crashed with an opaque
``AttributeError`` instead of the parser's help text. The fix prints
help and returns exit code 1 — see audit 15 F5.
"""

from __future__ import annotations

from pathlib import Path


def test_main_has_args_func_fallback() -> None:
    src = (
        Path(__file__).resolve().parents[3]
        / "src" / "spectramr" / "cli" / "app.py"
    ).read_text()
    assert 'hasattr(args, "func")' in src or "hasattr(args, 'func')" in src, (
        "src/spectramr/cli/app.py:main no longer guards `args.func`. A subcommand "
        "parser that omits set_defaults(func=...) would crash with "
        "AttributeError — see audit 15 F5."
    )
    assert "parser.print_help()" in src
    assert "return 1" in src
