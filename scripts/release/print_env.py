"""Print every framework env var and its current value.

Used by ``make env-show``. Reads the canonical inventory from
``spectramr.core.env`` so this script can never drift from the source of
truth — adding a new var to ``env.py`` automatically surfaces it here.
"""

from __future__ import annotations

import os
import sys

from spectramr.core import env

PALETTE = {"set": "\033[32m", "unset": "\033[90m", "reset": "\033[0m"}
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _fmt(value: str | None) -> str:
    if value is None:
        return f"{PALETTE['unset']}<unset>{PALETTE['reset']}" if USE_COLOR else "<unset>"
    return f"{PALETTE['set']}{value!r}{PALETTE['reset']}" if USE_COLOR else repr(value)


def main() -> None:
    print("Framework environment variables (current values):")
    print("-" * 70)
    width = max(len(n) for n in env.names()) + 2
    for name in env.names():
        value = os.environ.get(name)
        print(f"  {name:<{width}} = {_fmt(value)}")
    print("-" * 70)
    print(
        "Set these via .env (cp .env.example .env) or your shell profile.\n"
        "See docs/CLUSTER_DATA_LAYOUT.md for which variables do what."
    )


if __name__ == "__main__":
    main()
