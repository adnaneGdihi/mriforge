"""Package-mode entry point for ``python -m spectramr.cli``.

When ``src/cli`` is a package, ``python -m spectramr.cli`` looks for this
``__main__`` submodule rather than running ``__init__.py`` directly.
Without it, the canonical CLAUDE.md commands

    python -m spectramr.cli train --config experiments/active/<exp>.yaml
    python -m spectramr.cli list-features --module all --format markdown

fail with ``No module named spectramr.cli.__main__``. We delegate to
:func:`spectramr.cli.app.main` to preserve the historical behaviour of the
old ``src/cli.py`` module.
"""

from __future__ import annotations

import sys

from spectramr.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
