"""Package-mode entry point for ``python -m mriforge.cli``.

When ``src/cli`` is a package, ``python -m mriforge.cli`` looks for this
``__main__`` submodule rather than running ``__init__.py`` directly.
Without it, the canonical CLAUDE.md commands

    python -m mriforge.cli train --config experiments/active/<exp>.yaml
    python -m mriforge.cli list-features --module all --format markdown

fail with ``No module named mriforge.cli.__main__``. We delegate to
:func:`mriforge.cli.app.main` to preserve the historical behaviour of the
old ``src/cli.py`` module.
"""

from __future__ import annotations

import sys

from mriforge.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
