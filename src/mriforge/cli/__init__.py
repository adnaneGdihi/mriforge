"""CLI subcommand package — main app + regulatory submodule.

The unified CLI entrypoint historically lived in ``src/cli.py`` (a single
1500+ line module). When ``src/cli/regulatory.py`` was added in commit
141f29b6c, the package directory began *shadowing* the file: Python's
import machinery resolves ``mriforge.cli`` to ``src/cli/__init__.py``
whenever both targets exist, leaving the original file's symbols
(``train``, ``predict``, ``list_features``, …) unreachable.

The fix relocates the original module to :mod:`mriforge.cli.app` and re-
exports every public symbol from here so existing call sites
(``from mriforge.cli import list_features``, ``python -m mriforge.cli train …``)
continue to work without changes.
"""

# Re-export the full app surface so tests / scripts that do
# ``from mriforge.cli import list_features`` continue to resolve correctly.
# The wildcard pulls in every name not prefixed with ``_``; ``main``
# and the subcommand functions all qualify.
from mriforge.cli import regulatory
from mriforge.cli.app import *  # noqa: F403
from mriforge.cli.app import (
    audit,
    benchmark,
    campaign_cancel,
    campaign_evaluate,
    campaign_status,
    campaign_submit,
    campaign_watch,
    export,
    hpo_cmd,
    list_features,
    main,
    meta_evaluate,
    predict,
    report,
    sanity_check,
    train,
)

__all__ = [
    # App entrypoints — kept in roughly the same order as the original
    # argparse subcommand registration in ``app.main()``.
    "main",
    "train",
    "sanity_check",
    "predict",
    "benchmark",
    "export",
    "list_features",
    "audit",
    "campaign_submit",
    "campaign_status",
    "campaign_evaluate",
    "campaign_cancel",
    "campaign_watch",
    "hpo_cmd",
    "report",
    "meta_evaluate",
    # Specialised entrypoint package.
    "regulatory",
]
