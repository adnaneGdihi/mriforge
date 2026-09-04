"""Logging banners + phase timing (PR — backlog_beautify_logging_and_reporting Phase 1.3).

Provides two utilities:

- ``banner(title)`` — dim-bordered centered banner for pipeline phase headers.
- ``phase(title)`` — context manager that emits a banner on enter, an
  elapsed-time line on exit.

Both replace ad-hoc ``logger.info("=" * 60)`` patterns.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

DEFAULT_WIDTH = 60

_logger = logging.getLogger(__name__)


def banner(
    title: str,
    *,
    level: int = logging.INFO,
    width: int = DEFAULT_WIDTH,
    char: str = "=",
    logger: logging.Logger | None = None,
) -> None:
    """Emit a centered, dim-bordered banner."""
    log = logger or _logger
    border = char * width
    centered = title.center(width)
    log.log(level, border)
    log.log(level, centered)
    log.log(level, border)


@contextmanager
def phase(
    title: str,
    *,
    level: int = logging.INFO,
    width: int = DEFAULT_WIDTH,
    logger: logging.Logger | None = None,
) -> Iterator[None]:
    """Banner on enter, elapsed-time on exit.

    >>> with phase("Loading dataset"):
    ...     ...  # work
    [
       =================================
              Loading dataset
       =================================
       ...work...
       Loading dataset finished in 1.42s
    ]
    """
    log = logger or _logger
    banner(title, level=level, width=width, logger=log)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.log(level, "%s finished in %.2fs", title, elapsed)


__all__ = ["banner", "phase"]
