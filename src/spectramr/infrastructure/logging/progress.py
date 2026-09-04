"""tqdm ↔ logger integration (Phase 1.2).

Avoids ANSI / progress-bar interleaving in SLURM logs by detecting
non-TTY and falling back to a periodic ``logger.info(...)`` call.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


def smart_progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "progress",
    log_every_n: int = 100,
    log_every_seconds: float = 30.0,
) -> Iterator[Any]:
    """tqdm when stdout is a TTY, periodic logger.info otherwise.

    Args:
        iterable: any iterable to wrap.
        total: optional total length for ETA computation.
        desc: bar / log-line prefix.
        log_every_n: under non-TTY, emit a log line every N steps.
        log_every_seconds: AND every X seconds since the last line.
    """
    if sys.stdout.isatty():
        try:
            from tqdm import tqdm  # type: ignore
            from tqdm.contrib.logging import logging_redirect_tqdm

            with logging_redirect_tqdm():
                yield from tqdm(iterable, total=total, desc=desc)
            return
        except ImportError:  # pragma: no cover
            pass

    # Non-TTY fallback — periodic info lines.
    t0 = time.perf_counter()
    last_log = t0
    for i, item in enumerate(iterable):
        yield item
        now = time.perf_counter()
        if (i + 1) % log_every_n == 0 or now - last_log > log_every_seconds:
            elapsed = now - t0
            if total:
                eta = elapsed * (total - (i + 1)) / max(1, i + 1)
                logger.info(
                    "%s %d/%d (%.1fs elapsed, ~%.1fs eta)", desc, i + 1, total, elapsed, eta
                )
            else:
                logger.info("%s %d (%.1fs elapsed)", desc, i + 1, elapsed)
            last_log = now


__all__ = ["smart_progress"]
