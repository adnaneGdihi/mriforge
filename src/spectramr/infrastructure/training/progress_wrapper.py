"""Progress bar wrapper for training loops."""

import logging
import sys
from collections.abc import Iterable
from typing import Union

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Graceful fallback

logger = logging.getLogger(__name__)


def progress(
    iterable: Iterable,
    logger_instance: logging.Logger | None = None,
    use_tqdm: bool | None = None,
    desc: str | None = None,
    total: int | None = None,
    **tqdm_kwargs,
) -> Union[Iterable, "tqdm"]:
    """Wrapper to conditionally use tqdm progress bars.

    Shows progress bars when:
    - use_tqdm=True explicitly, OR
    - logging level is WARNING or above AND environment is interactive

    Args:
        iterable: Iterable to wrap
        logger_instance: Logger instance to check level (default: root logger)
        use_tqdm: Explicitly enable/disable tqdm (overrides auto-detection)
        desc: Description for progress bar
        total: Total items (for non-len iterables)
        **tqdm_kwargs: Additional arguments to pass to tqdm

    Returns:
        tqdm wrapper if progress bars enabled, else original iterable
    """
    if tqdm is None:
        logger.debug("tqdm not installed, returning raw iterable")
        return iterable

    # Get logger to check level
    if logger_instance is None:
        logger_instance = logging.getLogger()

    # Determine if we should use tqdm
    should_use_tqdm = False

    if use_tqdm is True:
        should_use_tqdm = True
    elif use_tqdm is False:
        should_use_tqdm = False
    else:
        # Auto-detect: use tqdm if logging level is WARNING or above (less verbose)
        # and we're in an interactive environment
        is_warning_or_above = logger_instance.level >= logging.WARNING
        is_interactive = sys.stdout.isatty()
        should_use_tqdm = is_warning_or_above and is_interactive

    if not should_use_tqdm:
        logger.debug("tqdm disabled (auto-detection or explicit)")
        return iterable

    # Set up tqdm kwargs
    tqdm_config = {
        "dynamic_ncols": True,
        "disable": not sys.stdout.isatty(),  # Disable for non-interactive
    }
    if desc:
        tqdm_config["desc"] = desc
    if total:
        tqdm_config["total"] = total
    tqdm_config.update(tqdm_kwargs)

    logger.debug(f"Using tqdm progress bar: {desc or 'training'}")
    return tqdm(iterable, **tqdm_config)


def create_epoch_progress(
    num_epochs: int,
    logger_instance: logging.Logger | None = None,
    use_tqdm: bool | None = None,
) -> Union[range, "tqdm"]:
    """Create a progress-wrapped range for epoch loops.

    Args:
        num_epochs: Number of epochs
        logger_instance: Logger instance to check level
        use_tqdm: Explicitly enable/disable tqdm

    Returns:
        Wrapped or raw range
    """
    return progress(
        range(num_epochs),
        logger_instance=logger_instance,
        use_tqdm=use_tqdm,
        desc="Epochs",
        total=num_epochs,
    )


def create_step_progress(
    num_steps: int,
    logger_instance: logging.Logger | None = None,
    use_tqdm: bool | None = None,
) -> Union[range, "tqdm"]:
    """Create a progress-wrapped range for training steps.

    Args:
        num_steps: Number of steps
        logger_instance: Logger instance to check level
        use_tqdm: Explicitly enable/disable tqdm

    Returns:
        Wrapped or raw range
    """
    return progress(
        range(num_steps),
        logger_instance=logger_instance,
        use_tqdm=use_tqdm,
        desc="Training",
        total=num_steps,
    )


__all__ = ["create_epoch_progress", "create_step_progress", "progress"]
