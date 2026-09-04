"""Console-verbosity policy for multi-rank launches.

A torchrun launch runs N *independent* interpreters, each with its own root
logger, so every ``logger.info`` before the process group exists lands in the
job log N times. On a 4-GPU node that turned the first screen of a run into
four interleaved copies of the same four lines::

    ⏳ spectramr train-distributed: importing PyTorch + model registry ...   x4
    Cache root from TMPDIR: /tmp/.../spectramr_cache                          x4
    [Parallel] process-group backend=nccl (source=config)                  x4
    UserWarning: spectraMR is research software and is NOT FOR CLINICAL USE   x4

``setup_distributed`` already quietened non-zero ranks, but it runs *after* the
config loads and the backend resolves -- by which point all four lines are
already out. The fix is to apply the same policy at the entry point, using the
rank torchrun exports into the environment before the process starts.

**WARNINGs deliberately survive.** The level is raised to ``WARNING``, not
``ERROR``: a per-rank warning (DataLoader worker oversubscription, a Triton
cache on NFS) is a fact about *that* rank and printing it once per rank is
correct. Only the informational narration is de-duplicated.
"""

from __future__ import annotations

import logging

from spectramr.core import env

__all__ = ["quiet_secondary_ranks", "rank_floor"]


def rank_floor(level: int, floor: int = logging.WARNING) -> int:
    """Return ``level``, but never below ``floor`` on a secondary rank.

    Companion to :func:`quiet_secondary_ranks` for the code that *re-sets* the
    level rather than reading it. ``LoggingService.setup`` resolves a level from
    the config and pushes it onto the root logger, every existing logger and
    every handler -- which silently undoes an earlier clamp the moment training
    configures logging. Clamping the resolved level once, at the point it is
    resolved, keeps that whole fan-out consistent instead of re-clamping after
    each place that writes it.

    Rank 0 and single-process runs get ``level`` back unchanged, so a
    non-distributed run is provably untouched.
    """
    if not env.is_secondary_rank():
        return level
    return max(level, floor)


def quiet_secondary_ranks(level: int = logging.WARNING) -> bool:
    """Raise the root logger level on every rank but 0. Idempotent.

    Returns ``True`` if the policy was applied (i.e. this is a non-zero rank of
    a distributed launch), ``False`` otherwise -- returned rather than logged so
    the gating is unit-testable, and so a single-process run is provably
    untouched.
    """
    if not env.is_secondary_rank():
        return False
    # Unconditional, matching what ``setup_distributed`` has always done on a
    # non-zero rank. To read a secondary rank's INFO/DEBUG stream, launch that
    # rank alone (``torchrun --nproc_per_node=1``) rather than adding a knob
    # whose only effect is to restore the N-way duplication.
    logging.getLogger().setLevel(level)
    return True
