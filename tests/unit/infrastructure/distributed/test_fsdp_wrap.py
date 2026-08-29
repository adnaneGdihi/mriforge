"""Tests for the FSDP wrap helper.

Distributed init needs a multi-process world, so these exercise the
single-process case -- which is all the test runner has, and where the
behaviour used to be wrong.

The old docstring here read: "The single-process behaviour is critical: anyone
who forgets to launch with torchrun must still get a working model." That
reasoning is what produced the bug. A "working model" in that situation is an
UNSHARDED one, so the run completes, reports success, and stamps ``fsdp`` into
its own provenance while never having sharded anything -- and DDP raised on the
identical mistake, so the two strategies disagreed about whether it was fatal.
Forgetting ``torchrun`` is now an error for both.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from pydantic import ValidationError

from mriforge.config.schemas.base import FSDPConfigSchema, ParallelismConfigSchema
from mriforge.infrastructure.distributed.fsdp_wrap import maybe_wrap_with_fsdp


class _Bag:
    """Tiny attribute bag standing in for ParallelismConfigSchema."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_fsdp_disabled_returns_input_unchanged() -> None:
    parallel = ParallelismConfigSchema(fsdp=FSDPConfigSchema(enabled=False))
    model = nn.Linear(4, 4)
    wrapped = maybe_wrap_with_fsdp(model, parallel)
    assert wrapped is model


def test_fsdp_enabled_without_distributed_init_raises() -> None:
    """Enabling FSDP with no process group must RAISE.

    This test previously asserted the opposite -- that it returns the bare model
    with a logged warning, "never raise". That was the bug, not the contract: a
    user who forgot ``torchrun`` got a full-length UNSHARDED run that completed,
    reported success, and stamped ``fsdp`` into its own provenance. DDP has
    always raised on the identical mistake, so the two strategies disagreed about
    whether the same misconfiguration was fatal.

    Note the config now needs BOTH halves -- ``strategy='fsdp'`` and
    ``fsdp.enabled=True`` -- because they were independent switches and could
    contradict each other (#620 F2).
    """
    parallel = ParallelismConfigSchema(
        strategy="fsdp", fsdp=FSDPConfigSchema(enabled=True)
    )
    model = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="not initialised"):
        maybe_wrap_with_fsdp(model, parallel)


def test_fsdp_requires_both_halves_of_the_declaration() -> None:
    """``fsdp.enabled`` alone used to silently shard while the config read
    'no parallelism'."""
    with pytest.raises(ValidationError):
        ParallelismConfigSchema(fsdp=FSDPConfigSchema(enabled=True))


def test_fsdp_no_parallel_block_passthrough() -> None:
    """Calling with a config that doesn't have an .fsdp attribute is a no-op."""
    parallel = _Bag()  # no .fsdp at all
    model = nn.Linear(4, 4)
    wrapped = maybe_wrap_with_fsdp(model, parallel)
    assert wrapped is model
