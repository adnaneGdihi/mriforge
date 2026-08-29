"""Regression: ``build_container`` tolerates an omitted ``checkpoint:`` block.

The root ``checkpoint:`` field is optional. ``bootstrap.build_container`` used
to raise ``config.checkpoint is required but not provided`` whenever it was
``None`` — which silently blocked the entire 28-arm ``mrixfields2026`` cohort
(every config omits the block; the audit passes 93/93 because it never builds
the container, then bootstrap kills the run). The resolver now coalesces a
missing/``None`` block to a default ``CheckpointConfigSchema()`` instead. See
``TODO/audit/run_debug_kspace_vf_mrixfields_20260617.md`` (2026-06-20 follow-up).

The checkpoint resolution is exercised through the small pure helper
``_resolve_checkpoint_config`` rather than the full ``build_container`` (which
would require a complete DI container + model registry), mirroring the
``_validate_data_availability_at_startup`` helper-level tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.bootstrap import _resolve_checkpoint_config
from mriforge.config.schemas.checkpoint import CheckpointConfigSchema


def test_resolve_checkpoint_config_defaults_when_none() -> None:
    """A ``None`` checkpoint block resolves to a default schema, not an error."""
    resolved = _resolve_checkpoint_config(SimpleNamespace(checkpoint=None))
    assert isinstance(resolved, CheckpointConfigSchema)


def test_resolve_checkpoint_config_passes_through_explicit() -> None:
    """An explicit checkpoint block is returned unchanged (no clobbering)."""
    explicit = CheckpointConfigSchema(checkpoint_dir="/tmp/explicit_ckpt")
    resolved = _resolve_checkpoint_config(SimpleNamespace(checkpoint=explicit))
    assert resolved is explicit
