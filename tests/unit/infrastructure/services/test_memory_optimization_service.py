"""``MemoryOptimizationService`` must not write cache-directory variables.

``setup_memory_optimization`` used to ``makedirs("/tmp/triton_cache")`` and point
``TRITON_CACHE_DIR`` at it. Two independent defects:

* the path carried no ``$USER`` term. ``/tmp`` is sticky and shared, so on a
  cluster node that directory belongs to whoever created it first and every
  later user gets a permission error out of a directory they cannot list.
  ``infrastructure/config/env_resolver`` documents exactly this hazard on its
  ``_EXAMPLE_ROOT`` constant, which is why its own example spells ``$USER``.
* it made this service a *second* writer to a variable
  ``configure_cache_environment()`` already owns. Both used ``setdefault``
  semantics, so the winner was whichever ran first -- and this one runs only
  once a training run reaches the service, i.e. import-order-dependent in a way
  no caller controls.

These tests pin the absence, not the old value: a regression here is someone
re-adding a writer, so the assertion has to be "this service sets no cache
directory at all".
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from mriforge.infrastructure.services.memory_optimization_service import (
    MemoryOptimizationService,
)

#: Every variable that names a *directory* and is owned by ``env_resolver``.
_CACHE_DIR_VARS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_HOME",
    "HF_HOME",
    "TMPDIR",
)


def _run_setup_with_cuda(environ: dict[str, str]) -> None:
    """Run ``setup_memory_optimization`` down the CUDA branch, on any host.

    The env-var block is nested inside ``if torch.cuda.is_available():``, so on a
    CPU test runner it is simply skipped and a naive test passes vacuously.
    Patch the probe so the branch actually executes.
    """
    with (
        mock.patch.dict(os.environ, environ, clear=True),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.set_per_process_memory_fraction"),
    ):
        MemoryOptimizationService().setup_memory_optimization()
        # Copy out before the patch.dict unwinds.
        environ.clear()
        environ.update(os.environ)


def test_setup_writes_no_cache_directory_variable() -> None:
    """No directory-valued cache variable is set by this service."""
    seen: dict[str, str] = {}
    _run_setup_with_cuda(seen)

    written = [var for var in _CACHE_DIR_VARS if var in seen]
    assert written == [], (
        f"MemoryOptimizationService set cache directory variable(s) {written}. "
        "env_resolver.configure_cache_environment() is the single owner of "
        "these; a second writer makes the effective value depend on import "
        "order. Steer the cache root via MRIFORGE_CACHE_ROOT in .env instead."
    )


def test_setup_does_not_create_a_shared_tmp_directory() -> None:
    """The service creates no directory under ``/tmp``.

    Skips rather than passes when ``/tmp/triton_cache`` already exists. The
    old code created it with ``exist_ok=True``, so on any machine that has run
    this service before, a "was it created?" check finds nothing new and
    reports success -- it would pass just as happily against the unfixed
    source. That is a silent success, so make the gap visible instead: the
    directory's mere presence is evidence the defect ran here at least once.
    """
    triton = Path("/tmp/triton_cache")
    if triton.exists():
        pytest.skip(
            f"{triton} already exists (owner uid "
            f"{triton.stat().st_uid}), so a newly-created-directory check "
            "cannot distinguish fixed from unfixed code on this host. Remove "
            "it to run this test. Its presence is itself the hazard: /tmp is "
            "sticky, so the next user to run the old code gets EACCES here."
        )

    before = set(Path("/tmp").iterdir())
    _run_setup_with_cuda({})
    created = set(Path("/tmp").iterdir()) - before

    assert not any("triton" in entry.name for entry in created), (
        f"MemoryOptimizationService created {created} under /tmp. /tmp is "
        "sticky and shared: a path with no $USER term belongs to whoever "
        "created it first, and every later user gets EACCES."
    )


def test_setup_still_sets_the_allocator_knobs() -> None:
    """Removing the cache writer must not have removed the rest of the block.

    Guards the over-correction: the fix deleted three lines from the middle of a
    dict literal, and deleting one line too many here would silently drop the
    allocator configuration this service exists to apply.
    """
    seen: dict[str, str] = {}
    _run_setup_with_cuda(seen)

    assert seen.get("PYTORCH_CUDA_ALLOC_CONF") == ("expandable_segments:True,max_split_size_mb:512")
    assert seen.get("TRITON_CACHE_SIZE") == "10737418240"
    assert seen.get("TORCHINDUCTOR_CACHE_SIZE") == "10737418240"


def test_setup_never_overwrites_an_existing_value() -> None:
    """The remaining writes keep ``setdefault`` semantics.

    ``main.py`` sets ``PYTORCH_CUDA_ALLOC_CONF`` above ``import torch`` -- where
    PyTorch actually reads it. If this service overwrote that, the later value
    would be inert *and* misreported in provenance.
    """
    seen = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    _run_setup_with_cuda(seen)

    assert seen["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
