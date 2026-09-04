"""Regression test for the audit-15-F13 fix.

``spectramr/main.py`` and ``spectramr/accelerator.py`` both set
``PYTORCH_CUDA_ALLOC_CONF``. main.py sets it at module-import-time
(before ``import torch``); accelerator.py uses ``os.environ.setdefault``
as a fallback for callers that didn't pre-set. The two strings must
stay in lock-step — otherwise an entry point that bypasses main.py
gets a less-aggressive allocator config than training expects.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _extract_assignment(source: str, key: str) -> str:
    """Find the RHS of `os.environ[\"<key>\"] = ...` or
    `os.environ.setdefault(\"<key>\", ...)` in the given source.
    Returns the literal string (without quotes)."""
    import re

    # Match either form.
    pattern = (
        rf"os\.environ\[\"{key}\"\]\s*=\s*\"([^\"]+)\""
        rf"|os\.environ\.setdefault\(\s*\"{key}\"\s*,\s*\"([^\"]+)\""
    )
    matches = re.findall(pattern, source)
    if not matches:
        raise AssertionError(f"No assignment for {key} found")
    # Return the first non-empty group from the first match.
    for groups in matches:
        for g in groups:
            if g:
                return g
    raise AssertionError(f"No assignment value for {key}")


def _package_source(repo: Path, module: str) -> str:
    """Read a module out of the package, failing with the reason if it moved.

    Both files were read from ``src/`` directly until the 2026-05
    ``src -> src/spectramr`` refactor, after which this test died on a raw
    ``FileNotFoundError`` out of ``pathlib`` — a message that names a missing
    path and not the fact that a guard had stopped guarding. It was red on every
    run from then until job 8000966 surfaced it. The two literals had NOT
    drifted in the meantime; the invariant held unattended.
    """
    path = repo / "src" / "spectramr" / module
    if not path.is_file():
        raise AssertionError(
            f"{path} does not exist — spectramr.{module.removesuffix('.py')} "
            "moved, and this sync guard cannot see the file it grades. "
            "Repoint it rather than letting it fail as a missing file."
        )
    return path.read_text()


def test_pytorch_cuda_alloc_conf_in_sync() -> None:
    repo = Path(__file__).resolve().parents[2]
    main_src = _package_source(repo, "main.py")
    accel_src = _package_source(repo, "accelerator.py")

    main_val = _extract_assignment(main_src, "PYTORCH_CUDA_ALLOC_CONF")
    accel_val = _extract_assignment(accel_src, "PYTORCH_CUDA_ALLOC_CONF")

    assert main_val == accel_val, (
        f"PYTORCH_CUDA_ALLOC_CONF drifted between main.py ({main_val!r}) "
        f"and accelerator.py ({accel_val!r}). An entry point that bypasses "
        "main.py would get a less-aggressive allocator config than training "
        "expects — see audit 15 F13."
    )


# ─────────────────────────────────────────────────────────────────────────────
# The same invariant, for the cache block.
#
# The guard above was written for ONE variable, but its stated rationale --
# "an entry point that bypasses main.py gets [less] than training expects" --
# always covered the whole env block. The gap was paid for on 2026-08-16:
# main.py set six cache variables inline and accelerator.py repeated three of
# them by hand, so `torchrun -m spectramr.cli train-distributed` (which never
# imports main.py) ran with TRITON_CACHE_DIR and XDG_CACHE_HOME unset. Both are
# written by `import deepspeed`, so DeepSpeed's caches went to the cluster $HOME
# and the import died with a bare PermissionError before any config was read.
#
# Rather than string-compare six literals, the invariant is now structural: the
# layout lives in ONE function, and each entry point must call it and must not
# set any member of the block by hand. Drift is then not expressible.
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_VARS = (
    "TMPDIR",
    "TORCH_HOME",
    "TORCH_METRICS_CACHE",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_CONFIG",
    "TRITON_CACHE_DIR",
)


@pytest.mark.parametrize("module", ["main.py", "accelerator.py"])
def test_both_entry_points_configure_caches_through_the_shared_helper(
    module: str,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    src = _package_source(repo, module)

    assert "configure_cache_environment()" in src, (
        f"spectramr/{module} does not call configure_cache_environment(). If this "
        "is the path a launcher uses, it will run with the library caches "
        "defaulting under $HOME — which is how `import deepspeed` came to fail "
        "on a cluster home directory."
    )


@pytest.mark.parametrize("module", ["main.py", "accelerator.py"])
@pytest.mark.parametrize("var", _CACHE_VARS)
def test_no_entry_point_sets_a_cache_variable_by_hand(module: str, var: str) -> None:
    """Anti-drift: a hand-set variable is one the *other* entry point will lack.

    This is the assertion that fails on the pre-fix tree — accelerator.py set
    TMPDIR / TORCH_HOME / CUDA_CACHE_CONFIG itself and therefore silently
    defined a second, smaller cache layout.
    """
    repo = Path(__file__).resolve().parents[2]
    src = _package_source(repo, module)

    for form in (f'os.environ["{var}"]', f'os.environ.setdefault("{var}"'):
        assert form not in src, (
            f"spectramr/{module} sets {var} directly ({form}). The cache layout is "
            "owned by infrastructure.config.env_resolver."
            "configure_cache_environment — setting it here re-creates the split "
            "that left the distributed entry point without it."
        )
