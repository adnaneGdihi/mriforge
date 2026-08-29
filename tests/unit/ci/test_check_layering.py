"""Tests for the layering ratchet (CLAUDE.md non-negotiable #5).

The baseline is keyed on a normalized form of each offending import. If the key were
the raw line, reformatting an import (single line -> parenthesized multi-line, which
`ruff format` does routinely) would read as a brand-new violation and fail the gate
spuriously. These tests pin that the key is reflow-invariant.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_layering.sh"


def _normalize(raw: str) -> str:
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--norm-filter"],
        input=raw,
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def test_key_is_invariant_to_import_line_wrapping() -> None:
    module = "mriforge.infrastructure.training.utils.kspace_masks"
    single = (
        f"src/mriforge/models/diffusion/kspace_process.py:19:"
        f"from {module} import create_kspace_mask_generator"
    )
    wrapped = f"src/mriforge/models/diffusion/kspace_process.py:21:from {module} import ("
    assert _normalize(single) == _normalize(wrapped)


def test_key_is_invariant_to_line_number() -> None:
    early = "src/mriforge/models/x.py:3:from mriforge.pipelines.train import run"
    late = "src/mriforge/models/x.py:900:from mriforge.pipelines.train import run"
    assert _normalize(early) == _normalize(late)


def test_key_still_distinguishes_different_modules() -> None:
    a = "src/mriforge/models/x.py:3:from mriforge.pipelines.train import run"
    b = "src/mriforge/models/x.py:3:from mriforge.pipelines.infer import run"
    assert _normalize(a) != _normalize(b)


def test_key_still_distinguishes_different_files() -> None:
    a = "src/mriforge/models/x.py:3:from mriforge.pipelines.train import run"
    b = "src/mriforge/models/y.py:3:from mriforge.pipelines.train import run"
    assert _normalize(a) != _normalize(b)


def test_key_preserves_plain_import_statements() -> None:
    """`import x` has no symbol list to collapse; it must survive intact."""
    raw = "src/mriforge/models/x.py:3:import mriforge.pipelines.train"
    assert _normalize(raw).endswith("import mriforge.pipelines.train")


def test_key_preserves_non_import_violations() -> None:
    """DataLoader / yaml.safe_load / @register_loss findings are not imports."""
    raw = "src/mriforge/application/x.py:42:    loader = DataLoader(ds)"
    assert "DataLoader(ds)" in _normalize(raw)


def test_guard_passes_on_the_current_tree() -> None:
    """The committed baseline must be in sync: no new violations on a clean tree."""
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--quiet"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
