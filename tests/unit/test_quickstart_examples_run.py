"""Every shipped example runs, in a COLD interpreter, and says so.

``scripts/release/public_allowlist.txt`` ships ``examples/`` with the annotation
*"E4: each of these must audit --probe clean before release"*. Nothing enforced
it: no test in this suite executed an example, and
``examples/quickstart_reconstruction.py`` exited 1 in both the private tree and
the export, with an error blaming the reader's install::

    Model 'toeplitz_attention_unet' not in registry.
    Did you `pip install -e .` from the repo root?

The install was fine. ``MODEL_REGISTRY`` is **empty on a bare**
``import mriforge`` -- 0 models before ``populate_model_registry()`` and 586
after -- so the example was reading an unpopulated registry and reporting it as
the reader's fault.

Why a subprocess, and not an import
-----------------------------------
This is the registered-vs-reachable distinction the reachability contract turns
on. An in-process test proves nothing here, because the suite has almost
certainly imported something that populated the registry already; the defect
exists *only* in an interpreter that has imported nothing else. Six recorded
incidents in ``models/init_registry.py`` are this same gap.

The examples are the first thing a new reader runs. An example that exits 1 is
the release over-claiming in the most visible place there is.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Walk up for the directory holding both ``pyproject.toml`` and ``examples``.

    Not ``parents[N]`` (a hardcoded index is the most-repeated defect in this
    suite's history) and not ``tests.utils.corpus.repo_root`` (which needs git,
    while ``examples/`` ships in an sdist that has none).
    """
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file() and (directory / "examples").is_dir():
            return directory
    raise RuntimeError(f"no repo root above {Path(__file__).resolve()}")


_ROOT = _repo_root()
_EXAMPLES = sorted((_ROOT / "examples").glob("quickstart_*.py"))


def test_the_examples_were_actually_found() -> None:
    """Anti-vacuity: an empty glob would make every parametrised case a skip.

    That is the silent shape -- a green run having executed nothing. The README
    names three quickstarts, so fewer than three means the discovery is wrong,
    not that the examples are.
    """
    assert len(_EXAMPLES) >= 3, (
        f"expected >=3 quickstart examples under {_ROOT / 'examples'}, found {[p.name for p in _EXAMPLES]}"
    )


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: p.stem)
def test_example_runs_clean_in_a_cold_interpreter(example: Path) -> None:
    env = dict(os.environ)
    env["MRIFORGE_SUPPRESS_CLINICAL_WARNING"] = "1"
    src = _ROOT / "src"
    if src.is_dir():
        # A worktree runs against its own src, not whatever is pip-installed.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )

    out = subprocess.run(
        [sys.executable, str(example)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    assert out.returncode == 0, (
        f"{example.name} exited {out.returncode}. The README promises it "
        f"finishes in under 30 seconds.\n--- stdout ---\n{out.stdout}"
        f"\n--- stderr ---\n{out.stderr}"
    )
    # Each example prints OK as its last act. A script that exits 0 having
    # printed a diagnostic and done nothing would otherwise pass.
    assert "OK" in out.stdout, f"{example.name} exited 0 without reaching OK:\n{out.stdout}"
