"""Every script under ``examples/`` runs to completion, as a cold subprocess.

``examples/`` is public-facing: it is the first code a new user executes, and it
ships in the public export. Until this file existed, **nothing in the suite
referenced ``examples/`` at all**, and ``quickstart_reconstruction.py`` had been
exiting non-zero -- it read ``MODEL_REGISTRY`` without calling
``populate_model_registry()``, so it looked the registry up while it still held
0 entries and blamed the user's install for the miss.

Why a *subprocess* rather than importing ``main()``: the defect above is
invisible in-process. The suite imports model packages for its own reasons, so
by the time an in-process test ran, the registry would already be populated by
somebody else and the missing call would pass unnoticed. That is the
registered-vs-reachable distinction, and only a cold interpreter separates them.

The scripts are discovered rather than listed, so a new example is covered on
arrival instead of when someone remembers to add a case.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# parents[2] == the repository root: tests/smoke/<this file>. Kept relative so
# the test still resolves inside the public export, where it is part of the
# fresh-clone verification.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _example_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("quickstart_*.py"))


def test_the_examples_directory_is_present_and_non_empty() -> None:
    """Guards the discovery above: an empty glob would make every case vacuous."""
    assert EXAMPLES_DIR.is_dir(), f"{EXAMPLES_DIR} is missing"
    assert _example_scripts(), f"no quickstart_*.py found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("script", _example_scripts(), ids=lambda p: p.stem)
def test_example_runs_to_completion(script: Path) -> None:
    env = {
        **os.environ,
        # The clinical-use warning fires on first import and is not a failure.
        "MRIFORGE_SUPPRESS_CLINICAL_WARNING": "1",
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    detail = f"\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.returncode == 0, f"{script.name} exited {proc.returncode}{detail}"
    # Each script prints OK as its last act; a zero exit with no OK would mean
    # the script returned early without doing the work it advertises.
    assert "OK" in proc.stdout, f"{script.name} exited 0 but never printed OK{detail}"
