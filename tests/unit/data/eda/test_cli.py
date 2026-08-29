"""Smoke test for the dataset-EDA CLI entry point."""
from __future__ import annotations

import subprocess
import sys


def test_cli_runs_and_emits_card(present_kspace_manifest, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable, "scripts/eda/dataset_eda_runner.py",
            "--manifests-root", str(present_kspace_manifest.parent),
            "--data-root", str(tmp_path),
            "--databases-dir", str(tmp_path / "nonexistent_db"),
            "--output-dir", str(out),
            "--samples-per-dataset", "2",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "tiny_kspace" / "00_card.png").exists()
    assert (out / "_overview" / "04_coverage_matrix.png").exists()
