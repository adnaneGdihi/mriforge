"""Triton's cache must not land in a cluster home directory.

``main.py`` moves the library caches off ``$HOME`` at import: ``TMPDIR``,
``TORCH_HOME``, ``TORCH_METRICS_CACHE``, ``XDG_CACHE_HOME``,
``CUDA_CACHE_CONFIG``. Triton was the hole in that set, and it is the one that
bites hardest: it does **not** read ``XDG_CACHE_HOME``, it defaults to
``~/.triton``, and DeepSpeed populates it while ``import deepspeed`` is still
executing (``deepspeed.ops.transformer.inference.triton.matmul_ext`` constructs
``Fp16Matmul()`` at module scope, which builds the autotune cache).

Observed on a cluster login node, 2026-08-15::

    OSError: [Errno 28] No space left on device: '/home/<user>/.triton/autotune'

raised from inside the import, so it precedes every piece of our own
diagnostics -- no config is read, no device is selected, no health check runs.
A home quota, which has nothing to do with the run, presents as an unhandled
traceback out of a third-party import.

Asserted BEHAVIOURALLY, in a subprocess with a controlled environment, rather
than by grepping ``main.py``: the block is import-time, single-shot, and
order-sensitive, and a text assertion would keep passing if the line were moved
below something that imports deepspeed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = "import os, mriforge.main;print('TRITON=' + os.environ.get('TRITON_CACHE_DIR', '<unset>'))"


def _import_main(env_extra: dict[str, str]) -> str:
    """Import ``mriforge.main`` in a clean process and report TRITON_CACHE_DIR."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"TRITON_CACHE_DIR", "MRIFORGE_CACHE_ROOT", "XDG_CACHE_HOME"}
    }
    env["PYTHONPATH"] = str(REPO / "src")
    env["MRIFORGE_SUPPRESS_CLINICAL_WARNING"] = "1"
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.skip(f"mriforge.main is not importable here: {proc.stderr[-400:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("TRITON="):
            return line[len("TRITON=") :]
    raise AssertionError(f"probe printed no TRITON= line: {proc.stdout[-400:]!r}")


@pytest.mark.slow
def test_importing_main_sets_the_triton_cache_dir(tmp_path: Path) -> None:
    """The variable must be SET, not merely absent-and-hoped-for.

    Triton reads it at its own import; leaving it unset is what produced the
    ``~/.triton`` write.
    """
    resolved = _import_main({"MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot")})
    assert resolved != "<unset>", "TRITON_CACHE_DIR unset; Triton falls back to ~/.triton"
    assert str(tmp_path) in resolved, (
        f"TRITON_CACHE_DIR={resolved!r} does not follow MRIFORGE_CACHE_ROOT"
    )


@pytest.mark.slow
def test_the_triton_cache_is_not_under_home(tmp_path: Path) -> None:
    """The actual invariant, stated as the failure it prevents.

    Anti-vacuity for the test above: ``~/.triton`` is precisely the value the
    unfixed code produced, so a run that lands anywhere under ``$HOME`` has not
    been moved.
    """
    resolved = _import_main({"MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot")})
    home = str(Path.home())
    assert not resolved.startswith(home), (
        f"TRITON_CACHE_DIR={resolved!r} is inside $HOME ({home}). DeepSpeed "
        "writes its autotune cache there during `import deepspeed`, so a home "
        "quota becomes an unhandled OSError before any of our code runs."
    )


@pytest.mark.slow
def test_an_explicit_setting_is_not_clobbered(tmp_path: Path) -> None:
    """``setdefault``, unlike the sibling assignments in that block.

    DeepSpeed's own startup warning tells operators to set this variable to a
    non-NFS path, so an explicit export is an informed decision about NFS
    behaviour that this process must not overwrite.
    """
    chosen = str(tmp_path / "operator-chose-this")
    resolved = _import_main(
        {
            "MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot"),
            "TRITON_CACHE_DIR": chosen,
        }
    )
    assert resolved == chosen, (
        f"an explicitly exported TRITON_CACHE_DIR was overwritten with {resolved!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The same three questions, asked of the OTHER entry point.
#
# Every probe above imports ``mriforge.main``. The multi-GPU launch documented in
# ``scripts/training/train_distributed.sbatch:173`` is
# ``torchrun -m mriforge.cli train-distributed``, which never imports that module
# -- so all three passed for a year while the path that actually runs DeepSpeed
# had TRITON_CACHE_DIR unset. Measured on the pre-fix tree::
#
#     import mriforge.cli   ->  TRITON_CACHE_DIR=None  XDG_CACHE_HOME=None
#     import mriforge.main  ->  both set under cache_root
#
# The caches are configured at runtime on that path (``initialize_accelerator``),
# not at import, so the probe below calls it. XDG_CACHE_HOME is checked alongside
# Triton because torch resolves its JIT extension build root through it, and
# DeepSpeed writes compiled ops there during the same import.
# ─────────────────────────────────────────────────────────────────────────────

_DISTRIBUTED_PROBE = """
import os
os.environ["FORCE_CPU"] = "true"
import mriforge.cli  # the package `torchrun -m mriforge.cli` executes
from mriforge.accelerator import initialize_accelerator
initialize_accelerator(device_str="cpu", seed=42, pipeline="train")
for key in ("TRITON_CACHE_DIR", "XDG_CACHE_HOME"):
    print(key + "=" + os.environ.get(key, "<unset>"))
"""


def _via_cli_entry_point(env_extra: dict[str, str]) -> dict[str, str]:
    """Run the distributed entry point's setup and report the cache vars."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {"TRITON_CACHE_DIR", "MRIFORGE_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR"}
    }
    env["PYTHONPATH"] = str(REPO / "src")
    env["MRIFORGE_SUPPRESS_CLINICAL_WARNING"] = "1"
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", _DISTRIBUTED_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.skip(f"the CLI entry point is not runnable here: {proc.stderr[-400:]}")
    found = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    missing = {"TRITON_CACHE_DIR", "XDG_CACHE_HOME"} - found.keys()
    if missing:
        raise AssertionError(f"probe printed no {missing} line: {proc.stdout[-400:]!r}")
    return found


@pytest.mark.slow
def test_the_distributed_entry_point_sets_the_caches_too(tmp_path: Path) -> None:
    """The regression that shipped: main.py was fixed, the torchrun path was not."""
    resolved = _via_cli_entry_point(
        {"MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot")}
    )
    for key, value in resolved.items():
        assert value != "<unset>", (
            f"{key} is unset on the `torchrun -m mriforge.cli train-distributed` "
            "path. DeepSpeed writes both of these during its own import, so they "
            "fall back into $HOME and the import fails on a cluster."
        )
        assert str(tmp_path) in value, (
            f"{key}={value!r} does not follow MRIFORGE_CACHE_ROOT"
        )


@pytest.mark.slow
def test_the_distributed_entry_point_caches_are_not_under_home(tmp_path: Path) -> None:
    """Anti-vacuity, stated as the failure it prevents (cf. the main.py twin)."""
    resolved = _via_cli_entry_point(
        {"MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot")}
    )
    home = str(Path.home())
    for key, value in resolved.items():
        assert not value.startswith(home), (
            f"{key}={value!r} is inside $HOME ({home}) on the distributed path."
        )


@pytest.mark.slow
def test_the_distributed_entry_point_honours_an_explicit_triton_setting(
    tmp_path: Path,
) -> None:
    """The setdefault carve-out must not be lost when the block moved."""
    chosen = str(tmp_path / "operator-chose-this")
    resolved = _via_cli_entry_point(
        {
            "MRIFORGE_CACHE_ROOT": str(tmp_path / "cacheroot"),
            "TRITON_CACHE_DIR": chosen,
        }
    )
    assert resolved["TRITON_CACHE_DIR"] == chosen
