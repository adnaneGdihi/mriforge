"""Regression tests for the sim2rank ``--seed`` cross-script contract.

The 2026-05-21 fix added ``--seed`` to ``sim2rank.py`` after a SLURM
launch failed with::

    sim2rank.py: error: unrecognized arguments: --seed 0

Root cause: ``scripts/sim2rank/run_full_pipeline.sbatch`` broadcasts
``--seed ${SIM2RANK_SEED}`` to three sibling scripts as if they shared
a uniform CLI surface, but ``sim2rank.py`` had only ``--meta-seed``.
The orchestrator's third invocation died on argparse before any work
ran. This file pins the contract so a future contributor who deletes
``--seed`` from any of the broadcast targets fails CI instead of the
cluster job.

Why pin this rather than rely on smoke runs:

- The orchestrator runs on the cluster, not in CI; a divergence here
  silently waits to detonate at submission time.
- The contract is structural — three CLIs must accept a flag the
  orchestrator broadcasts. A unit test makes the structure explicit.
- ``--meta-seed`` must remain a separate override (some experiments
  fix the metric (theta, seed) contract independently of the
  pipeline-wide seed). This test pins that BOTH flags exist on
  ``sim2rank.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO / relpath).read_text()


# Every script the orchestrator broadcasts ``--seed`` to. If
# ``run_full_pipeline.sbatch`` ever adds a fourth target, extend this
# tuple AND the orchestrator-side regex below.
_SEED_BROADCAST_TARGETS = (
    "scripts/sim2rank/sim2rank.py",
    "scripts/sim2rank/run_all_generations.py",
    "scripts/sim2rank/degradation_snapshots.py",
    # Not in the orchestrator today, but shares the same family pattern
    # (was --meta-seed-only before 2026-05-21). Including it locks the
    # uniform surface so future orchestrator wiring is safe.
    "scripts/sim2rank/run_meta_eval_all_datasets.py",
)


@pytest.mark.parametrize("relpath", _SEED_BROADCAST_TARGETS)
def test_cli_accepts_seed_flag(relpath: str) -> None:
    """Every sim2rank CLI in the broadcast family must accept ``--seed``."""
    text = _read(relpath)
    pattern = re.compile(
        r"""add_argument\(\s*["']--seed["'][^)]*type\s*=\s*int""",
        re.DOTALL,
    )
    assert pattern.search(text), (
        f"{relpath} must declare ``--seed`` (type=int). The orchestrator "
        "``scripts/sim2rank/run_full_pipeline.sbatch`` broadcasts "
        "``--seed ${SIM2RANK_SEED}`` to this family; a missing flag aborts "
        "the cluster job before any work runs."
    )


def test_sim2rank_keeps_meta_seed_separately() -> None:
    """``sim2rank.py`` must keep ``--meta-seed`` as a distinct override.

    The (theta, seed) contract in ``run_meta_evaluation_core`` lets the
    operator pin the metric-evaluation seed independently of the
    pipeline-wide ``--seed``. Collapsing the two would break that
    contract.
    """
    text = _read("scripts/sim2rank/sim2rank.py")
    assert re.search(
        r"""add_argument\(\s*["']--meta-seed["']""",
        text,
    ), "sim2rank.py must keep ``--meta-seed`` as a distinct override of ``--seed``."


def test_sim2rank_meta_seed_defaults_to_seed_when_unset() -> None:
    """If ``--meta-seed`` is not passed, it must fall back to ``--seed``.

    A common mistake is forgetting to thread the meta-seed through a
    new call site. The fallback in ``__main__`` (``args.meta_seed =
    args.seed`` when None) guarantees a single ``--seed`` from the
    orchestrator pins every RNG path in one shot.
    """
    text = _read("scripts/sim2rank/sim2rank.py")
    # Tolerant to whitespace and comments between the two lines.
    assert re.search(
        r"if\s+args\.meta_seed\s+is\s+None\s*:\s*\n\s*args\.meta_seed\s*=\s*args\.seed",
        text,
    ), (
        "sim2rank.py must default args.meta_seed to args.seed when "
        "--meta-seed is unset. See test docstring for rationale."
    )


def test_run_meta_eval_all_datasets_meta_seed_falls_back_to_seed() -> None:
    """Same fallback contract for the cross-dataset runner."""
    text = _read("scripts/sim2rank/run_meta_eval_all_datasets.py")
    assert re.search(
        r"if\s+args\.meta_seed\s+is\s+None\s*:\s*\n\s*args\.meta_seed\s*=\s*args\.seed",
        text,
    ), (
        "run_meta_eval_all_datasets.py must default args.meta_seed to "
        "args.seed when --meta-seed is unset."
    )


def test_orchestrator_broadcasts_seed_to_every_target() -> None:
    """The orchestrator sbatch must keep passing ``--seed`` to every
    script in ``_SEED_BROADCAST_TARGETS`` that it invokes.

    The orchestrator only invokes three of the four targets today
    (``run_meta_eval_all_datasets.py`` is a sibling, not a step). This
    test pins the three actual broadcast call sites in
    ``run_full_pipeline.sbatch`` to catch the inverse mistake: someone
    removes ``--seed`` from the sbatch (so the script's
    non-deterministic default takes over silently).
    """
    text = _read("scripts/sim2rank/run_full_pipeline.sbatch")
    # Active --seed broadcasts: there should be exactly three (one per
    # python invocation). Comment lines (#  ... SIM2RANK_SEED=0) are
    # excluded — we only count the bash-active ``--seed "${SEED}"`` form.
    active = [
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ]
    seed_lines = [ln for ln in active if "--seed " in ln]
    assert len(seed_lines) >= 3, (
        f"run_full_pipeline.sbatch must broadcast ``--seed`` to all "
        f"three pipeline steps (run_all_generations / sim2rank / "
        f"degradation_snapshots). Found {len(seed_lines)} active "
        f"broadcast(s): {seed_lines!r}"
    )
