"""Unit tests for the SLURM-array per-task dispatcher (WS-4 PR-B).

These pin the logic that used to be a fragile ``grep | sed | … || true`` block
in ``dispatch_experiments.sbatch`` — now a torch-free YAML parse. The bash
DRYRUN tests (``test_dispatch_experiments.py``) cover the shell wiring; these
cover the resolution + cap math directly, which is faster and immune to shell
quoting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import spectramr.cli.manifest_dispatch as md
from spectramr.cli.manifest_dispatch import (
    compute_iter_cap_overrides,
    resolve_manifest_config,
)


def _manifest(tmp_path: Path) -> Path:
    m = tmp_path / "manifest.txt"
    m.write_text(
        "experiments/inprogress/vf/exp_vf_04.yaml\n"
        "experiments/inprogress/vf/exp_vf_33.yaml\n"
    )
    return m


# --- manifest resolution -----------------------------------------------------


def test_resolve_by_index(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    assert resolve_manifest_config(m, 0).endswith("exp_vf_04.yaml")
    assert resolve_manifest_config(m, 1).endswith("exp_vf_33.yaml")


def test_resolve_ignores_blank_lines(tmp_path: Path) -> None:
    """A trailing newline / blank line must not become a phantom (missing) arm."""
    m = tmp_path / "m.txt"
    m.write_text("a.yaml\n\nb.yaml\n\n")
    assert resolve_manifest_config(m, 1).endswith("b.yaml")
    with pytest.raises(IndexError, match="out of range"):
        resolve_manifest_config(m, 2)  # only 2 real entries, not 4


def test_resolve_out_of_range_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexError, match="out of range"):
        resolve_manifest_config(_manifest(tmp_path), 999)


# --- TRAIN_ITERS cap (the production-incident logic) -------------------------


def _cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


def test_cap_shrinks_both_iterations_and_eval_interval(tmp_path: Path) -> None:
    # max_iterations 500000 (> cap) and eval_interval 500 (> cap) → both capped.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 500000\nvalidation:\n  eval_interval: 500\n",
    )
    overrides, _ = compute_iter_cap_overrides(cfg, 100)
    assert "--override" in overrides
    assert "training.max_iterations=100" in overrides
    assert "validation.eval_interval=100" in overrides


def test_cap_does_not_inflate_short_calibration_arm(tmp_path: Path) -> None:
    # A 1-shot calibration arm (max_iterations 1) must NOT be inflated to 100.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 1\nvalidation:\n  eval_interval: 1\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" not in overrides
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config max_iterations=1" in m for m in messages)


def test_cap_strips_inline_comment_without_sed(tmp_path: Path) -> None:
    # `max_iterations: 1  # VF plan CW-6` — yaml.safe_load drops the comment
    # natively, so the value is 1 (not "116"); the bash needed a sed hack.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 1  # VF plan CW-6\n"
        "validation:\n  eval_interval: 2500\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" not in overrides  # NOT inflated
    assert any("keeping config max_iterations=1" in m for m in messages)
    assert "validation.eval_interval=1" in overrides  # eval capped to eff iters


def test_cap_survives_config_without_eval_interval(tmp_path: Path) -> None:
    # THE regression: a config that omits eval_interval. The bash grep exited 1
    # and `set -e` killed the task (56 mrixfields tasks, job 7145522). Here the
    # missing key is just None → the cap branch injects eval_interval cleanly.
    cfg = _cfg(tmp_path, "config_version: '1.0'\ntraining:\n  max_iterations: 500000\n")
    overrides, _ = compute_iter_cap_overrides(cfg, 100)
    assert "training.max_iterations=100" in overrides
    assert "validation.eval_interval=100" in overrides  # injected for the missing key


def test_cap_reads_the_canonical_schedule_block(tmp_path: Path) -> None:
    """A migrated arm declares ``validation.schedule.interval_steps``.

    Every other fixture in this file spells the interval FLAT, which is what
    hid the drift: the code read ``validation.eval_interval`` and so did the
    tests, so both stayed wrong together while the schema moved the leaf to
    ``validation.schedule.interval_steps`` (RENAMES, 2026-07-31). On a migrated
    arm the read returned ``None``, which reads as "absent" -- safe for the cap
    itself, but it made the already-fires branch unreachable and inflated any
    arm declaring a SMALLER interval than the cap, contradicting the
    cap-not-inflate contract in the docstring.

    50 < 100 here, so the correct answer is to leave it alone.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "training:\n"
        "  max_iterations: 500000\n"
        "validation:\n"
        "  schedule:\n"
        "    interval_steps: 50\n"
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config eval_interval=50" in m for m in messages)


def test_cap_prefers_the_canonical_spelling_over_the_legacy_one(
    tmp_path: Path,
) -> None:
    """Both present and disagreeing: the canonical block wins.

    The loader folds the legacy name INTO the block, so a file carrying both is
    resolved by the block. Reading the flat one first would cap against a value
    the run never uses.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "training:\n"
        "  max_iterations: 500000\n"
        "validation:\n"
        "  eval_interval: 5000\n"
        "  schedule:\n"
        "    interval_steps: 50\n"
    )
    _, messages = compute_iter_cap_overrides(cfg, 100)
    assert any("keeping config eval_interval=50" in m for m in messages)


def test_cap_leaves_eval_interval_that_already_fires(tmp_path: Path) -> None:
    # eval_interval (50) already fires within the capped run (eff=100) → no cap.
    # Legacy flat spelling: still read, as the fallback for unmigrated arms.
    cfg = _cfg(
        tmp_path,
        "training:\n  max_iterations: 500000\nvalidation:\n  eval_interval: 50\n",
    )
    overrides, messages = compute_iter_cap_overrides(cfg, 100)
    assert "validation.eval_interval=100" not in overrides
    assert any("keeping config eval_interval=50" in m for m in messages)


# --- audit gate (moved out of bash — must stay fail-loud) ---------------------


class _FakeRun:
    """Record subprocess.run calls and return a canned returncode per verb."""

    def __init__(self, audit_rc: int = 0, train_rc: int = 0) -> None:
        self.audit_rc = audit_rc
        self.train_rc = train_rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)

        class _R:
            returncode = self.train_rc if "train" in cmd else self.audit_rc

        return _R()


def _real_cfg(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest + an actually-existing config (manifest_dispatch is_file-checks)."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  max_iterations: 5\n")
    man = tmp_path / "m.txt"
    man.write_text(str(cfg) + "\n")
    return man, cfg


def test_audit_failure_returns_2_and_skips_train(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun(audit_rc=2)
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert rc == 2
    assert not any("train" in c for c in fake.calls)  # train never reached


def test_runs_audit_then_train_in_order(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun(audit_rc=0, train_rc=0)
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert rc == 0
    verbs = [
        "audit" if "audit" in c else "train" if "train" in c else "?"
        for c in fake.calls
    ]
    assert verbs == ["audit", "train"]  # audit pre-flight precedes train


def test_no_audit_skips_preflight(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
    )
    assert not any("audit" in c for c in fake.calls)
    assert any("train" in c for c in fake.calls)


def test_dry_run_never_shells_out(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=100,
        resume=False,
        no_audit=False,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
    )
    assert rc == 0
    assert fake.calls == []  # dry-run plans only — no audit/train subprocess


# --- --prod: run the actual experiment (full max_iterations, no smoke cap) ----
# --prod is the opposite of the TRAIN_ITERS smoke cap: it asserts a production run
# (no override) and stamps the mode into the SLURM .out. Combining it with a smoke
# cap is contradictory and must fail loud, not silently let one win (#9/#15).


def test_prod_rejects_train_iters(tmp_path) -> None:
    man, _ = _real_cfg(tmp_path)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=100,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
        prod=True,
    )
    assert rc == 2  # mutually-exclusive: --prod + --train-iters is rejected


def test_prod_dry_run_stamps_mode_and_no_override(tmp_path, capsys) -> None:
    man, _ = _real_cfg(tmp_path)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=True,
        prod=True,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[MODE] PROD" in out
    assert "--override" not in out  # production run — no smoke-cap overrides


def test_prod_passes_no_override_to_train(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    assert "--override" not in train_cmd  # --prod runs the FULL config max_iterations


def test_prod_flag_parsed_by_main(tmp_path, monkeypatch) -> None:
    """The CLI exposes --prod and forwards it through to _dispatch."""
    man, _ = _real_cfg(tmp_path)
    seen = {}

    def _fake_dispatch(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(md, "_dispatch", _fake_dispatch)
    rc = md.main(["--manifest", str(man), "--index", "0", "--prod", "--dry-run"])
    assert rc == 0
    assert seen.get("prod") is True


# --- --output-base: route each task's output into a campaign tree -------------
# When the campaign orchestrator submits a cohort as ONE array job, each task
# must write its checkpoints/CSVs into <campaign_dir>/<config-stem> so the
# orchestrator's _discover_results finds them. --output-base appends a
# `training.output_dir` override pointing there. None (default) keeps the
# config's own output_dir untouched, so the standalone array dispatcher is
# byte-for-byte unchanged.


def test_output_base_overrides_training_output_dir(tmp_path, monkeypatch) -> None:
    man, _cfg = _real_cfg(tmp_path)  # _cfg stem == "arm"
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    base = tmp_path / "campaign"
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
        output_base=str(base),
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    joined = " ".join(train_cmd)
    assert "--override" in train_cmd
    # routed into <base>/<stem>
    assert f"training.output_dir={base / 'arm'}" in joined


def test_output_base_none_leaves_output_dir_untouched(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=True,
        output_base=None,
    )
    assert rc == 0
    train_cmd = next(c for c in fake.calls if "train" in c)
    assert "training.output_dir" not in " ".join(train_cmd)


def test_output_base_composes_with_smoke_cap(tmp_path, monkeypatch) -> None:
    # --output-base routing and the TRAIN_ITERS cap overrides must both land.
    man, _ = _real_cfg(tmp_path)
    fake = _FakeRun()
    monkeypatch.setattr(md.subprocess, "run", fake)
    base = tmp_path / "camp"
    md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=3,
        resume=False,
        no_audit=True,
        dispatch_dir=str(tmp_path / "d"),
        dry_run=False,
        prod=False,
        output_base=str(base),
    )
    train_cmd = next(c for c in fake.calls if "train" in c)
    joined = " ".join(train_cmd)
    assert "training.max_iterations=3" in joined  # smoke cap
    assert f"training.output_dir={base / 'arm'}" in joined  # routing


def test_output_base_flag_parsed_by_main(tmp_path, monkeypatch) -> None:
    man, _ = _real_cfg(tmp_path)
    seen = {}

    def _fake_dispatch(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(md, "_dispatch", _fake_dispatch)
    rc = md.main(
        [
            "--manifest",
            str(man),
            "--index",
            "0",
            "--dry-run",
            "--output-base",
            "/camp/dir",
        ]
    )
    assert rc == 0
    assert seen.get("output_base") == "/camp/dir"
