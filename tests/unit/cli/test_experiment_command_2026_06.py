"""Regression: the ``experiment`` CLI command was dead-on-arrival.

2026-06-11. ``experiment_command`` built an ``ExperimentDirector`` whose
``validate()`` raised unconditionally (it required generator/loss config the CLI
never supplied), so the command exited 1 before reaching ``run_training_pipeline``.
It now drops the director and translates ``--experiment`` / ``--max-epochs`` /
``--checkpoint-interval`` into config overrides on the SSOT settings, then runs
the canonical training pipeline.
"""

from __future__ import annotations

import argparse

import pytest

from spectramr import main as main_mod


def test_experiment_command_applies_knobs_and_runs_pipeline(monkeypatch):
    captured = {}

    def fake_run(settings, device=None, resume_path=None, **kw):
        captured["settings"] = settings
        captured["resume_path"] = resume_path
        return {"success": True}

    # Avoid touching CUDA / determinism global state in the test.
    monkeypatch.setattr(main_mod, "initialize_accelerator", lambda *a, **k: None)
    # ``run_training_pipeline`` is imported lazily INSIDE experiment_command
    # (startup-perf work, 2026-06), so it must be patched at its source
    # module, not on spectramr.main.
    monkeypatch.setattr("spectramr.pipelines.run_training_pipeline", fake_run)

    args = argparse.Namespace(
        config="experiments/inprogress/dummy/dummy_gan.yaml",
        experiment="exp_unit_test",
        max_epochs=7,
        checkpoint_interval=3,
        resume=None,
        device="cpu",
        seed=0,
        override=None,
    )
    main_mod.experiment_command(args)

    s = captured["settings"]
    assert s.training.epochs == 7
    assert s.checkpoint.save_interval == 3
    assert s.training.output_dir == "experiments/results/exp_unit_test"
    assert captured["resume_path"] is None


def test_experiment_command_honours_config_seed_and_determinism(monkeypatch):
    """Pitfall-#15 regression (2026-07-01): the accelerator used to be
    initialized BEFORE the config was loaded — hardcoded seed 42 and
    deterministic=True — so ``run.seed`` / ``training.deterministic``
    were silent no-ops on this *training* path. The init now runs after
    config load + overrides with the config-resolved policy."""
    captured = {}

    def fake_init(device, seed, deterministic=True):
        captured["init"] = (device, seed, deterministic)

    monkeypatch.setattr(main_mod, "initialize_accelerator", fake_init)
    monkeypatch.setattr(
        "spectramr.pipelines.run_training_pipeline",
        lambda settings, device=None, resume_path=None, **kw: {"success": True},
    )
    args = argparse.Namespace(
        config="experiments/inprogress/dummy/dummy_gan.yaml",
        experiment="exp_determinism", max_epochs=1, checkpoint_interval=1,
        resume=None, device="cpu", seed=None,
        # `training.deterministic` was NOT renamed; only the seed moved.
        override=["training.deterministic=false", "run.seed=1234"],
    )
    main_mod.experiment_command(args)

    assert captured["init"] == ("cpu", 1234, False), (
        "experiment_command must resolve seed/determinism from the "
        "post-override config, not hardcode (42, deterministic=True)"
    )


def test_experiment_command_does_not_use_dead_director():
    import inspect

    src = inspect.getsource(main_mod.experiment_command)
    # The dead director must not be constructed/used (a prose mention in a
    # comment is fine — we assert on the constructor call).
    assert "ExperimentDirector(" not in src
    assert ".validate().build()" not in src
    assert "run_training_pipeline" in src


def test_experiment_command_propagates_pipeline_failure(monkeypatch):
    monkeypatch.setattr(main_mod, "initialize_accelerator", lambda *a, **k: None)
    monkeypatch.setattr(
        "spectramr.pipelines.run_training_pipeline",
        lambda *a, **k: {"success": False, "error": "boom"},
    )
    args = argparse.Namespace(
        config="experiments/inprogress/dummy/dummy_gan.yaml",
        experiment="exp_fail", max_epochs=1, checkpoint_interval=1,
        resume=None, device="cpu", seed=0, override=None,
    )
    with pytest.raises(SystemExit) as exc:
        main_mod.experiment_command(args)
    assert exc.value.code == 1
