"""Argparse-validation tests for every ``spectramr`` subcommand.

Strategy
--------
* Build the parser by calling a thin extractor that reconstructs just the
  ``argparse.ArgumentParser`` without running any training/IO.
* ``--help`` must parse (``SystemExit(0)``).
* Missing required args must produce a non-zero ``SystemExit`` (code 2),
  NOT an ``AttributeError`` or ``TypeError`` stack trace.
* No real training is invoked — we test argument parsing only.

All tests are pure-Python, CPU-only, and import-safe (the real
``main()`` side-effects like ``bootstrap_console_logging`` are skipped
because we never call ``main()``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Parser builder  (mirrors app.main() but returns the parser without running)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Return the REAL ``spectramr`` parser.

    Was a ~200-line hand-mirror of the ``app.main`` subparsers; now delegates to
    the extracted :func:`spectramr.cli.app.build_parser` (PR #130) so this test can
    never drift from the real entry point. The optional subcommands the mirror
    skipped are now present, but every assertion here is per-subcommand
    (required-arg parse / ``--help`` exit-0), so the extra verbs are harmless.
    """
    from spectramr.cli.app import build_parser

    return build_parser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ok(args: list[str]) -> argparse.Namespace:
    """Parse args that should succeed; fail the test if they raise."""
    parser = _build_parser()
    return parser.parse_args(args)


def _parse_fail(args: list[str]) -> None:
    """Assert that parsing these args causes a SystemExit(2) (argparse error)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(args)
    assert exc_info.value.code != 0, (
        f"Expected non-zero SystemExit for args {args!r}, got code 0"
    )


# ---------------------------------------------------------------------------
# No-subcommand: should fail because subparsers is required=True
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_nonzero() -> None:
    """Invoking the parser with no subcommand must exit non-zero."""
    _parse_fail([])


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrainSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["train", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok(["train", "--config", "experiments/x.yaml"])
        assert ns.config == Path("experiments/x.yaml")
        assert ns.debug is False
        assert ns.override is None

    def test_missing_config_exits_nonzero(self) -> None:
        _parse_fail(["train"])

    def test_debug_flag(self) -> None:
        ns = _parse_ok(["train", "--config", "e.yaml", "--debug"])
        assert ns.debug is True

    def test_override_append(self) -> None:
        ns = _parse_ok([
            "train", "--config", "e.yaml",
            "-O", "optimization.optimizer.learning_rate=1e-4",
            "-O", "data.batch_size=8",
        ])
        assert ns.override == [
            "optimization.optimizer.learning_rate=1e-4",
            "data.batch_size=8",
        ]

    def test_resume_accepted(self) -> None:
        ns = _parse_ok(["train", "--config", "e.yaml", "--resume", "auto"])
        assert ns.resume == "auto"

    def test_device_and_seed_now_reach_train(self) -> None:
        """Regression: ``--device``/``--seed`` were absent from the app.py
        train parser, so they could not reach training through the real
        entry point (the sys.argv round-trip into main.py dropped them).
        They must now parse on the single parser."""
        ns = _parse_ok([
            "train", "--config", "e.yaml", "--device", "cpu", "--seed", "7",
        ])
        assert ns.device == "cpu"
        assert ns.seed == 7

    def test_both_dry_run_spellings(self) -> None:
        """--dry-run and the legacy --dry_run both map to dry_run=True."""
        assert _parse_ok(["train", "-c", "e.yaml", "--dry-run"]).dry_run is True
        assert _parse_ok(["train", "-c", "e.yaml", "--dry_run"]).dry_run is True


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------


class TestAblationSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["ablation", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok([
            "ablation", "--config", "base.yaml",
            "--vary", "model.model_kwargs.force_pure_kspace=false",
        ])
        assert ns.config == Path("base.yaml")
        assert ns.vary == ["model.model_kwargs.force_pure_kspace=false"]

    def test_missing_config_exits_nonzero(self) -> None:
        _parse_fail(["ablation", "--vary", "a.b=1"])

    def test_vary_is_repeatable(self) -> None:
        ns = _parse_ok([
            "ablation", "-c", "base.yaml",
            "--vary", "model.model_kwargs.force_pure_kspace=false",
            "--vary", "optimization.optimizer.learning_rate=1e-4",
        ])
        assert len(ns.vary) == 2

    def test_optional_knobs(self) -> None:
        ns = _parse_ok([
            "ablation", "-c", "base.yaml", "--vary", "a.b=1",
            "--output-dir", "out/", "--max-iterations", "50", "--device", "cpu",
        ])
        assert ns.output_dir == Path("out/")
        assert ns.max_iterations == 50
        assert ns.device == "cpu"


# ---------------------------------------------------------------------------
# infer / infer-dataset / experiment / train-distributed
# (ported from main.py into the single cli/app.py parser — 2026-05-29)
# ---------------------------------------------------------------------------


class TestPortedMainCommands:
    def test_infer_minimal(self) -> None:
        ns = _parse_ok([
            "infer", "-c", "cfg.yaml", "-ckpt", "best.pt", "-i", "in/",
        ])
        assert ns.config == Path("cfg.yaml")
        assert ns.checkpoint == Path("best.pt")

    def test_infer_missing_required(self) -> None:
        _parse_fail(["infer", "-c", "cfg.yaml"])  # no checkpoint/input

    def test_infer_dataset_requires_output(self) -> None:
        _parse_fail([
            "infer-dataset", "-c", "c.yaml", "-ckpt", "b.pt", "-i", "in/",
        ])  # --output is required for infer-dataset

    def test_experiment_requires_name(self) -> None:
        _parse_fail(["experiment", "-c", "c.yaml"])  # --experiment required
        ns = _parse_ok(["experiment", "-c", "c.yaml", "-e", "exp1"])
        assert ns.experiment == "exp1"
        assert ns.max_epochs == 100

    def test_train_distributed_backend_choice(self) -> None:
        ns = _parse_ok(["train-distributed", "-c", "c.yaml", "--backend", "gloo"])
        assert ns.backend == "gloo"
        _parse_fail(["train-distributed", "-c", "c.yaml", "--backend", "mpi"])


# ---------------------------------------------------------------------------
# sanity_check
# ---------------------------------------------------------------------------


class TestSanityCheckSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["sanity_check", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok(["sanity_check", "--config", "experiments/x.yaml"])
        assert ns.config == Path("experiments/x.yaml")

    def test_missing_config_exits_nonzero(self) -> None:
        _parse_fail(["sanity_check"])


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredictSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["predict", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok([
            "predict",
            "--model", "checkpoints/best.pt",
            "--input", "data/test/",
        ])
        assert ns.model == Path("checkpoints/best.pt")
        assert ns.input == Path("data/test/")
        assert ns.output == Path("output/")  # default
        # --config is optional at PARSE time (config-less verb contract), but
        # predict() fails loud at run time without it (SSOT rewire).
        assert ns.config is None

    def test_accepts_optional_config(self) -> None:
        ns = _parse_ok([
            "predict",
            "--model", "m.pt",
            "--input", "inp/",
            "--config", "run.yaml",
        ])
        assert ns.config == Path("run.yaml")

    def test_missing_model_exits_nonzero(self) -> None:
        _parse_fail(["predict", "--input", "data/test/"])

    def test_missing_input_exits_nonzero(self) -> None:
        _parse_fail(["predict", "--model", "checkpoints/best.pt"])

    def test_custom_output_dir(self) -> None:
        ns = _parse_ok([
            "predict",
            "--model", "m.pt",
            "--input", "inp/",
            "--output", "out/",
        ])
        assert ns.output == Path("out/")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["benchmark", "--help"])
        assert ei.value.code == 0

    def test_default_suite(self) -> None:
        ns = _parse_ok(["benchmark"])
        assert ns.suite == "standard"

    def test_explicit_suite(self) -> None:
        for suite in ("standard", "memory", "throughput", "all"):
            ns = _parse_ok(["benchmark", "--suite", suite])
            assert ns.suite == suite

    def test_invalid_suite_exits_nonzero(self) -> None:
        _parse_fail(["benchmark", "--suite", "nonexistent"])


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExportSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["export", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok([
            "export",
            "--model", "m.pt",
            "--config", "e.yaml",
        ])
        assert ns.format == "onnx"

    def test_missing_model_exits_nonzero(self) -> None:
        _parse_fail(["export", "--config", "e.yaml"])

    def test_missing_config_exits_nonzero(self) -> None:
        _parse_fail(["export", "--model", "m.pt"])

    def test_format_choices(self) -> None:
        for fmt in ("onnx", "torchscript", "both"):
            ns = _parse_ok(["export", "--model", "m.pt", "--config", "e.yaml", "--format", fmt])
            assert ns.format == fmt

    def test_invalid_format_exits_nonzero(self) -> None:
        _parse_fail(["export", "--model", "m.pt", "--config", "e.yaml", "--format", "tflite"])


# ---------------------------------------------------------------------------
# list-features
# ---------------------------------------------------------------------------


class TestListFeaturesSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["list-features", "--help"])
        assert ei.value.code == 0

    def test_defaults(self) -> None:
        ns = _parse_ok(["list-features"])
        assert ns.module == "all"
        assert ns.format == "markdown"

    def test_all_module_choices(self) -> None:
        for m in ("all", "models", "losses", "metrics", "strategies", "physics", "services"):
            ns = _parse_ok(["list-features", "--module", m])
            assert ns.module == m

    def test_invalid_module_exits_nonzero(self) -> None:
        _parse_fail(["list-features", "--module", "unknown"])


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


class TestAuditSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["audit", "--help"])
        assert ei.value.code == 0

    def test_positional_config_required(self) -> None:
        _parse_fail(["audit"])

    def test_minimal_positional(self) -> None:
        ns = _parse_ok(["audit", "experiments/inprogress/arm.yaml"])
        assert ns.config == Path("experiments/inprogress/arm.yaml")
        assert ns.probe is False
        # D01#2 (2026-08-22): strict is ON by default. CLAUDE.md
        # non-negotiable 4 has always said so; argparse said otherwise until
        # this flip, so a bare `spectramr audit <arm>` exited 1 on warnings and
        # read as a soft pass. `--no-strict` is the opt-out.
        assert ns.strict is True
        assert ns.json is False

    def test_no_strict_opts_out(self) -> None:
        assert _parse_ok(["audit", "e.yaml", "--no-strict"]).strict is False

    def test_probe_and_strict_flags(self) -> None:
        ns = _parse_ok(["audit", "e.yaml", "--probe", "--strict", "--json"])
        assert ns.probe is True
        assert ns.strict is True
        assert ns.json is True

    def test_device_choices(self) -> None:
        for dev in ("cpu", "cuda"):
            ns = _parse_ok(["audit", "e.yaml", "--device", dev])
            assert ns.device == dev

    def test_invalid_device_exits_nonzero(self) -> None:
        _parse_fail(["audit", "e.yaml", "--device", "mps"])


# ---------------------------------------------------------------------------
# hpo
# ---------------------------------------------------------------------------


class TestHpoSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["hpo", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok([
            "hpo", "--config", "e.yaml", "--model-type", "unet",
        ])
        assert ns.config == Path("e.yaml")
        assert ns.model_type == ["unet"]
        assert ns.n_trials == 50  # default

    def test_missing_config_exits_nonzero(self) -> None:
        _parse_fail(["hpo", "--model-type", "unet"])

    def test_missing_model_type_exits_nonzero(self) -> None:
        _parse_fail(["hpo", "--config", "e.yaml"])

    def test_repeatable_model_type(self) -> None:
        ns = _parse_ok([
            "hpo", "--config", "e.yaml",
            "--model-type", "unet", "--model-type", "vit_kan",
        ])
        assert ns.model_type == ["unet", "vit_kan"]

    def test_sampler_choices(self) -> None:
        for s in ("tpe", "cmaes", "nsga2"):
            ns = _parse_ok(["hpo", "--config", "e.yaml", "--model-type", "u", "--sampler", s])
            assert ns.sampler == s

    def test_pruner_choices(self) -> None:
        for p in ("hyperband", "median", "successive_halving", "threshold", "none"):
            ns = _parse_ok(["hpo", "--config", "e.yaml", "--model-type", "u", "--pruner", p])
            assert ns.pruner == p

    def test_invalid_sampler_exits_nonzero(self) -> None:
        _parse_fail(["hpo", "--config", "e.yaml", "--model-type", "u", "--sampler", "random"])

    def test_list_presets_flag(self) -> None:
        ns = _parse_ok([
            "hpo", "--config", "e.yaml", "--model-type", "u", "--list-presets",
        ])
        assert ns.list_presets is True

    def test_n_trials_override(self) -> None:
        ns = _parse_ok([
            "hpo", "--config", "e.yaml", "--model-type", "u", "--n-trials", "10",
        ])
        assert ns.n_trials == 10


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReportSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["report", "--help"])
        assert ei.value.code == 0

    def test_minimal_required_args(self) -> None:
        ns = _parse_ok(["report", "--exp-dir", "experiments/results/run1/"])
        assert ns.exp_dir == Path("experiments/results/run1/")
        # task/html/qc default to None so the handler can fall back to a
        # --config reporting block; the handler resolves task→'default'.
        assert ns.task is None
        assert ns.html is None
        assert ns.qc is None
        assert ns.config is None

    def test_missing_exp_dir_exits_nonzero(self) -> None:
        _parse_fail(["report"])

    def test_task_choices(self) -> None:
        for t in ("default", "reconstruction", "synthesis", "super_resolution",
                  "gan", "diffusion", "vae"):
            ns = _parse_ok(["report", "--exp-dir", "x/", "--task", t])
            assert ns.task == t

    def test_invalid_task_exits_nonzero(self) -> None:
        _parse_fail(["report", "--exp-dir", "x/", "--task", "unknown_task"])

    def test_config_flag_parses(self) -> None:
        ns = _parse_ok(["report", "--exp-dir", "x/", "--config", "arm.yaml"])
        assert ns.config == Path("arm.yaml")

    def test_figures_opt_in_subset(self) -> None:
        # Default: no --figures -> None -> the handler plots ALL (the SSOT report).
        assert _parse_ok(["report", "-e", "x/"]).figures is None
        ns = _parse_ok(["report", "-e", "x/", "--figures",
                        "fig_1_2_learning_curves,qc_group_strip"])
        assert ns.figures == "fig_1_2_learning_curves,qc_group_strip"

    def test_html_toggle_tri_state(self) -> None:
        assert _parse_ok(["report", "-e", "x/", "--html"]).html is True
        assert _parse_ok(["report", "-e", "x/", "--no-html"]).html is False

    def test_qc_toggle_tri_state(self) -> None:
        assert _parse_ok(["report", "-e", "x/", "--qc"]).qc is True
        assert _parse_ok(["report", "-e", "x/", "--no-qc"]).qc is False

    def test_interactive_toggle_tri_state(self) -> None:
        assert _parse_ok(["report", "-e", "x/"]).interactive is None
        assert _parse_ok(["report", "-e", "x/", "--interactive"]).interactive is True
        assert _parse_ok(["report", "-e", "x/", "--no-interactive"]).interactive is False

    def test_recursive_flag(self) -> None:
        assert _parse_ok(["report", "-e", "results/"]).recursive is False
        assert _parse_ok(["report", "-e", "results/", "--recursive"]).recursive is True
        assert _parse_ok(["report", "-e", "results/", "-r"]).recursive is True
        assert _parse_ok(["report", "-e", "results/", "--all"]).recursive is True


# ---------------------------------------------------------------------------
# meta-evaluate
# ---------------------------------------------------------------------------


class TestMetaEvaluateSubcommand:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["meta-evaluate", "--help"])
        assert ei.value.code == 0

    def test_defaults_no_args(self) -> None:
        ns = _parse_ok(["meta-evaluate"])
        assert ns.input is None
        assert ns.severities == 8
        assert ns.max_subjects == 8
        assert ns.seed == 42
        assert ns.device == "auto"
        # Run-shape knobs default to the "real" 2-D implementation.
        assert ns.eval_mode == "2d"
        assert ns.betting is False
        assert ns.betting_alpha == 0.05

    def test_betting_and_eval_mode_flags(self) -> None:
        ns = _parse_ok([
            "meta-evaluate",
            "--eval-mode", "3d",
            "--betting",
            "--betting-alpha", "0.1",
        ])
        assert ns.eval_mode == "3d"   # parsing succeeds; the CLI raises at runtime
        assert ns.betting is True
        assert ns.betting_alpha == 0.1

    def test_eval_mode_rejects_unknown_choice(self) -> None:
        with pytest.raises(SystemExit) as ei:
            _build_parser().parse_args(["meta-evaluate", "--eval-mode", "4d"])
        assert ei.value.code == 2

    def test_explicit_input_and_metrics(self) -> None:
        ns = _parse_ok([
            "meta-evaluate",
            "--input", "data/refs/",
            "--metrics", "psnr", "ssim",
        ])
        assert ns.input == Path("data/refs/")
        assert ns.metrics == ["psnr", "ssim"]

    def test_figures_tables_flags(self) -> None:
        ns = _parse_ok(["meta-evaluate", "--figures", "--tables"])
        assert ns.figures is True
        assert ns.tables is True
