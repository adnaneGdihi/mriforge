"""Entry-point dispatch regression tests (2026-05-29 CLI cleanup).

Covers the two behavioral changes that the argparse-mirror tests in
``test_argparse_subcommands.py`` cannot reach:

1. ``cli/app.py::train``/``sanity_check`` dispatch to ``main.py``'s command
   functions DIRECTLY — they no longer rebuild ``sys.argv`` and re-invoke a
   second parser. This pins that the round-trip is gone (the source of the
   dropped ``--device``/``--seed`` and broken ``--debug`` flags).
2. The ``ablation`` subcommand: ``--vary`` → override-dict parsing with type
   coercion, and the training-backed ``train_and_score`` evaluator that reads
   ``validation_metrics.csv`` back into a metrics dict.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. train/sanity no longer round-trip through sys.argv
# ---------------------------------------------------------------------------


def test_train_dispatches_directly_without_sys_argv_rewrite(monkeypatch):
    """``train(args)`` must call ``main.train_command(args)`` and NOT mutate
    ``sys.argv`` (the historical fake-argv round-trip)."""
    import sys

    import mriforge.cli.app as app

    called = {}

    def fake_train_command(args):
        called["args"] = args

    # main.py import-time side effects are real; patch the command symbol it
    # exposes so we don't actually train.
    import mriforge.main as main_mod

    monkeypatch.setattr(main_mod, "train_command", fake_train_command)
    monkeypatch.setattr(main_mod, "sanity_check_command", lambda a: None)

    sentinel_argv = ["pytest-sentinel"]
    monkeypatch.setattr(sys, "argv", list(sentinel_argv))

    args = argparse.Namespace(
        config=Path("x.yaml"), debug=False, dry_run=False,
        device="cpu", seed=11, override=None, resume=None,
    )
    rc = app.train(args)

    assert rc == 0
    assert called["args"] is args, "train must pass the parsed args through directly"
    # The round-trip is gone: sys.argv is untouched.
    assert sys.argv == sentinel_argv, "train must not rebuild sys.argv"
    # The args surface main.py reads is normalized, not dropped.
    assert called["args"].device == "cpu"
    assert called["args"].seed == 11


def test_sanity_check_dispatches_to_sanity_command(monkeypatch):
    import mriforge.cli.app as app
    import mriforge.main as main_mod

    called = {}
    monkeypatch.setattr(main_mod, "sanity_check_command", lambda a: called.setdefault("hit", a))
    monkeypatch.setattr(main_mod, "train_command", lambda a: pytest.fail("wrong dispatch"))

    args = argparse.Namespace(config=Path("x.yaml"), debug=False)
    rc = app.sanity_check(args)
    assert rc == 0
    assert "hit" in called


# ---------------------------------------------------------------------------
# 2. ablation --vary parsing
# ---------------------------------------------------------------------------


def test_parse_vary_specs_type_coercion_and_naming():
    from mriforge.cli.app import _parse_vary_specs

    specs = _parse_vary_specs([
        "model.model_kwargs.force_pure_kspace=false",
        "optimization.optimizer.learning_rate=1e-4",
        "model.model_kwargs.dc_method=none",
    ])
    assert specs["force_pure_kspace_false"]["overrides"][
        "model.model_kwargs.force_pure_kspace"
    ] is False
    assert specs["learning_rate_1e-4"]["overrides"][
        "optimization.optimizer.learning_rate"
    ] == pytest.approx(1e-4)
    assert specs["dc_method_none"]["overrides"][
        "model.model_kwargs.dc_method"
    ] is None


def test_parse_vary_specs_rejects_malformed():
    from mriforge.cli.app import _parse_vary_specs

    with pytest.raises(ValueError, match="no '='"):
        _parse_vary_specs(["missing_equals"])
    with pytest.raises(ValueError, match="empty key"):
        _parse_vary_specs(["=value"])


# ---------------------------------------------------------------------------
# 3. train_and_score evaluator: train → read validation_metrics.csv
# ---------------------------------------------------------------------------


def _write_val_csv(path: Path) -> None:
    path.write_text(
        "iteration,val_robust_mri_psnr_2x,val_loss\n"
        "10,18.0,0.9\n"
        "20,19.5,0.8\n"   # best psnr (max) = 19.5, best loss (min) = 0.8
    )


def test_train_and_score_reads_best_metrics_from_csv(monkeypatch, tmp_path):
    """The default evaluation_fn must train then summarise the CSV — pinning
    the exact metric keys so a silent-empty regression is caught."""
    import mriforge.pipelines.ablation as abl

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_val_csv(log_dir / "validation_metrics.csv")

    # ``training.output_dir`` must point at tmp so the canonical CSV candidate
    # (<output_dir>/logs/validation_metrics.csv) resolves to THIS file —
    # otherwise a leftover repo ``experiments/outputs/logs/validation_metrics.csv``
    # from a real run shadows the test (it is tried first).
    config = argparse.Namespace(
        training=argparse.Namespace(output_dir=str(tmp_path)),
        logging=argparse.Namespace(log_dir=str(log_dir)),
    )

    # Patch the training pipeline + summariser at their import site inside
    # train_and_score (it imports them locally from mriforge.pipelines.train).
    import mriforge.pipelines.train as train_mod

    monkeypatch.setattr(
        train_mod, "run_training_pipeline",
        lambda cfg, device="cuda": {"success": True, "final_loss": 0.8},
    )

    out = abl.train_and_score(config, device="cpu")
    assert out["success"] is True
    assert out["val_robust_mri_psnr_2x_best"] == pytest.approx(19.5)
    assert out["val_loss_best"] == pytest.approx(0.8)
    assert out["final_loss"] == pytest.approx(0.8)


def test_train_and_score_propagates_training_failure(monkeypatch, tmp_path):
    import mriforge.pipelines.ablation as abl
    import mriforge.pipelines.train as train_mod

    config = argparse.Namespace(
        logging=argparse.Namespace(log_dir=str(tmp_path))
    )
    monkeypatch.setattr(
        train_mod, "run_training_pipeline",
        lambda cfg, device="cuda": {"success": False, "error": "boom"},
    )
    out = abl.train_and_score(config, device="cpu")
    assert out["success"] is False
    assert "boom" in out["error"]


# ---------------------------------------------------------------------------
# 4. ported commands dispatch to the right main.py handler
# ---------------------------------------------------------------------------


def test_infer_dispatches_to_infer_command(monkeypatch):
    import mriforge.cli.app as app
    import mriforge.main as main_mod

    hit = {}
    monkeypatch.setattr(main_mod, "infer_command", lambda a: hit.setdefault("a", a))
    args = argparse.Namespace(
        config=Path("c.yaml"), checkpoint=Path("b.pt"), input=Path("in/"),
        output=None, device="cpu",
    )
    assert app.infer(args) == 0
    assert hit["a"] is args


def test_experiment_dispatches_to_experiment_command(monkeypatch):
    import mriforge.cli.app as app
    import mriforge.main as main_mod

    hit = {}
    monkeypatch.setattr(main_mod, "experiment_command", lambda a: hit.setdefault("a", a))
    args = argparse.Namespace(
        config=Path("c.yaml"), experiment="exp1", max_epochs=10,
        checkpoint_interval=5, device="cpu",
    )
    assert app.experiment(args) == 0
    assert hit["a"].experiment == "exp1"


# ---------------------------------------------------------------------------
# 5. main.py:main() is now a deprecation shim that delegates to cli.app.main
# ---------------------------------------------------------------------------


def test_main_py_main_is_deprecated_shim(monkeypatch):
    """``mriforge.main:main`` must warn and delegate to ``cli.app.main`` rather
    than own a second parser."""
    import mriforge.cli.app as app
    import mriforge.main as main_mod

    monkeypatch.setattr(app, "main", lambda: 0)
    with pytest.warns(DeprecationWarning, match="deprecated"), pytest.raises(SystemExit) as ei:
        main_mod.main()
    assert ei.value.code == 0

    # The second parser is gone: main.py no longer builds argparse subparsers.
    src = (
        Path(main_mod.__file__).read_text()
    )
    assert 'add_subparsers' not in src, (
        "main.py still builds its own subparsers — the dual-parser seam "
        "must stay collapsed (cli/app.py is the single parser)."
    )
