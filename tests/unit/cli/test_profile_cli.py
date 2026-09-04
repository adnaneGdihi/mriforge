"""Tests for the ``spectramr profile`` verb: registration, dry-run, and outcome.

The config loader is stubbed throughout — these cover the verb's control flow,
not config parsing. The one test that must not be stubbed is the registration
check: a verb that builds a perfect argv but never appears on the parser is the
"registered but unwired" shape non-negotiable 16 exists for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectramr.cli import profile_cli

# ------------------------------------------------------------- registration


def test_profile_is_registered_on_the_real_parser():
    from spectramr.cli.app import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert "profile" in sub.choices


def test_profile_dispatches_to_the_handler():
    """``main`` dispatches through ``args.func``; a subparser that forgets
    ``set_defaults`` parses fine and then crashes on AttributeError."""
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(["profile", "--config", "a.yaml"])
    assert args.func is profile_cli.profile


def test_profile_is_flagged_as_a_heavy_startup_command():
    """The parent loads the arm's config before it can resolve the run dir, so
    it pays the torch import too — without this the CLI looks frozen."""
    from spectramr.cli.app import _HEAVY_STARTUP_COMMANDS

    assert "profile" in _HEAVY_STARTUP_COMMANDS


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--target", "train"), ("--mode", "full"), ("--focus", "all")],
)
def test_defaults(flag, expected):
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(["profile", "--config", "a.yaml"])
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == expected


@pytest.mark.parametrize(
    "bad", [["--target", "hpo"], ["--mode", "gpu-only"], ["--focus", "everything"]]
)
def test_unknown_option_values_are_rejected_not_defaulted(bad):
    """A closed choice set must raise, never degrade to a default (NN3)."""
    from spectramr.cli.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["profile", "--config", "a.yaml", *bad])


# ----------------------------------------------------------------- dry run


def stub_settings(**over):
    """A settings-shaped stub: enough for the verb and both pre-flight checks.

    Defaults are deliberately the *permissive* ones — ``strategy: none`` and an
    epoch-derived (``None``) budget — so a test that does not care about
    pre-flight is not silently exercising it. Tests that DO care override.
    """
    from types import SimpleNamespace

    base = {
        "training": SimpleNamespace(output_dir="experiments/results/exp_11", max_iterations=None),
        "validation": SimpleNamespace(schedule=SimpleNamespace(interval_steps=100, on_epoch=False)),
        "parallel": SimpleNamespace(strategy="none"),
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_config(monkeypatch):
    """Stub the config load; these tests are about the verb, not the schema."""
    monkeypatch.setattr(profile_cli, "_load_arm", lambda p: stub_settings())


def _args(**over):
    import argparse

    base = {
        "config": Path("experiments/validated/dummy_gan.yaml"),
        "target": "train",
        "mode": "cpu-only",
        "focus": "train-loop",
        "experiment": None,
        "device": "cpu",
        "val_batches": None,
        "no_phase_split": False,
        "dry_run": True,
        "extra": [],
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_dry_run_prints_the_command_and_writes_nothing(stub_config, capsys, tmp_path):
    created_before = set(Path("experiments/results").glob("exp_11/profiles/*"))
    assert profile_cli.profile(_args()) == 0
    out = capsys.readouterr().out
    assert "-m scalene run" in out
    assert "--program-path" in out
    assert set(Path("experiments/results").glob("exp_11/profiles/*")) == created_before


def test_dry_run_works_without_scalene(stub_config, monkeypatch, capsys):
    """The command should be inspectable on a box that cannot run it."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert profile_cli.profile(_args()) == 0
    assert "scalene run" in capsys.readouterr().out


def test_real_run_raises_without_scalene(stub_config, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        profile_cli.profile(_args(dry_run=False))


# ------------------------------------------------------------ outcome paths


@pytest.fixture
def fake_run(monkeypatch, tmp_path, stub_config):
    """Redirect the results root into tmp and stub the child process."""
    import spectramr.cli.profile_paths as pp

    monkeypatch.setattr(pp, "RESULTS_ROOT", tmp_path / "experiments" / "results")
    monkeypatch.setattr(profile_cli, "require_scalene", lambda: "2.3.0")
    return tmp_path / "experiments" / "results" / "exp_11" / "profiles"


def _only_profile_dir(profiles_root: Path) -> Path:
    (found,) = list(profiles_root.iterdir())
    return found


def test_successful_run_writes_manifest_and_reports_the_modes(fake_run, monkeypatch):
    def child(argv, log_path):
        # Stand in for scalene: emit the outfile the real one would write.
        Path(argv[argv.index("--outfile") + 1]).write_text("{}", encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["mode"] == "cpu-only"
    assert manifest["modes"]["focus"] == "train-loop"
    assert manifest["outcome"] == {
        "exit_code": 0,
        "duration_s": pytest.approx(manifest["outcome"]["duration_s"]),
        "outfile_written": True,
        # A green run carries no diagnosis. Asserted rather than omitted: a
        # classifier that fired on success would attach a failure explanation to
        # a profile that worked.
        "diagnosis": None,
    }


def test_manifest_survives_a_child_that_never_returns_a_profile(fake_run, monkeypatch):
    """The manifest is written BEFORE the run, so a killed job still records
    which modes were in flight."""

    def child(argv, log_path):
        log_path.write_text("boom\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(profile_cli, "_run_child", child)
    with pytest.raises(KeyboardInterrupt):
        profile_cli.profile(_args(dry_run=False))

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["modes"]["mode"] == "cpu-only"
    assert "outcome" not in manifest


def test_green_child_with_no_profile_written_is_a_failure(fake_run, monkeypatch, caplog):
    """The --program-path failure mode: the run succeeds and reports nothing.
    Exiting 0 there would read as 'nothing was slow'."""
    monkeypatch.setattr(
        profile_cli, "_run_child", lambda argv, log: log.write_text("", "utf-8") or 0
    )
    assert profile_cli.profile(_args(dry_run=False)) == 1
    assert any("wrote no profile" in r.getMessage() for r in caplog.records)


def test_child_exit_code_is_propagated(fake_run, monkeypatch):
    monkeypatch.setattr(
        profile_cli, "_run_child", lambda argv, log: log.write_text("", "utf-8") or 3
    )
    assert profile_cli.profile(_args(dry_run=False)) == 3


def test_a_recognized_crash_is_diagnosed_into_the_manifest(fake_run, monkeypatch, caplog):
    """The production wiring, not the classifier: registering a diagnostic that
    ``profile()`` never calls is the "capability, unwired" shape of
    non-negotiable 16. Driven through the real entry point, and asserted on the
    artifact an operator reads days later rather than only on the log."""
    crash = (
        '  File "<unknown>", line 0, in std::_Sp_counted_ptr<'
        "torch::profiler::impl::Result*, (__gnu_cxx::_Lock_policy)2>::_M_dispose()\n"
    )

    def child(argv, log_path):
        # NOT `log.write_text(...) or 245`: write_text returns the character
        # count, so that idiom yields the exit code only for an empty string.
        log_path.write_text(crash, encoding="utf-8")
        return 245

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 245

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    diagnosis = manifest["outcome"]["diagnosis"]
    assert diagnosis["status"] == "recognized"
    assert diagnosis["signature"] == "scalene_torch_profiler_segfault"
    assert any("cpu-only" in r.getMessage() for r in caplog.records)


def test_an_unrecognized_crash_still_propagates_and_records(fake_run, monkeypatch):
    """The classifier must not swallow a failure it cannot name."""

    def child(argv, log_path):
        log_path.write_text("odd\n", encoding="utf-8")
        return 7

    monkeypatch.setattr(profile_cli, "_run_child", child)
    assert profile_cli.profile(_args(dry_run=False)) == 7

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["outcome"]["diagnosis"]["status"] == "unrecognized"


# --------------------------------------------------- phase split + val cap


_SRC = "/repo/src/spectramr"
_TRAIN_FRAME = f"{_SRC}/pipelines/training_loop.py _execute_training_loop:812;"
_VAL_FRAME = f"{_SRC}/pipelines/train.py _run_validation:1955;"


def _profile_with_stacks() -> str:
    """A minimal profile in scalene 2.3.0's real shape."""

    def stack(frames, cpu):
        return [frames, {"cpu_samples": cpu, "c_time": 0.0, "python_time": cpu, "count": 1}]

    return json.dumps(
        {"stacks": [stack([_TRAIN_FRAME], 7.0), stack([_TRAIN_FRAME, _VAL_FRAME], 3.0)]}
    )


def _child_writing(payload: str):
    def child(argv, log_path):
        Path(argv[argv.index("--outfile") + 1]).write_text(payload, encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    return child


def test_phase_split_writes_separate_train_and_validation_reports(fake_run, monkeypatch):
    """The whole point of the feature: one run, two reports."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing(_profile_with_stacks()))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    profile_dir = _only_profile_dir(fake_run)
    phases = profile_dir / "phases"
    assert (phases / "train.json").exists()
    assert (phases / "validation.json").exists()

    summary = json.loads((phases / "summary.json").read_text())
    assert summary["phases"]["train"]["cpu_share"] == 7.0
    assert summary["phases"]["validation"]["cpu_share"] == 3.0

    manifest = json.loads((profile_dir / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "written"


def test_a_profile_without_stacks_degrades_instead_of_failing_the_run(fake_run, monkeypatch):
    """A splitter problem must not fail a run that already paid for the profile."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing("{}"))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "unavailable"
    assert "stacks" in manifest["phase_split"]["reason"]


def test_unreadable_profile_is_recorded_not_inferred(fake_run, monkeypatch):
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing("not json at all"))
    assert profile_cli.profile(_args(dry_run=False)) == 0

    manifest = json.loads((_only_profile_dir(fake_run) / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "unavailable"


def test_no_phase_split_says_it_was_disabled_rather_than_unavailable(fake_run, monkeypatch):
    """'Disabled' and 'unavailable' are different findings and must not blur."""
    monkeypatch.setattr(profile_cli, "_run_child", _child_writing(_profile_with_stacks()))
    assert profile_cli.profile(_args(dry_run=False, no_phase_split=True)) == 0

    profile_dir = _only_profile_dir(fake_run)
    assert not (profile_dir / "phases").exists()
    manifest = json.loads((profile_dir / "profile_manifest.json").read_text())
    assert manifest["phase_split"]["status"] == "disabled"


def test_val_batches_is_refused_where_it_could_not_act(stub_config):
    """`infer` runs no validation loop, so the cap would be silently dropped."""
    with pytest.raises(ValueError, match="--val-batches is meaningless"):
        profile_cli.profile(_args(target="infer", val_batches=4))
