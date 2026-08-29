"""Argparse-validation tests for :mod:`mriforge.cli.app`.

Per the test-suite roadmap Phase 6, every CLI subcommand needs at
least one **argparse-validation** test: ``--help`` exits 0, missing
required arguments exit 2, invalid choice values exit 2.

We don't actually run any training / inference / SLURM submission
here — argparse's ``parse_args`` halts (via ``SystemExit``) before
the action function fires. Tests just patch ``sys.argv``, call
:func:`mriforge.cli.app.main`, and assert on the resulting ``SystemExit``
code.

The "happy-path" tests (where every required arg is supplied) are
NOT covered here — those run the real subcommand actions, which need
torch / data fixtures and are part of the integration lane.
"""

from __future__ import annotations

import pytest

import mriforge.cli.app as cli_app


# ── Helpers ────────────────────────────────────────────────────────


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    """Invoke ``cli_app.main()`` with a patched ``sys.argv``. Returns
    the exit code raised via ``SystemExit``; re-raises anything else
    so we don't silently swallow real bugs."""
    monkeypatch.setattr("sys.argv", argv)
    try:
        cli_app.main()
    except SystemExit as e:
        # argparse stuffs the exit code into ``code``. It may be an
        # int, ``None`` (== success), or a string for some error
        # messages. Normalise.
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        # String code → treat as failure.
        return 1
    # If main() returned without raising it must have completed —
    # convert that to 0.
    return 0


# ── Top-level parser ───────────────────────────────────────────────


class TestTopLevelParser:
    def test_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "--help"])
        assert rc == 0

    def test_no_subcommand_exits_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``subparsers(..., required=True)`` makes argparse exit 2
        # when no subcommand is given.
        rc = _run_main(monkeypatch, ["mriforge"])
        assert rc == 2

    def test_unknown_subcommand_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "definitely_not_a_command"])
        assert rc == 2


# ── train subcommand ──────────────────────────────────────────────


class TestTrainSubcommand:
    def test_train_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "train", "--help"])
        assert rc == 0

    def test_train_without_config_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # --config is required.
        rc = _run_main(monkeypatch, ["mriforge", "train"])
        assert rc == 2


# ── sanity_check subcommand ───────────────────────────────────────


class TestSanityCheckSubcommand:
    def test_sanity_check_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "sanity_check", "--help"])
        assert rc == 0

    def test_sanity_check_without_config_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "sanity_check"])
        assert rc == 2


# ── predict / export / benchmark / list-features ──────────────────


class TestPredictSubcommand:
    def test_predict_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "predict", "--help"])
        assert rc == 0

    def test_predict_without_model_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # --model and --input are required.
        rc = _run_main(monkeypatch, ["mriforge", "predict", "--input", "/tmp/x"])
        assert rc == 2

    def test_predict_without_input_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch, ["mriforge", "predict", "--model", "/tmp/ckpt.pt"]
        )
        assert rc == 2


class TestExportSubcommand:
    def test_export_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "export", "--help"])
        assert rc == 0

    def test_export_invalid_format_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            [
                "mriforge", "export",
                "--model", "/tmp/m.pt",
                "--config", "/tmp/c.yaml",
                "--format", "definitely_not_a_format",
            ],
        )
        assert rc == 2


class TestBenchmarkSubcommand:
    def test_benchmark_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "benchmark", "--help"])
        assert rc == 0

    def test_benchmark_invalid_suite_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            ["mriforge", "benchmark", "--suite", "not_a_suite"],
        )
        assert rc == 2


class TestListFeaturesSubcommand:
    def test_list_features_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "list-features", "--help"])
        assert rc == 0

    def test_list_features_invalid_module_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            ["mriforge", "list-features", "--module", "not_a_module"],
        )
        assert rc == 2

    def test_list_features_invalid_format_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            ["mriforge", "list-features", "--format", "xml"],  # not allowed
        )
        assert rc == 2


# ── audit subcommand ───────────────────────────────────────────────


class TestAuditSubcommand:
    def test_audit_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "audit", "--help"])
        assert rc == 0

    def test_audit_without_config_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The `config` positional arg is required.
        rc = _run_main(monkeypatch, ["mriforge", "audit"])
        assert rc == 2

    def test_audit_invalid_device_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            ["mriforge", "audit", "/tmp/c.yaml", "--device", "tpu"],
        )
        assert rc == 2


# ── campaign sub-subcommands ───────────────────────────────────────


class TestCampaignSubcommand:
    def test_campaign_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "campaign", "--help"])
        assert rc == 0

    def test_campaign_no_action_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``campaign_sub.dest="campaign_action", required=True``.
        rc = _run_main(monkeypatch, ["mriforge", "campaign"])
        assert rc == 2

    @pytest.mark.parametrize(
        "action",
        ["submit", "status", "evaluate", "cancel", "watch"],
    )
    def test_campaign_action_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        rc = _run_main(
            monkeypatch, ["mriforge", "campaign", action, "--help"]
        )
        assert rc == 0

    @pytest.mark.parametrize(
        "action",
        ["submit", "status", "evaluate", "cancel", "watch"],
    )
    def test_campaign_action_without_required_arg_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        # Each sub-action takes a positional dir/config — missing → 2.
        rc = _run_main(monkeypatch, ["mriforge", "campaign", action])
        assert rc == 2


# ── hpo subcommand ─────────────────────────────────────────────────


class TestHPOSubcommand:
    def test_hpo_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "hpo", "--help"])
        assert rc == 0

    def test_hpo_without_required_args_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "hpo"])
        assert rc == 2

    def test_hpo_invalid_sampler_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            [
                "mriforge", "hpo",
                "--config", "/tmp/c.yaml",
                "--model-type", "unet",
                "--sampler", "definitely_not_a_sampler",
            ],
        )
        assert rc == 2

    def test_hpo_invalid_pruner_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            [
                "mriforge", "hpo",
                "--config", "/tmp/c.yaml",
                "--model-type", "unet",
                "--pruner", "garbage",
            ],
        )
        assert rc == 2


# ── report subcommand ──────────────────────────────────────────────


class TestReportSubcommand:
    def test_report_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "report", "--help"])
        assert rc == 0

    def test_report_without_exp_dir_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "report"])
        assert rc == 2

    def test_report_invalid_task_exits_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(
            monkeypatch,
            ["mriforge", "report", "--exp-dir", "/tmp/e", "--task", "not_a_task"],
        )
        assert rc == 2


# ── meta-evaluate subcommand ───────────────────────────────────────


class TestMetaEvaluateSubcommand:
    def test_meta_evaluate_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc = _run_main(monkeypatch, ["mriforge", "meta-evaluate", "--help"])
        assert rc == 0

    # No required positionals — invocation alone parses cleanly, so we
    # can't assert exit=2 on missing args. The choice-based parameters
    # are int / Path / str without restrictions, so there's no
    # validation surface to test beyond --help.
