import inspect

from spectramr.pipelines import train
from tests.utils.config_block_stub import block_stub


def test_maybe_run_reporting_threads_style_and_formats():
    src = inspect.getsource(train._maybe_run_reporting)
    assert "style=" in src
    assert "formats=" in src
    assert "dpi=" in src
    assert "emit_manifest=" in src
    assert "submission_bundle=" in src


def test_build_report_case_recorder_returns_none_when_disabled():
    class _Rep:
        enabled = False
    class _Cfg:
        reporting = _Rep()
    class _Strat:
        pass
    assert train._build_report_case_recorder(_Cfg(), _Strat()) is None


def test_build_report_case_recorder_attaches_when_enabled():
    class _Rep:
        enabled = True
        n_report_cases = 4
        case_selection = "best_median_worst"
    # `save_report_cases` folded to `report_cases.enabled`; train.py:793/:1012
    # read `logging.report_cases.{subdir,enabled}`, so a flat boolean left the
    # whole sub-block missing.
    class _Cfg:
        reporting = _Rep()
        validation = block_stub("validation", primary_metric="psnr")
        logging = block_stub("logging", save_report_cases=True)
    class _Strat:
        pass
    strat = _Strat()
    rec = train._build_report_case_recorder(_Cfg(), strat)
    assert rec is not None
    assert getattr(strat, "_report_case_recorder", None) is rec
    assert rec.n_cases == 4


def test_maybe_run_reporting_threads_panel_labels():
    import inspect

    src = inspect.getsource(train._maybe_run_reporting)
    assert "panel_labels=" in src


def test_maybe_run_reporting_threads_tikz():
    """The reporting.tikz knob must reach generate_report (pitfall #15:
    an advertised knob is read + passed through in the same change)."""
    src = inspect.getsource(train._maybe_run_reporting)
    assert "tikz=" in src


def test_maybe_run_reporting_threads_qc_and_html_knobs():
    """The qc_figures / html_report knobs must reach generate_report."""
    src = inspect.getsource(train._maybe_run_reporting)
    assert "qc_figures=" in src
    assert "html_report=" in src


def test_maybe_run_reporting_threads_interactive_knob():
    """The interactive knob must reach generate_report (pitfall #15)."""
    src = inspect.getsource(train._maybe_run_reporting)
    assert "interactive=" in src


def test_build_report_case_recorder_threads_record_volumes():
    """reporting.record_volumes must reach the ReportCaseRecorder."""
    class _Rep:
        enabled = True
        n_report_cases = 3
        case_selection = "best_median_worst"
        record_volumes = True
    # `save_report_cases` folded to `report_cases.enabled`; train.py:793/:1012
    # read `logging.report_cases.{subdir,enabled}`, so a flat boolean left the
    # whole sub-block missing.
    class _Cfg:
        reporting = _Rep()
        validation = block_stub("validation", primary_metric="psnr")
        logging = block_stub("logging", save_report_cases=True)
    strat = type("_S", (), {})()
    rec = train._build_report_case_recorder(_Cfg(), strat)
    assert rec is not None and rec.record_volumes is True


def test_build_per_case_sink_none_when_disabled():
    class _Rep:
        enabled = False
    class _Cfg:
        reporting = _Rep()
    assert train._build_per_case_metric_sink(_Cfg(), object()) is None


def test_build_per_case_sink_none_when_knob_off():
    class _Rep:
        enabled = True
        per_call_metrics = False
    class _Cfg:
        reporting = _Rep()
    assert train._build_per_case_metric_sink(_Cfg(), object()) is None


def test_build_per_case_sink_attaches_when_enabled():
    class _Rep:
        enabled = True
        per_call_metrics = True
    class _Cfg:
        reporting = _Rep()
    class _Strat:
        pass
    strat = _Strat()
    sink = train._build_per_case_metric_sink(_Cfg(), strat)
    assert sink is not None and sink.enabled is True
    assert getattr(strat, "_per_case_metric_sink", None) is sink


class TestRunSummaryIsOnDiskBeforeTheHookDraws:
    """``run_summary.json`` is a *report artifact*, so it has to exist before
    the reporting hook runs -- not after it.

    ``reporting/aggregator.py`` collects five artifacts, and this is one of
    them; two registered plotters read it. ``fig_1_16_run_summary_card`` is in
    all eight task presets and ``fig_1_15_computational_profile`` in five, so
    while the hook ran first, *every* training run in the repo drew a figure
    set with those two missing -- they rendered only when someone re-ran the
    ``report`` verb by hand afterwards. Same run, two different figure sets,
    decided by which entry point drew them.

    This is a behavioural test, not a source-order pin: the spy answers
    "did the file exist at the moment the hook fired", which is the property
    that actually matters and the one a later refactor could silently break.
    """

    @staticmethod
    def _config_dict(tmp_path) -> dict:
        return {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "checkpoint": {},
            "losses": {
                "output_domain": "image",
                "image_losses": [{"name": "l1", "weight": 1.0}],
            },
            "training": {
                "strategy_class": "reconstruction",
                "output_dir": str(tmp_path),
                "epochs": 1,
            },
        }

    @staticmethod
    def _env(cfg):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from spectramr.infrastructure.training.builders.environment import (
            TrainingEnvironment,
        )

        model = nn.Conv2d(2, 2, 3, padding=1)
        loader = DataLoader(TensorDataset(torch.randn(2, 2, 8, 8)), batch_size=1)
        return TrainingEnvironment.from_components(
            config=cfg,
            models={"generator": model},
            optimizers={"opt_g": torch.optim.Adam(model.parameters())},
            losses={"l1": nn.L1Loss()},
            data_loaders={"train": loader, "val": loader},
            device="cpu",
        )

    def test_the_hook_sees_run_summary_json_already_written(
        self, monkeypatch, tmp_path
    ) -> None:
        from pathlib import Path

        import spectramr.pipelines.training_loop as training_loop_mod
        from spectramr.config.settings import TrainingSettings

        cfg = TrainingSettings.settings_from_dict(self._config_dict(tmp_path))
        env = self._env(cfg)

        seen: dict = {}

        def _spy(config, *, run_dir, logger_):
            # Recorded AT CALL TIME -- checking after the pipeline returns
            # would pass either way, since the file exists by then regardless
            # of the order the two calls ran in.
            seen["run_dir"] = Path(run_dir)
            seen["existed"] = (Path(run_dir) / "run_summary.json").exists()

        monkeypatch.setattr(train, "_maybe_run_reporting", _spy)
        monkeypatch.setattr(
            training_loop_mod,
            "_execute_training_loop",
            lambda *a, **k: {"success": True, "iterations_completed": 1},
        )

        train.run_training_pipeline(cfg, env=env, device="cpu")

        assert seen, "the reporting hook never fired -- the test proves nothing"
        assert seen["existed"] is True, (
            "the reporting hook ran before run_summary.json was written, so "
            "fig_1_16_run_summary_card and fig_1_15_computational_profile "
            "soft-skip at training time"
        )
        # And the emission is real, not just ordered: guard against a future
        # change that satisfies the order by never writing the file at all.
        assert (seen["run_dir"] / "run_summary.json").exists()


class TestOnlyRankZeroWritesTheEndOfRunArtifacts(
    TestRunSummaryIsOnDiskBeforeTheHookDraws
):
    """Every rank used to reach the end-of-run artifact block.

    ``run_dir`` is rank-INVARIANT, so under DDP all N ranks wrote the same
    paths concurrently -- ``report_cases/case_0.npz``, ``per_call_metrics.csv``,
    ``run_summary.json`` -- and then the reporting hook read them back. A
    ``np.savez_compressed`` torn by a second writer stays a structurally valid
    zip whose member fails its CRC, so the observed symptom was
    ``reporting hook: generation failed (Bad CRC-32 for file 'input.npy')`` on
    3 of 4 ranks, pointing at the reader rather than at the concurrent write.

    Inherits the harness above so the planted shape runs through the SAME real
    ``run_training_pipeline`` call, not a re-stubbed approximation of it.
    """

    def _fired(self, monkeypatch, tmp_path, *, rank_zero: bool) -> bool:
        import spectramr.pipelines.training_loop as training_loop_mod
        from spectramr.config.settings import TrainingSettings

        cfg = TrainingSettings.settings_from_dict(self._config_dict(tmp_path))
        env = self._env(cfg)

        calls: list[int] = []
        monkeypatch.setattr(train, "_maybe_run_reporting", lambda *a, **k: calls.append(1))
        monkeypatch.setattr(train, "is_rank_zero", lambda _cfg: rank_zero)
        monkeypatch.setattr(
            training_loop_mod,
            "_execute_training_loop",
            lambda *a, **k: {"success": True, "iterations_completed": 1},
        )
        train.run_training_pipeline(cfg, env=env, device="cpu")
        return bool(calls)

    def test_a_secondary_rank_writes_nothing(self, monkeypatch, tmp_path) -> None:
        """The planted violation (CLAUDE.md #15): rank 1 must not reach the
        writes at all. Before the gate this returned True and three such ranks
        raced rank 0 for every path in the run dir."""
        assert self._fired(monkeypatch, tmp_path, rank_zero=False) is False
        assert not (tmp_path / "run_summary.json").exists()

    def test_rank_zero_still_writes(self, monkeypatch, tmp_path) -> None:
        """The other half: a gate that silences every rank would 'fix' the
        corruption by producing no report at all."""
        assert self._fired(monkeypatch, tmp_path, rank_zero=True) is True
        assert (tmp_path / "run_summary.json").exists()
