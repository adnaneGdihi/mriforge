"""Paired tests for ``infrastructure/reporting/run_hook.py``.

The hook's *content* is already guarded by source-text pins in
``tests/unit/pipelines/test_train_reporting_hook.py``, and those keep working
through the move because ``inspect.getsource`` resolves an object to its real
definition. What no test covered before is the thing the move actually changed:
that the hook is one shared object rather than two drifting copies, and that its
three config-driven branches behave as documented from a non-training caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.reporting import run_hook


@pytest.fixture
def logger_():
    return logging.getLogger("test_run_hook")


class TestTrainReExportsRatherThanCopies:
    """The failure mode this guards is a silent fork.

    If someone later pastes the body back into ``train.py`` instead of importing
    it, both verbs keep working and nothing fails -- while the training report
    and the inference report quietly diverge, which is the exact defect this
    whole change set exists to remove. Identity is the only cheap way to notice.
    """

    def test_train_hook_is_the_same_object(self):
        from spectramr.pipelines import train

        assert train._maybe_run_reporting is run_hook.maybe_run_reporting
        assert train._run_unconfigured_report is run_hook.run_unconfigured_report
        assert train._UNCONFIGURED_REPORT_TABLES is run_hook.UNCONFIGURED_REPORT_TABLES

    def test_the_hook_does_not_live_in_the_training_pipeline_any_more(self):
        import inspect

        from spectramr.pipelines import train

        src_file = inspect.getsourcefile(train._maybe_run_reporting)
        assert src_file is not None
        assert Path(src_file).name == "run_hook.py", (
            "the hook resolved back into pipelines/, which would put `report` "
            "behind `train` again for infer/predict"
        )


class TestBranchSelection:
    def test_absent_reporting_block_takes_the_tables_only_floor(
        self, tmp_path, logger_, monkeypatch
    ):
        seen = {}
        monkeypatch.setattr(
            run_hook,
            "run_unconfigured_report",
            lambda run_dir, lg: seen.update(run_dir=run_dir),
        )
        run_hook.maybe_run_reporting(
            SimpleNamespace(reporting=None), run_dir=tmp_path, logger_=logger_
        )
        assert seen["run_dir"] == tmp_path

    def test_disabled_reporting_block_also_takes_the_floor(self, tmp_path, logger_, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            run_hook,
            "run_unconfigured_report",
            lambda run_dir, lg: seen.update(run_dir=run_dir),
        )
        run_hook.maybe_run_reporting(
            SimpleNamespace(reporting=SimpleNamespace(enabled=False)),
            run_dir=tmp_path,
            logger_=logger_,
        )
        assert seen["run_dir"] == tmp_path

    def test_enabled_reporting_reaches_generate_report_with_the_run_dir(
        self, tmp_path, logger_, monkeypatch
    ):
        import spectramr.infrastructure.reporting as reporting_pkg

        seen = {}

        def _fake(run_dir, **kwargs):
            seen["run_dir"] = run_dir
            seen["kwargs"] = kwargs
            return {"figures": {}, "tables": {}, "tikz": {}, "out_dir": run_dir}

        monkeypatch.setattr(reporting_pkg, "generate_report", _fake, raising=False)
        cfg = SimpleNamespace(
            reporting=SimpleNamespace(enabled=True, task="default"),
            run=SimpleNamespace(seed=7),
        )
        run_hook.maybe_run_reporting(cfg, run_dir=tmp_path, logger_=logger_)
        assert seen["run_dir"] == tmp_path
        # `run.seed` is the canonical location -- the flat read stamped None.
        assert seen["kwargs"]["seed"] == 7
        # method_name defaults to the run directory's name, not an empty string.
        assert seen["kwargs"]["method_name"] == tmp_path.name


class TestFailurePolicy:
    def _cfg(self, *, fail_on_error: bool):
        return SimpleNamespace(
            reporting=SimpleNamespace(enabled=True, task="default", fail_on_error=fail_on_error),
            run=SimpleNamespace(seed=0),
        )

    def _boom(self, *a, **k):
        raise RuntimeError("plot exploded")

    def test_soft_fails_by_default_so_a_long_run_still_wraps_up(
        self, tmp_path, logger_, monkeypatch
    ):
        import spectramr.infrastructure.reporting as reporting_pkg

        monkeypatch.setattr(reporting_pkg, "generate_report", self._boom, raising=False)
        run_hook.maybe_run_reporting(
            self._cfg(fail_on_error=False), run_dir=tmp_path, logger_=logger_
        )  # must not raise

    def test_fail_on_error_propagates(self, tmp_path, logger_, monkeypatch):
        import spectramr.infrastructure.reporting as reporting_pkg

        monkeypatch.setattr(reporting_pkg, "generate_report", self._boom, raising=False)
        with pytest.raises(RuntimeError, match="plot exploded"):
            run_hook.maybe_run_reporting(
                self._cfg(fail_on_error=True), run_dir=tmp_path, logger_=logger_
            )


class TestSoftFailedReportingIsStampedNotOnlyWarned:
    """#1685 -- ``fail_on_error`` defaults False, so the hook can fail silently.

    On the four-rank cluster run that motivated this, three ranks raced on
    ``case_*.npz`` and the hook died on a ``Bad CRC-32``. The run exited 0 and
    its ``run_summary.json`` looked exactly like a run that had simply not
    configured a report -- absent inferred, not reported. The stamp makes the
    two states distinguishable from the artifact alone.
    """

    @staticmethod
    def _cfg(*, fail_on_error=False):
        return SimpleNamespace(
            reporting=SimpleNamespace(enabled=True, fail_on_error=fail_on_error),
            run=SimpleNamespace(seed=0),
        )

    def test_a_failed_hook_stamps_run_summary_and_keeps_the_run_alive(
        self, tmp_path, logger_, monkeypatch
    ):
        import json

        from spectramr.infrastructure import reporting as reporting_pkg

        (tmp_path / "run_summary.json").write_text(json.dumps({"final_psnr": 21.4}))

        def _boom(*a, **k):
            raise RuntimeError("Bad CRC-32 for file 'prediction.npy'")

        monkeypatch.setattr(reporting_pkg, "generate_report", _boom, raising=False)
        run_hook.maybe_run_reporting(self._cfg(), run_dir=tmp_path, logger_=logger_)

        data = json.loads((tmp_path / "run_summary.json").read_text())
        assert data["reporting"]["status"] == "failed"
        assert data["reporting"]["error_type"] == "RuntimeError"
        assert "Bad CRC-32" in data["reporting"]["error"]
        # Read-modify-write: the keys two registered plotters read must survive.
        assert data["final_psnr"] == 21.4

    def test_the_stamp_leaves_no_temp_file_behind(self, tmp_path, logger_, monkeypatch):
        from spectramr.infrastructure import reporting as reporting_pkg

        monkeypatch.setattr(
            reporting_pkg,
            "generate_report",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
            raising=False,
        )
        run_hook.maybe_run_reporting(self._cfg(), run_dir=tmp_path, logger_=logger_)
        assert not (tmp_path / "run_summary.tmp.json").exists()

    def test_a_non_object_run_summary_is_reported_not_overwritten(
        self, tmp_path, logger_, monkeypatch, caplog
    ):
        """Never destroy an artifact you did not understand."""
        from spectramr.infrastructure import reporting as reporting_pkg

        (tmp_path / "run_summary.json").write_text("[1, 2, 3]")
        monkeypatch.setattr(
            reporting_pkg,
            "generate_report",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
            raising=False,
        )
        with caplog.at_level(logging.WARNING):
            run_hook.maybe_run_reporting(self._cfg(), run_dir=tmp_path, logger_=logger_)

        assert (tmp_path / "run_summary.json").read_text() == "[1, 2, 3]"
        assert any("not an object" in r.getMessage() for r in caplog.records)

    def test_fail_on_error_still_raises_rather_than_stamping(
        self, tmp_path, logger_, monkeypatch
    ):
        """The stamp is for the SOFT path only; the hard path must stay hard."""
        from spectramr.infrastructure import reporting as reporting_pkg

        monkeypatch.setattr(
            reporting_pkg,
            "generate_report",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
            raising=False,
        )
        with pytest.raises(RuntimeError):
            run_hook.maybe_run_reporting(
                self._cfg(fail_on_error=True), run_dir=tmp_path, logger_=logger_
            )
        assert not (tmp_path / "run_summary.json").exists()
