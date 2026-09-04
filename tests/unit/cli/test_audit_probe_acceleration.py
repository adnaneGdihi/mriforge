"""Tier-2 acceleration gate: ``audit --probe`` must run on a non-CPU device.

The Tier-2 probe exists to catch (a) CUDA OOM at the configured batch/patch size
and (b) AMP / GradScaler double-unscale traps. **Neither is reachable on CPU.**
Yet the audit CLI defaulted to ``--device cpu`` and ``smoke_test_vf_configs.sh``
never overrode it — so every Tier-2 probe ever run in this repo was the degraded
kind, certifying "did not crash on CPU" while advertising OOM/AMP coverage. That
is a facade (pitfall #16) at the hardware layer.

``_gate_probe_acceleration`` closes it:

* accelerated probe            → pass (info)
* CPU probe, user dictated     → pass (info), recorded as degraded
* CPU probe, no accelerator    → **error**, probe skipped, audit exits 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.cli.app import _gate_probe_acceleration
from spectramr.infrastructure.validation.config_health_checker import HealthCheckReport

APP_SRC = (
    Path(__file__).resolve().parents[3] / "src" / "spectramr" / "cli" / "app.py"
).read_text()


def _args(**kw) -> argparse.Namespace:
    base = {"probe": True, "device": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _config(device: str | None = None, training_device: str | None = None):
    return SimpleNamespace(
        device=device,
        training=SimpleNamespace(device=training_device),
    )


def _result(report: HealthCheckReport):
    hits = [r for r in report.results if r.check_name == "tier2_probe_accelerated"]
    assert len(hits) == 1, "the gate must emit exactly one verdict"
    return hits[0]


class TestGateFailsWithoutAccelerator:
    def test_no_accelerator_is_an_audit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The requirement: --probe must check the run is accelerated."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delenv("FORCE_CPU", raising=False)

        report = HealthCheckReport()
        decision = _gate_probe_acceleration(_args(), _config(device="cuda"), report)

        assert decision is None, "the probe must be SKIPPED, not run as a facade"
        res = _result(report)
        assert res.passed is False
        assert res.severity == "error"
        assert report.passed is False, "the audit must fail (exit 2)"

    def test_error_message_names_the_escape_hatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delenv("FORCE_CPU", raising=False)

        report = HealthCheckReport()
        _gate_probe_acceleration(_args(), _config(device="auto"), report)
        fix = _result(report).fix_hint or ""
        assert "--device cpu" in fix and "FORCE_CPU" in fix


class TestGatePassesWhenAccelerated:
    def test_cuda_probe_passes_and_is_marked_accelerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        report = HealthCheckReport()
        decision = _gate_probe_acceleration(_args(), _config(device="cuda"), report)

        assert decision is not None
        assert decision.accelerated is True
        assert decision.device == "cuda"
        assert _result(report).passed is True


class TestUserDictatedCpu:
    def test_explicit_cpu_passes_but_is_recorded_as_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--device cpu` is the user dictating it — allowed, but honest.

        Deliberately not a *warning*: warnings must be actionable, and "you
        asked for CPU" is not. It is recorded so triage can tell a real probe
        from a degraded one.
        """
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        report = HealthCheckReport()
        decision = _gate_probe_acceleration(
            _args(device="cpu"), _config(device="cuda"), report
        )

        assert decision is not None
        assert decision.accelerated is False
        assert decision.cpu_opt_in is True
        res = _result(report)
        assert res.passed is True
        assert "DISABLED" in res.message, "the degradation must be stated plainly"
        assert report.passed is True


class TestProbeDeviceResolution:
    def test_cli_device_wins_over_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        report = HealthCheckReport()
        decision = _gate_probe_acceleration(
            _args(device="cuda:1"), _config(device="cuda"), report
        )
        assert decision.device == "cuda:1"
        assert decision.source == "cli"

    def test_defaults_to_the_configs_own_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent --device, probe the device the REAL run will use."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        report = HealthCheckReport()
        decision = _gate_probe_acceleration(
            _args(), _config(device=None, training_device="cuda"), report
        )
        assert decision.device == "cuda"
        assert decision.source == "config.device"

    def test_gate_is_a_noop_without_probe(self) -> None:
        report = HealthCheckReport()
        assert _gate_probe_acceleration(_args(probe=False), _config(), report) is None
        assert report.results == []


class TestCliDefaults:
    def test_audit_device_no_longer_defaults_to_cpu(self) -> None:
        """Regression: the old `choices=["cpu","cuda"], default="cpu"` made every
        cluster probe run on CPU, silently voiding its OOM/AMP coverage."""
        assert (
            'default="cpu",\n        help="Device for the Tier-2 probe' not in APP_SRC
        )
        assert 'choices=["cpu", "cuda"]' not in APP_SRC

    def test_probe_dispatches_use_the_resolved_device(self) -> None:
        """All four paradigm probes must receive the gated device, not args.device."""
        assert "device=probe_device" in APP_SRC
        assert APP_SRC.count("device=probe_device") == 4


class TestBenchmarkIsGated:
    """`benchmark` is a heavy pipeline: CPU timings compared against GPU
    baselines are worse than no numbers at all."""

    def test_benchmark_no_longer_silently_picks_cpu(self) -> None:
        assert (
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
            not in APP_SRC
        ), "benchmark() still resolves its device outside the accelerated-run contract"
        assert 'pipeline="benchmark"' in APP_SRC

    def test_benchmark_exposes_a_device_flag(self) -> None:
        """The CPU escape hatch must be reachable from the CLI (pitfall #15)."""
        from spectramr.cli.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["benchmark", "--device", "cpu"])
        assert args.device == "cpu"
