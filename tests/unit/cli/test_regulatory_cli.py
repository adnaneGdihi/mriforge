"""Tests for the regulatory CLI (ULF-PR-27 / V.4)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mriforge.cli.regulatory import _cmd_bundle, _cmd_status, _cmd_verify, main


class _NS:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _write_badge(path: Path, *, eligible: bool, gates: dict[str, bool]) -> None:
    payload = {
        "eligible": eligible,
        "gates": {name: {"passed": p} for name, p in gates.items()},
    }
    path.write_text(json.dumps(payload))


def test_status_emits_eligibility(tmp_path, capsys) -> None:
    badge = tmp_path / "validation_badge.json"
    _write_badge(badge, eligible=True, gates={"iqm": True, "downstream": True})
    rc = _cmd_status(_NS(badge_path=str(badge)))
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["eligible"] is True
    assert "iqm" in out["gates"]


def test_status_returns_zero_for_eligible_one_for_not(tmp_path, capsys) -> None:
    badge = tmp_path / "validation_badge.json"
    _write_badge(badge, eligible=False, gates={"iqm": False})
    # status always exits 0 regardless of eligibility — verify is the gate.
    assert _cmd_status(_NS(badge_path=str(badge))) == 0


def test_status_missing_file_returns_2(tmp_path) -> None:
    rc = _cmd_status(_NS(badge_path=str(tmp_path / "does_not_exist.json")))
    assert rc == 2


def test_bundle_collects_artefacts(tmp_path) -> None:
    src = tmp_path / "results"
    src.mkdir()
    _write_badge(src / "validation_badge.json", eligible=True, gates={"iqm": True})
    (src / "calibration_report.json").write_text("{}")
    (src / "stray_other_file.txt").write_text("ignored")
    out_zip = tmp_path / "bundle.zip"
    rc = _cmd_bundle(_NS(output_dir=str(src), bundle_path=str(out_zip)))
    assert rc == 0
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert "validation_badge.json" in names
    assert "calibration_report.json" in names
    assert "stray_other_file.txt" not in names


def test_verify_passes_when_eligible(tmp_path, capsys) -> None:
    src = tmp_path / "results"
    src.mkdir()
    _write_badge(src / "validation_badge.json", eligible=True,
                 gates={"iqm": True, "downstream": True})
    out_zip = tmp_path / "bundle.zip"
    _cmd_bundle(_NS(output_dir=str(src), bundle_path=str(out_zip)))
    rc = _cmd_verify(_NS(bundle_path=str(out_zip)))
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["eligible"] is True
    assert out["n_gates"] == 2


def test_verify_fails_when_gate_failing(tmp_path, capsys) -> None:
    src = tmp_path / "results"
    src.mkdir()
    _write_badge(src / "validation_badge.json", eligible=False,
                 gates={"iqm": True, "downstream": False})
    out_zip = tmp_path / "bundle.zip"
    _cmd_bundle(_NS(output_dir=str(src), bundle_path=str(out_zip)))
    rc = _cmd_verify(_NS(bundle_path=str(out_zip)))
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "downstream" in out["failing_gates"]


def test_main_dispatches_subcommands(tmp_path) -> None:
    badge = tmp_path / "validation_badge.json"
    _write_badge(badge, eligible=True, gates={"iqm": True})
    rc = main(["status", "--badge-path", str(badge)])
    assert rc == 0


def test_regulatory_attached_to_top_level_mriforge_cli(tmp_path, monkeypatch) -> None:
    """The ``regulatory`` subcommand is reachable via the ``mriforge``
    console-script (i.e. ``mriforge.cli.app:main``), not only via
    ``python -m mriforge.cli.regulatory``.

    Regression for the 2026-05-28 registration-audit gap:
    ``regulatory.py`` was imported and unit-tested but never attached
    to ``app.py``'s subparser, so end-users had no way to reach it
    from the documented entry point.
    """
    badge = tmp_path / "validation_badge.json"
    _write_badge(badge, eligible=True, gates={"iqm": True})

    monkeypatch.setattr(
        "sys.argv",
        ["mriforge", "regulatory", "status", "--badge-path", str(badge)],
    )
    from mriforge.cli.app import main as app_main

    rc = app_main()
    assert rc == 0


def test_regulatory_subcommand_appears_in_help(capsys, monkeypatch) -> None:
    """``mriforge --help`` must advertise the ``regulatory`` subcommand."""
    monkeypatch.setattr("sys.argv", ["mriforge", "--help"])
    from mriforge.cli.app import main as app_main

    with pytest.raises(SystemExit) as exc:
        app_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "regulatory" in out
