"""Tests for the CLI startup notice (:func:`mriforge.cli.app._emit_startup_notice`).

Performance follow-up (2026-06-19): a heavy verb (``train`` / ``audit`` / …) must
import PyTorch + the model registry on first use — tens of seconds cold — during
which the terminal looked frozen ("after the import line it waits for a whole
minute"). ``_emit_startup_notice`` prints one stderr line before dispatch so the
wait is legible. These tests pin the gating: heavy verbs notify, light verbs and
the batch-mode env switches stay silent, and the message never touches stdout
(so ``audit --json | jq`` stays parseable). Importing ``mriforge.cli.app`` is
torch-free, so this module runs anywhere.
"""

from __future__ import annotations

import pytest

from mriforge.cli.app import _HEAVY_STARTUP_COMMANDS, _emit_startup_notice


@pytest.fixture(autouse=True)
def _clear_quiet_env(monkeypatch):
    # Ensure neither suppression flag leaks in from the ambient environment.
    monkeypatch.delenv("MRIFORGE_QUIET", raising=False)
    monkeypatch.delenv("MRIFORGE_SUPPRESS_CLINICAL_WARNING", raising=False)
    # Nor a torchrun rank from a surrounding job (see the rank tests below).
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


@pytest.mark.parametrize("command", sorted(_HEAVY_STARTUP_COMMANDS))
def test_heavy_command_emits_notice_to_stderr(command, capsys):
    emitted = _emit_startup_notice(command)
    assert emitted is True
    captured = capsys.readouterr()
    assert command in captured.err
    assert "PyTorch" in captured.err
    # Never on stdout — `audit --json | jq` piping must stay clean.
    assert captured.out == ""


@pytest.mark.parametrize(
    "command", ["doctor", "campaign", "regulatory", "launch", None, ""]
)
def test_light_or_missing_command_is_silent(command, capsys):
    emitted = _emit_startup_notice(command)
    assert emitted is False
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize(
    "flag", ["MRIFORGE_QUIET", "MRIFORGE_SUPPRESS_CLINICAL_WARNING"]
)
def test_env_flag_suppresses_even_for_heavy_command(flag, monkeypatch, capsys):
    monkeypatch.setenv(flag, "1")
    emitted = _emit_startup_notice("train")
    assert emitted is False
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("rank", ["1", "2", "3"])
def test_secondary_rank_is_silent(rank, monkeypatch, capsys):
    """A 4-GPU launch printed this line four times (2026-08-15).

    This is a bare ``print``, not a log record, so the root-logger clamp in
    ``quiet_secondary_ranks`` cannot reach it — it needs its own gate.
    """
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", rank)
    assert _emit_startup_notice("train-distributed") is False
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_rank_zero_still_notifies(monkeypatch, capsys):
    """One notice per job, not zero — the wait it explains is still real."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    assert _emit_startup_notice("train-distributed") is True
    assert "PyTorch" in capsys.readouterr().err


def test_light_verb_short_circuits_before_the_rank_lookup(monkeypatch, capsys):
    """The rank check must not drag ``mriforge.core`` (torch) onto a light path.

    ``_HEAVY_STARTUP_COMMANDS`` is tested first precisely so a lightweight verb
    returns before the import. Pinned because reordering the gates would be an
    invisible regression in ``--help``-adjacent startup cost.
    """
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    import inspect

    src = inspect.getsource(_emit_startup_notice)
    heavy_check = src.index("_HEAVY_STARTUP_COMMANDS")
    rank_import = src.index("from mriforge.core import env")
    assert heavy_check < rank_import, (
        "the heavy-verb guard must run before the rank lookup imports core.env"
    )
    assert _emit_startup_notice("doctor") is False
    assert capsys.readouterr().err == ""


def test_heavy_set_matches_registered_subcommands():
    # Guard against drift: every name in the heavy set must be a real subcommand
    # the parser registers, so a rename cannot silently make the notice dead.
    from mriforge.cli.app import build_parser

    parser = build_parser()
    subact = next(
        a
        for a in parser._actions
        if getattr(a, "choices", None) and "train" in a.choices
    )
    registered = set(subact.choices)
    missing = _HEAVY_STARTUP_COMMANDS - registered
    assert not missing, (
        f"heavy-startup names not registered as subcommands: {missing}"
    )
