"""Regression: SLURM ``CANCELLED by <uid>`` was treated as still-pending.

2026-06-11 infrastructure audit. ``sacct`` reports cancelled jobs as the literal
``"CANCELLED by 12345"`` (and may append ``+``). The old exact-match mapping fell
through to the ``"submitted"`` default and ``is_terminal`` was False, so a
cancelled job was re-queried forever and ``all_terminal`` never went True.
"""

from __future__ import annotations

from mriforge.infrastructure.orchestration.slurm_backend import JobStatus


def _status(state):
    return JobStatus(job_id=1, state=state)


def test_cancelled_by_uid_is_terminal_and_cancelled():
    st = _status("CANCELLED by 12345")
    assert st.is_terminal is True
    assert st.normalised_state == "cancelled"


def test_cancelled_plus_suffix():
    st = _status("CANCELLED+")
    assert st.is_terminal is True
    assert st.normalised_state == "cancelled"


def test_plain_terminal_states():
    assert _status("COMPLETED").normalised_state == "completed"
    assert _status("OUT_OF_MEMORY").is_terminal is True
    assert _status("DEADLINE").is_terminal is True


def test_unknown_state_keeps_polling_default():
    # A genuinely-unknown (possibly transient) state keeps the historical
    # "submitted" default rather than being dropped — and crucially is not
    # terminal, so it is re-queried rather than counted as done.
    st = _status("WEIRD_NEW_STATE")
    assert st.normalised_state == "submitted"
    assert st.is_terminal is False


def test_running_and_pending_unchanged():
    assert _status("RUNNING").normalised_state == "running"
    assert _status("PENDING").normalised_state == "submitted"
    assert _status("RUNNING").is_terminal is False
