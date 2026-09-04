"""The Tier-1 audit must run the compatibility matrix (cached-cascade WS-X #1).

Proves cli/app.py:_audit_one folds compatibility_matrix messages into the health
report so an error-severity compat rule blocks the audit (exit 2). This is the
wiring that makes the "components agree" guarantee real at `spectramr audit` time.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from spectramr.infrastructure.validation import compatibility_matrix as cm

_YAML = "experiments/inprogress/10_paradigms/exp_01_rectified_flow.yaml"


@pytest.fixture
def _isolated_rules() -> Iterator[None]:
    saved = list(cm._RULES)
    try:
        yield
    finally:
        cm._RULES[:] = saved


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        json=True, probe=False, strict=False, device="cpu",
        noise=False, save_probe_images=None,
    )


@pytest.mark.skipif(not Path(_YAML).exists(), reason="sample inprogress YAML absent")
def test_audit_blocks_on_compat_error(_isolated_rules: None, capsys) -> None:
    from spectramr.cli.app import _audit_one

    cm.register_rule(
        lambda ctx: [
            cm.CompatMessage(
                rule="probe_forced",
                message="forced compat error for wiring test",
                severity="error",
                fix_hint="n/a",
            )
        ]
    )
    rc = _audit_one(_args(), _YAML)
    out = capsys.readouterr().out
    assert rc == 2  # an error-severity compat message must fail the audit
    assert "compat:probe_forced" in out


@pytest.mark.skipif(not Path(_YAML).exists(), reason="sample inprogress YAML absent")
def test_audit_clean_when_no_compat_rules(_isolated_rules: None) -> None:
    from spectramr.cli.app import _audit_one

    cm._RULES[:] = []  # no rules -> compat contributes nothing
    rc = _audit_one(_args(), _YAML)
    assert rc in (0, 1)  # YAML's own state (1 = pre-existing warning), not a compat error
