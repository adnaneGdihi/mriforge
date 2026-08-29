"""Unit tests for :mod:`mriforge.core.metrics.dkw`.

Beyond the closed form, these pin the two properties that motivated the move out
of ``infrastructure/calibration/chd.py`` (#1183):

* the module imports **nothing** from an outward layer, so ``core/`` stays a leaf;
* ``chd`` re-exports the *same object*, so there is one implementation and the
  historical import paths keep working (non-negotiable 17).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from mriforge.core.metrics.dkw import dkw_slack


class TestDkwSlack:
    """Closed form and input validation."""

    def test_matches_closed_form(self) -> None:
        n, delta = 500, 1e-3
        assert dkw_slack(n, delta) == pytest.approx(
            math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        )

    def test_decreases_with_n(self) -> None:
        values = [dkw_slack(n, 0.05) for n in (10, 100, 1_000, 10_000)]
        assert values == sorted(values, reverse=True)

    def test_tightens_as_delta_grows(self) -> None:
        assert dkw_slack(1_000, 0.5) < dkw_slack(1_000, 1e-3)

    @pytest.mark.parametrize("bad_n", [0, -1])
    def test_rejects_non_positive_n(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n must be > 0"):
            dkw_slack(bad_n, 0.05)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_delta_outside_unit_interval(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match=r"delta must lie in \(0, 1\)"):
            dkw_slack(100, bad_delta)


class TestSingleOwner:
    """The move is only a fix if it left exactly one implementation behind."""

    def test_chd_reexports_the_same_object(self) -> None:
        from mriforge.infrastructure.calibration import chd

        assert chd.dkw_slack is dkw_slack

    def test_package_reexport_is_the_same_object(self) -> None:
        from mriforge.infrastructure import calibration

        assert calibration.dkw_slack is dkw_slack

    def test_chd_defines_no_second_implementation(self) -> None:
        """A re-export, not a copy — a redefinition would silently diverge."""
        src = Path(chd_path()).read_text(encoding="utf-8")
        defs = [
            node.name
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.FunctionDef) and node.name == "dkw_slack"
        ]
        assert defs == [], f"chd.py redefines dkw_slack: {defs}"


class TestLayerCleanliness:
    """``core/`` must not reach outward — this module is why #1183 was red."""

    def test_module_imports_no_outward_layer(self) -> None:
        src = Path(dkw_path()).read_text(encoding="utf-8")
        outward = ("mriforge.infrastructure", "mriforge.application", "mriforge.pipelines", "mriforge.cli")
        imported: list[str] = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert [m for m in imported if m.startswith(outward)] == []

    def test_trajectory_metrics_no_longer_reaches_into_infrastructure(self) -> None:
        """The consumer whose function-local import kept the gate red (#1183)."""
        from mriforge.core.metrics import trajectory_metrics

        src = Path(trajectory_metrics.__file__).read_text(encoding="utf-8")
        offenders = [
            f"{node.lineno}:{node.col_offset}"
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("mriforge.infrastructure")
        ]
        assert offenders == [], (
            "core/metrics/trajectory_metrics.py imports mriforge.infrastructure at "
            f"{offenders} — including function-local imports, which the ^-anchored "
            "check_layering.sh grep cannot see."
        )


def dkw_path() -> str:
    from mriforge.core.metrics import dkw

    return dkw.__file__


def chd_path() -> str:
    from mriforge.infrastructure.calibration import chd

    return chd.__file__
