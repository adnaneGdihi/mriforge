"""Multiple-comparison correction tests (PR-6 / H4).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-6.
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.statistical_tests import StatisticalTests

# ---------------------------------------------------------------------------
# Bonferroni (existing — regression guard)
# ---------------------------------------------------------------------------


def test_bonferroni_is_conservative() -> None:
    p = [0.01, 0.04, 0.05, 0.5]
    out = StatisticalTests.bonferroni_correction(p, alpha=0.05)
    # 0.01 * 4 = 0.04 < 0.05 → significant
    # 0.04 * 4 = 0.16 → not significant
    assert out["significant"] == [True, False, False, False]


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_is_less_conservative_than_bonferroni() -> None:
    p = [0.01, 0.02, 0.03, 0.5]
    holm = StatisticalTests.holm_bonferroni_correction(p, alpha=0.05)
    bonf = StatisticalTests.bonferroni_correction(p, alpha=0.05)
    # Holm cannot reject fewer than Bonferroni at the same alpha.
    assert sum(holm["significant"]) >= sum(bonf["significant"])


def test_holm_monotone_corrected_p_values() -> None:
    p = [0.01, 0.02, 0.03, 0.04]
    out = StatisticalTests.holm_bonferroni_correction(p, alpha=0.05)
    sorted_p = sorted(out["corrected_p_values"])
    # Sorted corrected p-values must be non-decreasing (monotone).
    assert sorted_p == sorted(sorted_p)


def test_holm_empty_input() -> None:
    out = StatisticalTests.holm_bonferroni_correction([], alpha=0.05)
    assert out == {"corrected_p_values": [], "significant": []}


# ---------------------------------------------------------------------------
# FDR-BH (existing — regression guard)
# ---------------------------------------------------------------------------


def test_fdr_bh_signature() -> None:
    p = [0.001, 0.01, 0.04, 0.5]
    out = StatisticalTests.fdr_correction(p, alpha=0.05)
    assert "corrected_p_values" in out and "significant" in out
    assert len(out["significant"]) == 4


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["bonferroni", "holm_bonferroni", "holm", "fdr", "fdr_bh", "benjamini_hochberg"],
)
def test_correct_p_values_dispatches(method: str) -> None:
    p = [0.01, 0.02, 0.03, 0.04]
    out = StatisticalTests.correct_p_values(p, method=method, alpha=0.05)
    assert "corrected_p_values" in out
    assert len(out["corrected_p_values"]) == 4


def test_correct_p_values_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown correction method"):
        StatisticalTests.correct_p_values([0.05], method="bogus")


# ---------------------------------------------------------------------------
# Empty-input guards (WS-2 fix: m == 0 must not raise)
#
# Before the fix, bonferroni_correction divided alpha by m (== 0) and
# fdr_correction indexed sorted_idx[-1] into an empty array — both raised.
# The guard now mirrors holm_bonferroni_correction's empty-input early return.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bonferroni_empty_input() -> None:
    out = StatisticalTests.bonferroni_correction([], alpha=0.05)
    # Must not raise; corrected_p_values and significant are empty lists.
    assert out["corrected_p_values"] == []
    assert out["significant"] == []
    assert isinstance(out["corrected_p_values"], list)
    assert isinstance(out["significant"], list)


@pytest.mark.unit
def test_fdr_empty_input() -> None:
    out = StatisticalTests.fdr_correction([], alpha=0.05)
    # Must not raise; corrected_p_values and significant are empty lists.
    assert out["corrected_p_values"] == []
    assert out["significant"] == []
    assert isinstance(out["corrected_p_values"], list)
    assert isinstance(out["significant"], list)
