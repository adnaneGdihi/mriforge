"""Planted violations for the ratchet-baseline mechanism (non-negotiable 15).

``_fitness_lib`` is the shared engine behind every ``tests/architecture/``
fitness function, so a defect here is silently multiplied across the whole tier.
Its two known holes both stem from one deliberate decision — :func:`ratchet`
keys on identity alone, stripping the measurement, because keying on the whole
string made every long-standing offender re-report as NEW on any edit (#629):

* a baselined identity whose file was **deleted** stays pre-exempted, and
* an offender that **grows** past its recorded measurement is invisible.

The checks that close them are pure functions over (baselined, current), so
every plant below is a literal pair of sets rather than a tmp tree.
"""

from __future__ import annotations

import pytest

from . import _fitness_lib
from ._fitness_lib import (
    baseline_identity,
    baseline_measurement,
    clamp_to_recorded,
    grown_measurements,
    load_baseline,
    ratchet,
    stale_identities,
    with_measurement,
)

pytestmark = pytest.mark.architecture


class TestBaselineMeasurement:
    """The parser both checks stand on — it must read BOTH recorded forms."""

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ("src/a.py  # 371 loc", 371),  # baseline-file form (write_baseline)
            ("src/a.py (371 loc)", 371),  # scanner form
            ("MyClass (depth 3)", 3),
            ("src/a.py::f::g (2 retries)", 2),
            ("src/a.py::f (5 branches)", 5),
        ],
    )
    def test_reads_a_measurement(self, entry: str, expected: int) -> None:
        assert baseline_measurement(entry) == expected

    @pytest.mark.parametrize(
        "entry",
        [
            "src/a.py::step(('batch', 'epoch'))",  # signature drift: params ARE the identity
            "src/a.py",
            "src/a.py  # justified, see PR #123",
        ],
    )
    def test_entries_without_a_measurement_read_as_none(self, entry: str) -> None:
        """``None`` must not be coerced to 0 — that would report every such
        entry as having grown from 0 to its current size."""
        assert baseline_measurement(entry) is None

    def test_identity_survives_both_forms(self) -> None:
        assert baseline_identity("src/a.py  # 371 loc") == "src/a.py"
        assert baseline_identity("src/a.py (371 loc)") == "src/a.py"


class TestStaleDetector:
    """Plants for :func:`stale_identities`."""

    def test_plant_deleted_file_is_reported(self) -> None:
        """The real shape: 3 directors deleted in 8d01b95c5, entries left behind."""
        baselined = {"src/gone.py  # 400 loc", "src/here.py  # 500 loc"}
        current = {"src/here.py (500 loc)"}
        assert stale_identities(baselined, current) == {"src/gone.py"}

    def test_plant_offender_that_shrank_under_the_ceiling_is_reported(self) -> None:
        """Debt paid is also staleness — the entry must go, or the path stays
        pre-exempted and can regrow to 900 LOC green."""
        assert stale_identities({"src/fixed.py  # 400 loc"}, set()) == {"src/fixed.py"}

    def test_a_still_live_offender_is_not_stale(self) -> None:
        """Negative plant: growth must NOT read as staleness. Identity is
        measurement-independent, which is the #629 fix this must not undo."""
        assert stale_identities({"src/a.py  # 300 loc"}, {"src/a.py (900 loc)"}) == set()

    def test_an_unbaselined_offender_is_not_stale(self) -> None:
        """That is ``ratchet``'s job, not this one — the two must not overlap."""
        assert stale_identities(set(), {"src/new.py (400 loc)"}) == set()

    def test_empty_baseline_is_vacuously_clean(self) -> None:
        assert stale_identities(set(), set()) == set()


class TestGrowthDetector:
    """Plants for :func:`grown_measurements`."""

    def test_plant_growth_is_reported_with_both_numbers(self) -> None:
        """The real shape: physics/sampling.py 3466 -> 4369."""
        got = grown_measurements({"src/a.py  # 3466 loc"}, {"src/a.py (4369 loc)"})
        assert got == {"src/a.py": (3466, 4369)}

    def test_plant_deepened_inheritance_is_reported(self) -> None:
        """Not LOC-specific — any recorded integer is watched."""
        assert grown_measurements({"C (depth 3)"}, {"C (depth 5)"}) == {"C": (3, 5)}

    def test_unchanged_measurement_is_not_growth(self) -> None:
        """Negative plant: ``>``, never ``>=``. A ``>=`` here would report all
        540 entries on every run and the report would be read as noise."""
        assert grown_measurements({"src/a.py  # 371 loc"}, {"src/a.py (371 loc)"}) == {}

    def test_shrinkage_is_not_growth(self) -> None:
        assert grown_measurements({"src/a.py  # 900 loc"}, {"src/a.py (400 loc)"}) == {}

    def test_unbaselined_offender_is_not_growth(self) -> None:
        """A brand-new offender is ``ratchet``'s finding; double-reporting it
        here would make the debt report and the gate disagree on the count."""
        assert grown_measurements(set(), {"src/new.py (400 loc)"}) == {}

    def test_entry_without_a_measurement_is_skipped_not_zeroed(self) -> None:
        """The bug this guards: treating an unparsable measurement as 0 reports
        every signature-drift entry as having grown from nothing."""
        assert grown_measurements({"src/a.py::f(('batch',))"}, {"src/a.py::f(('batch',))"}) == {}
        assert grown_measurements({"src/a.py  # justified"}, {"src/a.py (400 loc)"}) == {}


class TestGrowthSlack:
    """Flat, never proportional (00_MASTER.md §5)."""

    def test_growth_within_slack_is_not_reported(self) -> None:
        assert grown_measurements({"a.py  # 300 loc"}, {"a.py (325 loc)"}, slack=25) == {}

    def test_growth_one_past_slack_is_reported(self) -> None:
        got = grown_measurements({"a.py  # 300 loc"}, {"a.py (326 loc)"}, slack=25)
        assert got == {"a.py": (300, 326)}

    def test_slack_is_flat_not_proportional(self) -> None:
        """The reason the master forbids a percentage: a 10 % rule on a
        12,745-LOC file waves through ~1,275 lines and would have permitted the
        exact +481 growth the row exists to catch."""
        got = grown_measurements(
            {"big.py  # 12745 loc"}, {"big.py (13226 loc)"}, slack=25
        )
        assert got == {"big.py": (12745, 13226)}

    def test_default_slack_is_zero(self) -> None:
        assert grown_measurements({"a.py  # 300 loc"}, {"a.py (301 loc)"}) == {"a.py": (300, 301)}


class TestAntiLaundering:
    """`MRIFORGE_UPDATE_ARCH_BASELINE=1` must not raise a ceiling (NN20)."""

    def test_with_measurement_rewrites_both_forms(self) -> None:
        assert with_measurement("a.py (400 loc)", 300) == "a.py (300 loc)"
        assert with_measurement("a.py  # 400 loc", 300) == "a.py (300 loc)"
        assert with_measurement("C (depth 5)", 3) == "C (depth 3)"

    def test_with_measurement_leaves_measurementless_entries_alone(self) -> None:
        entry = "a.py::f(('batch',))"
        assert with_measurement(entry, 300) == entry

    def test_plant_growth_is_clamped_back_to_the_recorded_value(self) -> None:
        """The laundering shape: re-baselining to record one NEW offender would
        otherwise write down every other file's larger current size."""
        got = clamp_to_recorded({"a.py  # 300 loc"}, {"a.py (900 loc)"})
        assert got == {"a.py (300 loc)"}

    def test_shrinkage_is_recorded_so_the_ratchet_can_tighten(self) -> None:
        assert clamp_to_recorded({"a.py  # 900 loc"}, {"a.py (400 loc)"}) == {"a.py (400 loc)"}

    def test_a_genuinely_new_offender_records_its_real_size(self) -> None:
        """Nothing to clamp against — clamping to 0 would be a fake ceiling."""
        assert clamp_to_recorded(set(), {"new.py (400 loc)"}) == {"new.py (400 loc)"}

    def test_measurementless_entries_pass_through(self) -> None:
        entry = "a.py::f(('batch',))"
        assert clamp_to_recorded({entry}, {entry}) == {entry}

    @staticmethod
    def _isolate(tmp_path, monkeypatch, recorded: str) -> str:
        monkeypatch.setattr(_fitness_lib, "BASELINE_DIR", tmp_path)
        (tmp_path / "t.txt").write_text(recorded + "\n")
        return "t.txt"

    def test_plant_update_mode_does_not_raise_the_ceiling(
        self, tmp_path, monkeypatch
    ) -> None:
        name = self._isolate(tmp_path, monkeypatch, "a.py  # 300 loc")
        monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")
        monkeypatch.delenv("MRIFORGE_RAISE_ARCH_CEILING", raising=False)
        assert ratchet(name, {"a.py (900 loc)"}) == set()
        assert load_baseline(name) == {"a.py  # 300 loc"}

    def test_plant_the_explicit_flag_does_raise_it(self, tmp_path, monkeypatch) -> None:
        """The escape hatch must exist and must be a SECOND, deliberate flag —
        otherwise the only way to record a justified growth is to hand-edit."""
        name = self._isolate(tmp_path, monkeypatch, "a.py  # 300 loc")
        monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")
        monkeypatch.setenv("MRIFORGE_RAISE_ARCH_CEILING", "1")
        assert ratchet(name, {"a.py (900 loc)"}) == set()
        assert load_baseline(name) == {"a.py  # 900 loc"}

    def test_update_mode_still_records_new_offenders(self, tmp_path, monkeypatch) -> None:
        """Clamping must not break the thing update mode is FOR."""
        name = self._isolate(tmp_path, monkeypatch, "a.py  # 300 loc")
        monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")
        monkeypatch.delenv("MRIFORGE_RAISE_ARCH_CEILING", raising=False)
        ratchet(name, {"a.py (900 loc)", "b.py (400 loc)"})
        assert load_baseline(name) == {"a.py  # 300 loc", "b.py  # 400 loc"}

    def test_update_mode_drops_entries_that_are_no_longer_offenders(
        self, tmp_path, monkeypatch
    ) -> None:
        name = self._isolate(tmp_path, monkeypatch, "gone.py  # 300 loc")
        monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")
        ratchet(name, set())
        assert load_baseline(name) == set()
