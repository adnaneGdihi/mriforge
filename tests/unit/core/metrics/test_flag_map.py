"""SSOT tests for the ``metrics.compute_*`` flag -> metric-name mapping.

``core.metrics.flag_map.metric_for_flag`` is the single owner of the flag->name
relationship. Three consumers historically hand-maintained their own dict and drifted
(same flag, different name -> fatal on one path, silent skip on another). These tests:

1. pin the resolver (identity + the one documented alias);
2. assert the per-batch builder map targets only registered metrics; and
3. **ratchet** the two remaining literal consumer maps (the validation-computer name
   list in ``MetricsMixin`` and the expected-name set in ``pipelines.training_loop``)
   to :func:`metric_for_flag`, so the three maps can never disagree again.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from mriforge.core.metrics.flag_map import (
    BUILDER_METRIC_FLAGS,
    FLAG_TO_METRIC,
    NON_METRIC_FLAGS,
    metric_for_flag,
    schema_compute_flags,
    schema_flag_to_metric,
)


def _registered_metrics() -> set[str]:
    from mriforge.core.metrics.registry import list_available

    return set(list_available())


# ─────────────────────────────────────────────────────────────────────────────
# 1. The resolver
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricForFlag:
    def test_identity_is_the_default(self):
        assert metric_for_flag("compute_psnr") == "psnr"
        assert metric_for_flag("compute_ms_ssim") == "ms_ssim"

    def test_alias_overrides_identity(self):
        # The registered metric is 'negative_voxels'; a plain strip gives the
        # unregistered 'neg_voxels'. The alias is the whole reason this near-miss
        # stopped silently never firing.
        assert metric_for_flag("compute_neg_voxels") == "negative_voxels"
        assert metric_for_flag("compute_neg_voxels") != "neg_voxels"

    def test_rejects_a_non_compute_flag(self):
        with pytest.raises(ValueError, match="compute_"):
            metric_for_flag("psnr")


# ─────────────────────────────────────────────────────────────────────────────
# 2. The per-batch builder map targets only registered metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestBuilderMap:
    def test_flag_to_metric_covers_exactly_the_builder_flags(self):
        assert set(FLAG_TO_METRIC) == set(BUILDER_METRIC_FLAGS)

    def test_every_builder_target_is_registered(self):
        """The builder raises ConfigurationError on an unregistered enabled flag, so a
        map value that names no registered metric is a crash landmine (pitfall #15)."""
        registered = _registered_metrics()
        gaps = {name for name in FLAG_TO_METRIC.values() if name not in registered}
        assert (
            not gaps
        ), f"builder FLAG_TO_METRIC targets are not registered: {sorted(gaps)}"

    def test_the_alias_target_is_registered(self):
        assert "negative_voxels" in _registered_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ratchet: neither consumer hand-writes a map at all
# ─────────────────────────────────────────────────────────────────────────────

_NAME_FLAG = re.compile(r'"([\w_]+)":\s*"(compute_\w+)"')


def _training_loop_map() -> dict[str, str]:
    from mriforge.pipelines.training_loop import _CSV_METRIC_NAME_MAP

    return dict(_CSV_METRIC_NAME_MAP)


def _mixin_map() -> dict[str, str]:
    """The flag->name map ``MetricsMixin._extract_metrics_from_config`` selects with."""
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    src = inspect.getsource(MetricsMixin._extract_metrics_from_config)
    assert "schema_flag_to_metric()" in src, (
        "MetricsMixin no longer derives its flag map from the SSOT — it has gone "
        "back to a hand-written literal, which is exactly the #340 regression."
    )
    return schema_flag_to_metric()


class TestConsumerMapsDeriveFromSSOT:
    """Name agreement was only half of #340; *coverage* agreement is the other half.

    The previous version of this class compared two hand-written literals against
    ``metric_for_flag`` and passed — because both literals spelled their names
    correctly. What it could not see is that one had 78 entries and the other 43.
    A flag in the CSV map but not the mixin map produces a ``losses.csv`` column
    header that nothing can ever fill, which reads as "measured, came back empty"
    rather than "never selected". 22 flags were in that state.

    So the guard is no longer "do the literals agree" but "is there a literal at
    all". Both consumers now derive from :func:`schema_flag_to_metric`, which makes
    drift unrepresentable rather than merely detectable.
    """

    def test_both_consumers_are_the_ssot_map(self):
        ssot = schema_flag_to_metric()
        assert ssot, "the SSOT map is empty — schema introspection broke"
        assert _training_loop_map() == ssot
        assert _mixin_map() == ssot

    def test_coverage_is_complete_against_the_schema(self):
        """Every metric-selecting schema flag is reachable from both consumers."""
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        schema_flags = {
            f for f in MetricsConfigSchema.model_fields if f.startswith("compute_")
        }
        uncovered = schema_flags - set(schema_flag_to_metric()) - NON_METRIC_FLAGS
        assert not uncovered, (
            f"schema flags reachable from no consumer: {sorted(uncovered)}. "
            "Add the metric, or record the flag in flag_map.NON_METRIC_FLAGS."
        )

    def test_names_still_resolve_through_the_resolver(self):
        drift = {
            (f, n)
            for f, n in schema_flag_to_metric().items()
            if metric_for_flag(f) != n
        }
        assert not drift, f"derived map disagrees with metric_for_flag: {sorted(drift)}"


class TestNonMetricFlagsAreExcluded:
    """``compute_advanced_metrics`` must never reach a registry lookup.

    It names no metric, and it defaults **True** — so if it were treated as an
    ordinary flag whose name happens not to be registered, the mixin's
    dangling-flag warning would fire on every arm in the corpus. Warnings exit 2
    under ``audit --strict`` (non-negotiable #4), so that would convert a
    behaviour-neutral refactor into a corpus-wide failure.
    """

    def test_advanced_metrics_is_excluded(self):
        assert "compute_advanced_metrics" in NON_METRIC_FLAGS
        assert "compute_advanced_metrics" not in schema_compute_flags()
        assert "compute_advanced_metrics" not in schema_flag_to_metric()

    def test_every_excluded_flag_really_is_a_schema_field(self):
        """An exclusion for a flag that does not exist is dead weight that hides."""
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        stale = NON_METRIC_FLAGS - set(MetricsConfigSchema.model_fields)
        assert not stale, f"NON_METRIC_FLAGS names non-existent fields: {sorted(stale)}"

    def test_every_excluded_flag_names_no_registered_metric(self):
        """Anti-vacuity: excluding a flag whose metric IS registered would silently
        make that metric unselectable — the #340 defect, reintroduced by the fix."""
        registered = _registered_metrics()
        wrongly_excluded = {
            f for f in NON_METRIC_FLAGS if metric_for_flag(f) in registered
        }
        assert not wrongly_excluded, (
            f"these flags name a REGISTERED metric and must not be excluded: "
            f"{sorted(wrongly_excluded)}"
        )

    def test_a_default_true_flag_cannot_warn(self):
        """The corpus-wide guard, stated as a property rather than a single name.

        Any flag defaulting True whose name is unregistered would warn on every
        run. Such a flag belongs in NON_METRIC_FLAGS or must be registered.
        """
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        registered = _registered_metrics()
        offenders = {
            flag
            for flag, name in schema_flag_to_metric().items()
            if MetricsConfigSchema.model_fields[flag].default is True
            and name not in registered
        }
        assert not offenders, (
            f"flags defaulting True that name no registered metric: "
            f"{sorted(offenders)} — each warns on EVERY arm (audit --strict exits 2)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestClaudeMdFlagCensusIsCurrent:
    """CLAUDE.md's metrics-drain section quotes four measured numbers.

    They drifted: the "flags naming an unregistered metric" figure read 17 while
    the live answer was 21. Nothing could have caught that, because the numbers
    were prose with no stated derivation — and the two plausible derivations
    disagree. ``metric_for_flag`` (identity strip + a one-entry alias table) is
    the SSOT and gives 144 unreachable metrics; ``FLAG_TO_METRIC`` is a 36-entry
    per-consumer *coverage* map and gives 173. A number nobody can attribute is
    a number nobody can check.

    So CLAUDE.md now carries the reproducer, and this test runs it and compares
    against the figures in the prose. Update the prose and this passes again;
    let them drift and it fails with both values.
    """

    _CLAUDE_MD = Path(__file__).resolve().parents[4] / "CLAUDE.md"

    @staticmethod
    def _census() -> tuple[int, int, int, int]:
        from mriforge.config.schemas.metrics import MetricsConfigSchema
        from mriforge.core.metrics.flag_map import metric_for_flag
        from mriforge.core.metrics.registry import MetricsRegistry

        flags = [
            f for f in MetricsConfigSchema.model_fields if f.startswith("compute_")
        ]
        reg = set(MetricsRegistry._metrics)
        return (
            len(flags),
            len(reg),
            len(reg - {metric_for_flag(f) for f in flags}),
            len([f for f in flags if metric_for_flag(f) not in reg]),
        )

    def test_the_documented_tuple_matches_the_live_census(self) -> None:
        text = self._CLAUDE_MD.read_text()
        match = re.search(r"# -> \((\d+), (\d+), (\d+), (\d+)\)", text)
        assert match, "CLAUDE.md no longer carries the census reproducer's output"
        documented = tuple(int(g) for g in match.groups())
        assert documented == self._census(), (
            f"CLAUDE.md says {documented}, the registry says {self._census()}. "
            "Re-run the snippet in the 'Drain metrics.compute_*' section."
        )

    def test_the_census_is_alias_blind_and_that_is_recorded(self) -> None:
        """The census counts against ``_metrics``; the runtime asks ``is_registered``.

        ``is_registered`` also consults the 296-entry ALIAS table, so the census
        overstates how many flags dangle. ``compute_fwhm``, ``compute_gsr``,
        ``compute_ndc`` and ``compute_volume_similarity`` each name an alias of a
        canonical metric: unreachable by the census's arithmetic, perfectly
        selectable at runtime.

        This is not a bug in the census — it measures *canonical* coverage, which
        is what the drain-to-``metrics.compute`` migration cares about. It IS a
        trap for anyone who reads those four numbers as "reachability", so the
        gap is pinned here rather than left to be rediscovered. Reachability is
        ``is_registered``; the census is not it.
        """
        from mriforge.config.schemas.metrics import MetricsConfigSchema
        from mriforge.core.metrics.registry import MetricsRegistry

        flags = [
            f for f in MetricsConfigSchema.model_fields if f.startswith("compute_")
        ]
        canonical_only = {
            f for f in flags if metric_for_flag(f) not in set(MetricsRegistry._metrics)
        }
        alias_aware = {
            f for f in flags if not MetricsRegistry.is_registered(metric_for_flag(f))
        }
        assert alias_aware < canonical_only, (
            "the two predicates no longer disagree — if the alias table stopped "
            "covering flag names, this test has lost its subject and the census "
            "may now be quoted as reachability."
        )
        assert canonical_only - alias_aware == {
            "compute_fwhm",
            "compute_gsr",
            "compute_ndc",
            "compute_volume_similarity",
        }, "the alias-only reachable set moved; re-check before quoting the census"

    def test_the_prose_numbers_match_the_tuple(self) -> None:
        """The prose and the reproducer must not disagree with each other either.

        The tuple could be refreshed while the sentence above it keeps the old
        figure — which is the shape of the original drift.
        """
        text = self._CLAUDE_MD.read_text()
        flags, reg, no_flag, unreg = self._census()
        for value, phrase in (
            (flags, f"{flags} `compute_*` booleans"),
            (reg, f"the registry holds {reg} metrics"),
            (no_flag, f"**{no_flag} of those metrics"),
            (unreg, f"**{unreg} flags name a metric"),
        ):
            assert phrase in text, f"CLAUDE.md prose is stale for {value}: {phrase!r}"

    def test_the_two_derivations_really_do_disagree(self) -> None:
        """Anti-vacuity: if they agreed, naming the SSOT would be pointless.

        This is what makes the reproducer worth carrying rather than letting the
        next person re-derive it whichever way they reach for first.
        """
        from mriforge.config.schemas.metrics import MetricsConfigSchema
        from mriforge.core.metrics.flag_map import FLAG_TO_METRIC, metric_for_flag
        from mriforge.core.metrics.registry import MetricsRegistry

        flags = [
            f for f in MetricsConfigSchema.model_fields if f.startswith("compute_")
        ]
        reg = set(MetricsRegistry._metrics)
        via_ssot = len(reg - {metric_for_flag(f) for f in flags})
        via_coverage = len(reg - set(FLAG_TO_METRIC.values()))
        assert via_ssot != via_coverage, (
            "the coverage map and the name resolver now agree; if that is "
            "deliberate, this test and CLAUDE.md's caveat can both be simplified"
        )
