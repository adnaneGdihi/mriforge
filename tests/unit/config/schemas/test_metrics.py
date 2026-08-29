"""``metrics.compute`` -- the list that makes the registry reachable.

The schema exposes 86 ``compute_*`` booleans. 209 metrics are registered. Those
two numbers have never matched, in either direction:

* **145 registered metrics have no flag** (issue #343), so registering a metric
  did not make it selectable and nothing said so.
* **17 flags name a metric that is not registered** (issue #340), so setting one
  measures nothing. ``compute_advanced_metrics`` is the worst of them -- it
  defaults to ``True``, 249 corpus arms set it explicitly, and no code in
  ``src/mriforge`` reads it at all.

A list closes both by construction, because registry membership becomes the
validator.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.metrics import MetricsConfigSchema


class TestTheComputeList:
    def test_it_defaults_to_empty(self) -> None:
        """Empty means "this arm is still on the flags". ~800 corpus arms are,
        so the default must not change what any of them measure."""
        assert MetricsConfigSchema().compute == []

    def test_it_accepts_a_metric_with_no_flag(self) -> None:
        """The point of the list. ``brisque`` is registered and has no
        ``compute_brisque`` flag, so before this it could not be requested."""
        assert "compute_brisque" not in MetricsConfigSchema.model_fields
        cfg = MetricsConfigSchema(compute=["brisque"])
        assert cfg.compute == ["brisque"]

    def test_the_schema_does_not_validate_names_itself(self) -> None:
        """Name validation lives in the Tier-1 audit, not here.

        ``config/`` and ``core/`` are peers at the rightmost layer, so importing
        ``MetricsRegistry`` from a schema module to validate at construction
        would couple them for a check the audit already owns.
        """
        MetricsConfigSchema(compute=["not_a_real_metric"])


class TestTheFlagsStillWork:
    def test_flags_are_untouched(self) -> None:
        cfg = MetricsConfigSchema(compute_psnr=True, compute_ssim=False)
        assert cfg.compute_psnr is True
        assert cfg.compute_ssim is False

    def test_the_flag_count_has_not_grown(self) -> None:
        """The list is the way forward; new metrics get a registry entry, not a
        new boolean. Seeded at today's count so adding an 87th is a decision
        someone has to make deliberately."""
        flags = [
            n for n in MetricsConfigSchema.model_fields if n.startswith("compute_")
        ]
        # `compute` itself is the list, not a flag.
        flags = [n for n in flags if n != "compute"]
        assert len(flags) <= 86, (
            f"{len(flags)} compute_* flags, up from 86. Register the metric and "
            "select it via metrics.compute instead of adding a boolean."
        )


class TestExtractionPrefersTheList:
    """The list must reach the metrics computer, not merely parse.

    A field that parses and is never read is the failure mode this whole effort
    exists to remove -- and the one `latent_losses` shipped with earlier in this
    same branch.
    """

    @staticmethod
    def _extract(cfg: MetricsConfigSchema) -> list[str]:
        from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        return MetricsMixin._extract_metrics_from_config(None, cfg)

    def test_a_declared_list_is_returned_verbatim(self) -> None:
        cfg = MetricsConfigSchema(compute=["brisque", "auroc", "psnr"])
        assert self._extract(cfg) == ["brisque", "auroc", "psnr"]

    def test_the_list_is_not_merged_with_the_flags(self) -> None:
        """Merging would make the list understate what the arm measures, which
        defeats the readability goal: the list has to BE the answer."""
        cfg = MetricsConfigSchema(compute=["psnr"], compute_ssim=True)
        assert self._extract(cfg) == ["psnr"]

    def test_an_empty_list_falls_back_to_the_flags(self) -> None:
        cfg = MetricsConfigSchema(compute_psnr=True, compute_ssim=True)
        extracted = self._extract(cfg)
        assert "psnr" in extracted and "ssim" in extracted


class TestTheRegistryIsBiggerThanTheFlags:
    """Anti-vacuity for the whole exercise: if these ever converge, the list is
    no longer buying reachability and this file's premise is stale."""

    def test_registered_metrics_outnumber_the_flags(self) -> None:
        pytest.importorskip("torch")
        from mriforge.core.metrics import MetricsRegistry

        registered = len(getattr(MetricsRegistry, "_metrics", {}))
        flags = len(
            [
                n
                for n in MetricsConfigSchema.model_fields
                if n.startswith("compute_") and n != "compute"
            ]
        )
        assert registered > flags, (
            f"{registered} registered vs {flags} flags -- if the flags have "
            "caught up, re-measure the reachability gap this list exists to close"
        )


class TestTrainMetricIntervalIsSettable:
    """The training-metric throttle was unreachable from YAML.

    `MetricsMixin._compute_training_metrics` reads `train_metric_interval`
    (`metrics_mixin.py:786`), but it was never declared on the schema — and
    `MetricsConfigSchema` is `extra="forbid"`, so no YAML could set it. The
    throttle was hard-wired to its 100 default.

    Declaring it at that same default is deliberately additive: the point is that
    NO existing arm changes. `metrics.metric_interval`, which 855 arms do set, is
    a different field that nothing reads — wiring THAT would change behaviour for
    418 arms, so it stays inert and documented rather than quietly repurposed.
    """

    def test_it_is_now_settable(self) -> None:
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        assert MetricsConfigSchema(train_metric_interval=250).train_metric_interval == 250

    def test_the_default_is_unchanged(self) -> None:
        """The control: this commit must be a no-op for every existing arm.

        100 is the literal `_get_config_value(..., "train_metric_interval", 100)`
        fallback the mixin has always used.
        """
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        assert MetricsConfigSchema().train_metric_interval == 100

    def test_it_is_validated(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mriforge.config.schemas.metrics import MetricsConfigSchema

        with pytest.raises(ValidationError):
            MetricsConfigSchema(train_metric_interval=0)

    def test_metric_interval_is_still_a_separate_inert_field(self) -> None:
        """Both must exist and be independent — conflating them is the change
        this commit deliberately does NOT make."""
        from mriforge.config.schemas.metrics import MetricsConfigSchema

        cfg = MetricsConfigSchema(metric_interval=5)
        assert cfg.metric_interval == 5
        assert cfg.train_metric_interval == 100

    def test_the_mixin_reads_the_declared_field(self) -> None:
        """Anti-vacuity: a field nothing reads would pass every test above."""
        import inspect

        from mriforge.infrastructure.training.strategies.mixins import metrics_mixin

        src = inspect.getsource(metrics_mixin)
        assert '"train_metric_interval"' in src, (
            "the mixin no longer reads train_metric_interval; this field would "
            "be inert, which is the defect it was added to fix"
        )


# ---------------------------------------------------------------------------
# #340/#660: flags naming a metric that is not registered.
#
# A `compute_*` flag whose metric has no registry entry can never produce a
# column. Selecting it yielded a silently missing column for the whole run --
# indistinguishable from "that metric was never any good on this data" (#173).
# An advertised knob must be wired or must refuse (pitfall #15).
# ---------------------------------------------------------------------------


class TestNoFlagAdvertisesAnUnregisteredMetric:
    @staticmethod
    def _dangling() -> list[str]:
        from mriforge.core.metrics.flag_map import metric_for_flag, schema_compute_flags
        from mriforge.core.metrics.registry import MetricsRegistry

        return sorted(
            f
            for f in schema_compute_flags()
            if not MetricsRegistry.is_registered(metric_for_flag(f))
        )

    def test_only_the_guarded_flag_remains(self):
        """15 zero-corpus flags were deleted; the 16th is guarded instead.

        Resolve through `metric_for_flag` and `is_registered` -- the ALIAS-AWARE
        predicate the runtime uses. `MetricsRegistry._metrics` alone is
        alias-blind and reports metrics as missing that are perfectly
        selectable.
        """
        assert self._dangling() == ["compute_precision_recall"]

    @pytest.mark.parametrize(
        "flag",
        [
            "compute_blur",
            "compute_dvars",
            "compute_gcor",
            "compute_fd",
            "compute_pe_cross_corr",
            "compute_aor",
            "compute_dietrich_snr",
            "compute_iauc",
            "compute_kesa",
            "compute_medicalnet_distance",
            "compute_mscn_var",
            "compute_piesno",
            "compute_rase",
            "compute_sam",
            "compute_sfnr",
        ],
    )
    def test_the_deleted_flags_are_gone(self, flag):
        """All 15 had ZERO corpus declarations, so deletion breaks no arm."""
        assert flag not in MetricsConfigSchema.model_fields

    def test_a_deleted_flag_is_now_refused_outright(self):
        """`extra='forbid'` turns a resurrected spelling into a load error
        rather than a silently ignored key."""
        with pytest.raises(ValidationError):
            MetricsConfigSchema(compute_blur=True)


class TestPrecisionRecallCanOnlyHoldItsDefault:
    """29 arms under experiments/training/ declare it, all False.

    Deleting the field would fail their load (`extra='forbid'`), so it survives
    as a field that cannot be switched on. The arm still loads; an arm that
    tries to enable it is told at STARTUP rather than discovering a missing
    column after a full run.
    """

    def test_false_is_accepted(self):
        assert MetricsConfigSchema(compute_precision_recall=False).compute_precision_recall is False

    def test_the_default_is_false(self):
        assert MetricsConfigSchema().compute_precision_recall is False

    def test_true_raises_and_says_why(self):
        with pytest.raises(ValidationError, match="not registered"):
            MetricsConfigSchema(compute_precision_recall=True)

    def test_the_description_states_it_is_unimplemented(self):
        desc = MetricsConfigSchema.model_fields["compute_precision_recall"].description or ""
        assert "UNIMPLEMENTED" in desc
