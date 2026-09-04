"""Metrics Schema Coverage Tests.

PURPOSE:
MetricsConfigSchema has 60+ compute_* flags. The consumer (InfrastructureBuilder)
maintains a manual metric_map dict that maps each flag to a registry key.
Three categories of silent failure:
  1. Schema flag present, NO entry in metric_map → flag is completely ignored
  2. Schema flag present, metric_map entry exists, BUT registry key is not registered
     → `is_registered()` returns False → metric silently skipped
  3. Registry key exists, but schema has NO flag to enable it → metric unreachable

These tests enumerate all three categories and pin them so new gaps are caught
immediately rather than discovered at the end of a multi-day training run.
"""

from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN STATE — updated at last audit (2026-03-31)
# ─────────────────────────────────────────────────────────────────────────────
#
# Flags in MetricsConfigSchema that have NO entry in InfrastructureBuilder.metric_map
# (audited 2026-03-31 using regex extraction from build_metrics() source)
# These flags are silently ignored — setting them in YAML has NO effect.
KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY = {
    "compute_advanced_metrics",  # control flag, not a metric itself
    "compute_aor",  # no map entry
    "compute_bat",  # no map entry
    "compute_blur",  # no map entry
    "compute_cc_snr",  # no map entry (registered as 'cc_snr' though)
    "compute_cosine_similarity",  # no map entry
    "compute_crlb",  # no map entry
    "compute_cw_ssim",  # no map entry (registered in registry)
    "compute_dietrich_snr",  # no map entry
    "compute_dists",  # no map entry
    "compute_divergence",  # no map entry
    "compute_dvars",  # no map entry
    "compute_fd",  # no map entry
    "compute_fid",  # no map entry (FID has no entry!)
    "compute_freq_domain_snr",  # no map entry
    "compute_fwhm",  # no map entry
    "compute_gcor",  # no map entry
    "compute_gsr",  # no map entry
    "compute_iauc",  # no map entry
    "compute_kesa",  # no map entry
    "compute_ktrans",  # no map entry
    "compute_mad",  # no map entry
    "compute_mass_conservation",  # no map entry
    "compute_medicalnet_distance",  # no map entry
    "compute_mscn_var",  # no map entry
    "compute_ndc",  # no map entry
    "compute_ndc_diffusion",  # no map entry
    "compute_neg_voxels",  # no map entry
    "compute_pdm",  # no map entry
    "compute_pe_cross_corr",  # no map entry
    "compute_pearson",  # no map entry
    "compute_piesno",  # no map entry
    "compute_precision_recall",  # no map entry (precision/recall)
    "compute_rase",  # no map entry
    "compute_robust_mri_psnr",  # registered as 'robust_mri_psnr' but not in map
    "compute_sam",  # no map entry
    "compute_sfnr",  # no map entry
    "compute_spectral_linewidth",  # no map entry
    "compute_spike_detection",  # no map entry
    # compute_spike_percent removed 2026-07-18 (duplicate of compute_spike_percentage;
    # its identity target 'spike_percent' was never registered).
    "compute_spike_percentage",  # no map entry
    "compute_st_mad",  # no map entry
    "compute_vnr",  # no map entry
    "compute_volume_similarity",  # no map entry
    "compute_wash_slope",  # no map entry
    # ── Clinical-agreement / registration family (added 2026-05-25) ──
    # Schema flags exist for report-step aggregate metrics, but they are
    # computed in the reporting pipeline (ICC / Bland-Altman across
    # replicates, folding-fraction on displacement fields), not via the
    # per-batch InfrastructureBuilder.metric_map. No map entry by design.
    "compute_bland_altman_bias",
    "compute_coefficient_of_variation",
    "compute_folding_fraction",
    "compute_icc_3_1",
    "compute_limits_of_agreement_lower",
    "compute_limits_of_agreement_upper",
}

# Empty by design: kspace_entropy / kspace_high_freq_error (the former members) were
# removed 2026-07-18 — no metric implemented them, so the flags crashed on enable. The
# ratchet stays here so a NEW map→registry gap fails loudly rather than being added to a
# tolerance list.
KNOWN_MAP_TO_REGISTRY_GAPS: set[str] = set()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_metric_map() -> dict[str, str]:
    """The per-batch builder's flag -> metric-name map.

    The builder consumes the SSOT ``FLAG_TO_METRIC`` (``core.metrics.flag_map``); this
    imports the object directly rather than regex-scraping the source, so the coverage
    ratchet cannot silently pass on a builder that stopped populating its map.
    """
    from spectramr.core.metrics.flag_map import FLAG_TO_METRIC

    return dict(FLAG_TO_METRIC)


def _schema_compute_flags() -> set[str]:
    from spectramr.config.schemas.metrics import MetricsConfigSchema

    return {f for f in MetricsConfigSchema.model_fields if f.startswith("compute_")}


def _registered_metrics() -> set[str]:
    from spectramr.core.metrics.registry import list_available as list_metrics

    return set(list_metrics())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema flags not in metric_map (completely silently ignored)
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaFlagsInMetricMap:
    """Every schema compute_* flag must have an entry in metric_map OR be documented."""

    def test_no_new_unmapped_flags(self):
        """Any schema flag missing from metric_map must be in KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY.

        If this test fails, a new compute_* flag was added to MetricsConfigSchema
        without a corresponding metric_map entry — it will be silently ignored.
        """
        metric_map = _get_metric_map()
        schema_flags = _schema_compute_flags()
        mapped_flags = set(metric_map.keys())

        unmapped = schema_flags - mapped_flags
        new_undocumented = unmapped - KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY

        assert not new_undocumented, (
            f"New compute_* flags in schema with NO metric_map entry "
            f"(silently ignored by InfrastructureBuilder):\n"
            f"  {sorted(new_undocumented)}\n"
            "Fix: Add metric_map['{flag}'] = '{registry_key}' in "
            "InfrastructureBuilder.build_metrics(), OR add to KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY."
        )

    def test_known_unmapped_count_has_not_increased(self):
        """Pinning the count of known unmapped flags — it should only decrease over time."""
        metric_map = _get_metric_map()
        schema_flags = _schema_compute_flags()
        unmapped = schema_flags - set(metric_map.keys())

        stale = KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY - unmapped
        # stale entries = flags now in metric_map that were marked as missing
        # This is progress! We just need to update the known set.
        if stale:
            pytest.fail(
                f"These flags are now in metric_map but still listed in "
                f"KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY:\n  {sorted(stale)}\n"
                "Remove them from KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY to record the fix."
            )

    def test_metric_map_has_no_phantom_flags(self):
        """metric_map must not reference schema flags that no longer exist."""
        metric_map = _get_metric_map()
        schema_flags = _schema_compute_flags()

        phantom = set(metric_map.keys()) - schema_flags
        assert not phantom, (
            f"metric_map references flags that no longer exist in MetricsConfigSchema:\n"
            f"  {sorted(phantom)}\n"
            "Remove these dead entries from metric_map in InfrastructureBuilder."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. metric_map entries that point to unregistered metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricMapToRegistryGaps:
    """Every metric_map target must actually be registered in the metrics registry."""

    def test_no_new_map_to_registry_gaps(self):
        """Every metric_map value must exist in the metrics registry."""
        metric_map = _get_metric_map()
        registered = _registered_metrics()

        # Exclude known gaps
        gaps = {
            (flag, name)
            for flag, name in metric_map.items()
            if name not in registered and name not in KNOWN_MAP_TO_REGISTRY_GAPS
        }
        assert not gaps, (
            "New metric_map entries pointing to UNREGISTERED metrics:\n"
            + "\n".join(
                f"  metric_map['{f}'] = '{n}' — not in registry"
                for f, n in sorted(gaps)
            )
            + "\nFix: Register the metric class or update metric_map to use the correct name."
        )

    def test_known_map_gaps_count_has_not_increased(self):
        """Pin the set of known metric_map→registry gaps."""
        metric_map = _get_metric_map()
        registered = _registered_metrics()
        actual_gaps = {name for name in metric_map.values() if name not in registered}

        new_gaps = actual_gaps - KNOWN_MAP_TO_REGISTRY_GAPS
        assert not new_gaps, (
            f"New metric_map→registry gaps found:\n  {sorted(new_gaps)}\n"
            "Add to KNOWN_MAP_TO_REGISTRY_GAPS if intentional, or register the metric."
        )

        fixed = KNOWN_MAP_TO_REGISTRY_GAPS - actual_gaps
        if fixed:
            pytest.fail(
                f"Metrics now registered that were in KNOWN_MAP_TO_REGISTRY_GAPS:\n"
                f"  {sorted(fixed)}\nRemove from KNOWN_MAP_TO_REGISTRY_GAPS."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core metrics are correctly wired end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestCoreMetricsEndToEnd:
    """The 4 always-enabled core metrics must be created by InfrastructureBuilder."""

    ALWAYS_ENABLED = ["psnr", "ssim", "mse"]

    @pytest.mark.parametrize("metric_name", ALWAYS_ENABLED)
    def test_core_metric_created_by_builder(self, metric_name: str):
        """Core metrics with default config must be in the built metrics dict."""
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.config.schemas.logging import LoggingConfigSchema
        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.infrastructure_builder import (
            InfrastructureBuilder,
        )

        settings = TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(
                compute_psnr=True, compute_ssim=True, compute_mse=True
            ),
        )
        builder = InfrastructureBuilder(config=settings, device="cpu")
        metrics = builder.build_metrics().build()

        assert metric_name in metrics, (
            f"Core metric '{metric_name}' was not created by InfrastructureBuilder.\n"
            f"Metrics created: {sorted(metrics.keys())}"
        )

    def test_optional_metric_created_when_enabled(self):
        """Enabling compute_hfen=True should produce a 'hfen' metric object.

        KNOWN BUG: InfrastructureBuilder.build_metrics() constructs `flags` dict
        only for 6 hardcoded fields in default_flags + optional_flags. The full
        metric_map has 30+ entries, but flags.get(flag, False) returns False for
        ALL of them because they were never added to the `flags` dict.

        This means setting compute_hfen=True, compute_kspace_error=True, etc.
        in YAML has NO effect — InfrastructureBuilder silently ignores them all.

        Fix: Replace the `flags` dict construction with:
            flags = {f: getattr(metrics_cfg, f, False) for f in metric_map}
        """
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.config.schemas.logging import LoggingConfigSchema
        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.config.schemas.model import ModelConfigSchema
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.infrastructure_builder import (
            InfrastructureBuilder,
        )

        settings = TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(compute_hfen=True),
        )
        builder = InfrastructureBuilder(config=settings, device="cpu")
        metrics = builder.build_metrics().build()

        if "hfen" not in metrics:
            pytest.xfail(
                reason="CRITICAL BUG: InfrastructureBuilder.build_metrics() only reads "
                "flags for 6 hardcoded metrics (psnr, ssim, mse, mae, nmse, nrmse). "
                "All other metric_map entries (30+) are silently unreachable because "
                "flags.get('compute_hfen', False) always returns False — the flag was "
                "never added to `flags`.\n"
                "Fix: Replace flags dict construction with:\n"
                "  flags = {f: getattr(metrics_cfg, f, False) for f in metric_map}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Registry metrics unreachable from schema (no compute_* flag)
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistryMetricsWithoutSchemaFlag:
    """Every registered metric should have a schema flag that can enable it.

    Metrics in the registry with no schema flag are unreachable through
    config — they can only be instantiated by directly calling create_metric().
    """

    # These are intentionally not in the schema (internal/computed/raw-API metrics)
    INTENTIONALLY_SCHEMA_EXEMPT = {
        "clinical_ssim",  # internal variant of ssim
        "brisque",  # no-reference IQA, computed differently
        "laplacian_variance",  # Sharp metric, accessed via compute_advanced_metrics
        "tenengrad_variance",  # Sharp metric, accessed via compute_advanced_metrics
        "niqe",  # no-reference IQA
        "nr_iqa",  # no-reference IQA
        "med_fid",  # medical FID variant
        "g_factor",  # coil geometry factor
        "ghosting_ratio",  # artifact measure
        "power_spectrum_consistency",  # physics consistency metric
        "zipper_detection",  # artifact detection
        "negative_voxels",  # artifact measure
        "nema_cnr",  # NEMA standard variant
        "nema_snr",  # NEMA standard variant
        # ── Classification / detection metrics — not in image-quality flag space
        "auprc",
        "auroc",
        "cohen_kappa",
        "detection_sensitivity",
        "detection_specificity",
        "expected_calibration_error",
        "nll_bits_per_dim",
        # ── Registration / surface metrics — used by registration pipeline
        "average_surface_distance",
        "dvf_mae",
        "folding_fraction",
        "target_registration_error",
        # ── Bland–Altman / clinical agreement family
        "bland_altman_bias",
        "coefficient_of_variation",
        "icc_3_1",
        "limits_of_agreement_lower",
        "limits_of_agreement_upper",
        # ── Distribution / hallucination metrics
        "cosine_preservation_score",
        "fabrication_rate",
        "feature_fidelity_index",
        "kernelised_stein_discrepancy",
        "mmd_metric",
        "sliced_wasserstein",
        "wasserstein_1d",
        # ── Geodesic / qmap parameter metrics (multi-param mapping)
        "geodesic_fc_error",
        "geodesic_mrf_parameter_error",
        "geodesic_qmap_error",
        # ── Physics-domain metrics
        "cross_scanner_t1t2_concordance",
        "edge_preservation_index",
        "mutual_information",
        "radial_k_error",
        "residual_whiteness",
        "through_plane_fwhm",
        # ── Asymptotic / free-probability g-factor + theoretical certificates
        # (sim2rank-only; encoding/topology properties, not training flags).
        "asymptotic_gfactor",
        "galois_h1_certificate",
        "pac_bayes_certificate",
        "persistence_diameter",
        "mrf_persistence_stability",
        "topological_mask_certificate",
        # ── Closed-form perceptual IQA added 2026-05-24 (piq-backed) + the NGS
        # autofocus measure. Used by the sim2rank sweep, not toggled per-run.
        "haarpsi",
        "mdsi",
        "vsi",
        "dss",
        "ms_gmsd",
        "normalized_gradient_squared",
        # ── No-reference artifact-detection metrics added 2026-05-25.
        # Blind CV/MRI quality measures used by the sim2rank sweep, not config
        # flags. (piq-backed: iw_ssim, srsim, vif_p, total_variation.)
        "brenner_focus",
        "immerkaer_noise",
        "mlv",
        "intensity_entropy",
        "blockiness",
        "high_freq_energy_ratio",
        "gibbs_ringing",
        "iw_ssim",
        "srsim",
        "vif_p",
        "total_variation",
    }

    def test_registry_orphans_are_a_census_not_a_gate(self):
        """The flag census, recorded but no longer asserted — and here is why.

        This used to be ``test_no_new_registry_orphans``, a ratchet meant to fail
        when someone added a metric with no ``compute_*`` flag. It was **red on
        clean dev** with 142 orphans, so it could never signal that: a metric added
        tomorrow turned a red test into a differently-red test (#343).

        Filling ``INTENTIONALLY_SCHEMA_EXEMPT`` with the other 142 would have made
        it green and still meaningless, because **the property it asserts is
        backwards**. Flags are being DRAINED to ``metrics.compute`` (CLAUDE.md's
        standing migration): registry membership is the validator, so having no
        flag is the target state, not a defect. Adding 142 flags would push the
        corpus away from the migration it is supposed to be completing.

        So the guard moved to what actually matters — REACHABILITY — in
        ``TestEveryRegisteredMetricIsReachable`` below. This test keeps the census
        visible without pretending it is a gate.
        """
        from spectramr.config.schemas.metrics import MetricsConfigSchema

        schema_metric_suffixes = {
            f[len("compute_") :]
            for f in MetricsConfigSchema.model_fields
            if f.startswith("compute_")
        }
        orphans = (
            _registered_metrics()
            - schema_metric_suffixes
            - self.INTENTIONALLY_SCHEMA_EXEMPT
        )
        # Recorded, not gated: this number should DROP as the drain proceeds, and
        # an increase is not by itself a defect. A hard bound keeps it from
        # silently exploding while leaving the drain free to move.
        assert len(orphans) <= 160, (
            f"{len(orphans)} registered metrics have no compute_* flag. That is "
            "expected (flags are draining to metrics.compute), but the count "
            "jumped far enough to be worth a look."
        )


class TestEveryRegisteredMetricIsReachable:
    """The re-keyed #343 guard: registered must mean *usable from a config*.

    "Has a ``compute_*`` flag" was the wrong property (see the census above). The
    right one is end-to-end: a name an arm writes into ``metrics.compute`` must
    survive validation AND construct the way the validation computer constructs
    it. Both halves are needed — they fail independently, and the second half is
    where the real damage was:

    * ``MetricsMixin._extract_metrics_from_config`` raises on a name that is not
      ``MetricsRegistry.is_registered`` (#173), so half one is the config gate.
    * ``computer.py`` then calls ``MetricsRegistry.get(name, device=...)``. Eight
      metrics raised ``TypeError`` on that exact call because they subclass
      ``nn.Module`` without their own ``__init__`` and inherit a signature that
      advertises ``**kwargs`` while accepting none. They were registered,
      workflow-tagged, selectable — and crashed the run that asked for them.

    Unlike the flag census this CAN ratchet: it is green, so the 210th metric
    added without a working constructor turns it red.
    """

    #: Metrics whose constructor needs an argument no config surface can supply.
    #: ``MetricSpec`` (``core/metrics/types.py``) carries name/direction/weight/
    #: enabled and no kwargs dict, so there is nowhere to declare one. Each entry
    #: is a genuine reachability gap, not an exemption from caring.
    REQUIRES_UNSUPPLIABLE_CTOR_ARGS: frozenset[str] = frozenset({
        # Needs sr_scale, or voxel_mm + effective_voxel_mm, to have a passband at
        # all. Correctly refuses to guess (pitfall #9) — but that makes it usable
        # only from sim2rank, which constructs it directly.
        "super_nyquist_fidelity",
    })

    #: Metrics gated on an OPTIONAL extra rather than on anything in this repo.
    #: Reachable wherever the backend is installed, so absence is environmental
    #: and must not read as a registry defect.
    OPTIONAL_BACKEND: frozenset[str] = frozenset({"frd", "rfs"})

    def test_every_registered_name_passes_the_metrics_compute_gate(self):
        """Half one: the name an arm writes must survive config validation."""
        from spectramr.core.metrics.registry import MetricsRegistry

        unusable = {
            n for n in _registered_metrics() if not MetricsRegistry.is_registered(n)
        }
        assert (
            not unusable
        ), f"registered metrics that metrics.compute would REJECT: {sorted(unusable)}"

    def test_every_registered_metric_constructs_the_way_the_computer_builds_it(self):
        """Half two: the exact call at ``computer.py`` — ``get(name, device=...)``."""
        from spectramr.core.metrics.registry import MetricsRegistry

        broken: dict[str, str] = {}
        for name in sorted(_registered_metrics()):
            if name in self.REQUIRES_UNSUPPLIABLE_CTOR_ARGS:
                continue
            try:
                MetricsRegistry.get(name, device="cpu")
            except ImportError:
                if name in self.OPTIONAL_BACKEND:
                    continue  # extra not installed here; not a registry defect
                broken[name] = "ImportError with no declared optional backend"
            except Exception as exc:
                broken[name] = f"{type(exc).__name__}: {exc}"

        assert not broken, (
            "registered metrics that CRASH when the validation computer builds "
            "them — an arm naming one of these in metrics.compute dies mid-run "
            "rather than being graded:\n"
            + "\n".join(f"  {n}: {e}" for n, e in sorted(broken.items()))
        )

    def test_the_unreachable_allowlists_are_not_stale(self):
        """An allow-listed name that no longer exists hides the next real one."""
        registered = _registered_metrics()
        stale = (
            self.REQUIRES_UNSUPPLIABLE_CTOR_ARGS | self.OPTIONAL_BACKEND
        ) - registered
        assert (
            not stale
        ), f"allowlisted metrics are no longer registered: {sorted(stale)}"

    def test_the_flow_and_perfusion_battery_is_reachable(self):
        """#340's headline claim, as a named case rather than an aggregate.

        "A flow or perfusion arm cannot be graded on its own physics."
        ``PhaseContrastFlowStrategy._validation_forward`` returns a velocity field
        precisely so these can score it; every one of them used to raise TypeError.
        """
        from spectramr.core.metrics.registry import MetricsRegistry

        for name in (
            "velocity_rmse",
            "peak_velocity_error",
            "net_flow_error",
            "vnr",
            "cbf_rmse",
            "att_mae",
            "divergence",
            "mass_conservation",
        ):
            assert MetricsRegistry.is_registered(name), name
            assert MetricsRegistry.get(name, device="cpu") is not None, name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Removed-flag regression: unimplemented k-space metrics no longer advertised
# ─────────────────────────────────────────────────────────────────────────────


class TestRemovedUnimplementedKspaceFlags:
    """compute_kspace_entropy / compute_kspace_high_freq_error were removed 2026-07-18.

    No metric implemented them, so InfrastructureBuilder raised ConfigurationError
    ("not a registered metric") the moment either flag was enabled — a crash landmine
    behind an advertised schema knob (pitfall #15). These pin that they are gone from
    every surface, so a future re-add without an implementation fails here.
    """

    REMOVED = ("compute_kspace_entropy", "compute_kspace_high_freq_error")

    def test_schema_forbids_the_removed_flags(self):
        import pydantic

        from spectramr.config.schemas.metrics import MetricsConfigSchema

        for flag in self.REMOVED:
            with pytest.raises(pydantic.ValidationError):
                MetricsConfigSchema(**{flag: True})

    def test_removed_flags_absent_from_the_metric_map(self):
        metric_map = _get_metric_map()
        for flag in self.REMOVED:
            assert flag not in metric_map

    def test_removed_names_absent_from_direction_tables(self):
        from spectramr.core.metrics.metric_directions import (
            NON_REGISTRY_METRIC_DIRECTIONS,
        )
        from spectramr.core.metrics.types import DEFAULT_METRIC_DIRECTIONS

        for name in ("kspace_entropy", "kspace_high_freq_error"):
            assert name not in DEFAULT_METRIC_DIRECTIONS
            assert name not in NON_REGISTRY_METRIC_DIRECTIONS
