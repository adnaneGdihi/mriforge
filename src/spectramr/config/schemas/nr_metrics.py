"""Frozen config sub-block for the no-reference (NR) metric battery.

Composed into :class:`~spectramr.config.schemas.metrics.MetricsConfigSchema` as
the ``nr`` field. Holds three things:

1. **which** NR metrics to compute at validation time (``enabled_metrics`` —
   a list of registry keys, the knob read by
   ``MetricsMixin._extract_metrics_from_config`` and the §6 audit check).
2. **tunables** for the metrics that need them (PSDC β reference, GRI target,
   …) — surfaced to each metric through ``MetricContext.nr_params``.
3. **learned-asset gating** (``enable_prior_geometry`` + checkpoint paths) for
   the heavy prior-geometry / cross-contrast metrics.

Every knob here is *read* somewhere (pitfall #15): ``enabled_metrics`` by the
mixin + audit, the tunables by the metrics via ``as_nr_params()``, and the
gating/ckpt fields by the audit check ``learned_metric_checkpoint_resolves``
and the validation-context assembler.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class NRMetricConfig(BaseModel):
    """No-reference metric battery configuration (spec §4).

    RESEARCH MODE — the NR battery is *validation-pending* and runs ONLY in the
    offline meta-evaluation harness (``spectramr meta-evaluate --nr-battery``),
    never during training. ``enabled_metrics`` here selects the offline battery
    to validate; it is NOT consumed by the training-validation metric loop. The
    ``nr_metrics_research_mode`` audit check warns if a training config lists
    them. See docs/no_reference_metrics.rst.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- research-mode marker (provenance stamp) ---------------------------
    research_mode: bool = Field(
        default=True,
        description=(
            "True while the NR battery is validation-pending. Research-mode "
            "metrics run ONLY via the offline meta-evaluate harness and are "
            "never computed during training. Flip to False only after the §8 "
            "concordance/construct-validity tiers certify the battery."
        ),
    )

    # --- which NR metrics to validate offline (registry keys) --------------
    enabled_metrics: tuple[str, ...] = Field(
        default=(),
        description=(
            "Registry keys of NR metrics for the OFFLINE meta-evaluate sweep "
            "(e.g. ('ndcr', 'hfrw', 'gri')). NOT run during training. Empty ⇒ "
            "use the use-case/CLI default battery. Validated by the "
            "nr_metrics_research_mode audit check (typo + research-mode warning)."
        ),
    )

    # --- numeric tunables ---------------------------------------------------
    psdc_beta_ref: float = Field(default=2.0, description="PSDC target radial-PSD log-log slope.")
    psdc_lambda: float = Field(default=0.5, description="PSDC noise-floor penalty weight.")
    are_use_protocol_R: bool = Field(  # noqa: N815 — R is the MRI acceleration symbol
        default=True, description="ARE: fix Δ_R from the protocol acceleration R."
    )
    ccsa_lambda: float = Field(default=0.5, description="CCSA MIND-vs-NMI mixing weight.")
    pedu_n_samples: int = Field(default=8, description="PEDU stochastic-sample count.")
    savs_white_slope: float = Field(default=-2.0, description="SAVS reference white-noise slope.")
    gri_target: float = Field(default=0.09, description="GRI Gibbs-constant target.")
    bpci_max_freq: int = Field(default=16, description="BPCI low-frequency band half-width.")
    sher_kernel_size: int = Field(default=5, description="SHER block-Hankel kernel size.")
    ssel_bandwidth: float = Field(default=0.25, description="SSEL Slepian time-half-bandwidth W.")
    ssel_n_tapers: int = Field(default=8, description="SSEL leading Slepian taper count.")

    # --- learned-asset gating + checkpoints --------------------------------
    enable_prior_geometry: bool = Field(
        default=False,
        description=(
            "Gate the GPU-heavy prior-geometry metrics (PMMD/DPST/FITC/KSDP/"
            "PFLD/HALLUCINATION_INDEX). When False they are skipped even if "
            "listed in enabled_metrics."
        ),
    )
    prior_ckpt: str | None = Field(
        default=None, description="Diffusion-prior checkpoint (DPST/FITC/KSDP/PFLD/HI)."
    )
    prior_model_name: str = Field(
        default="score_based_diffusion",
        description="Registry name of the diffusion-prior model to build for the ckpt.",
    )
    pmmd_encoder_ckpt: str | None = Field(
        default=None, description="SSL encoder checkpoint (PMMD/NCPD)."
    )
    encoder_model_name: str = Field(
        default="mae_mri", description="Registry name of the SSL encoder model."
    )
    ccpr_predictor_ckpt: str | None = Field(
        default=None, description="Cross-contrast predictor checkpoint (CCPR)."
    )
    ncpd_template: str | None = Field(
        default=None, description="Normative (μ_R, Σ_R) template asset path (NCPD)."
    )

    # --- prior-geometry metric keys (the gated subset) ---------------------
    PRIOR_GEOMETRY_KEYS: ClassVar[tuple[str, ...]] = (
        "pmmd",
        "dpst",
        "fitc",
        "ksdp",
        "pfld",
        "hallucination_index",
        "ncpd",
        "ccpr",
    )

    def as_nr_params(self) -> dict[str, Any]:
        """Flat dict of tunables for ``MetricContext.nr_params``.

        Excludes asset paths / gating (those are resolved into concrete
        objects by the context assembler, not passed through as raw strings).
        """
        return {
            "psdc_beta_ref": self.psdc_beta_ref,
            "psdc_lambda": self.psdc_lambda,
            "are_use_protocol_R": self.are_use_protocol_R,
            "ccsa_lambda": self.ccsa_lambda,
            "pedu_n_samples": self.pedu_n_samples,
            "savs_white_slope": self.savs_white_slope,
            "gri_target": self.gri_target,
            "bpci_max_freq": self.bpci_max_freq,
            "sher_kernel_size": self.sher_kernel_size,
            "ssel_bandwidth": self.ssel_bandwidth,
            "ssel_n_tapers": self.ssel_n_tapers,
        }

    def effective_enabled(self) -> tuple[str, ...]:
        """``enabled_metrics`` with the prior-geometry subset dropped when
        ``enable_prior_geometry`` is False (so a gated metric never silently
        runs without its asset gate)."""
        if self.enable_prior_geometry:
            return self.enabled_metrics
        gated = set(self.PRIOR_GEOMETRY_KEYS)
        return tuple(m for m in self.enabled_metrics if m not in gated)


__all__ = ["NRMetricConfig"]
