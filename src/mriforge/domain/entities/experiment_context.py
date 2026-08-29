"""Resolved-semantic IR for a whole experiment -- the *what* of a YAML.

``ResolvedExperimentContext`` is the frozen intermediate representation the
cached-cascade campaign (WS-X) introduces at the config->infrastructure seam.
A single :func:`mriforge.infrastructure.validation.context_resolver.resolve`
call turns the frozen ``TrainingSettings`` into one of these by looking up each
chosen component's *contract* from its registry and reconciling top-level<->nested
config. Everything downstream then consumes this IR instead of re-deriving from
raw config:

* compatibility rules are **pure functions over the context**
  (``check(ctx) -> messages``) -- no registry/config digging inside a rule;
* builders take ``ctx`` (the *what*) alongside ``BuilderContext`` (the *with-what*);
* the spec card becomes ``ctx.render()``;
* the synthetic forward-probe seeds from ``ctx``.

The capability *contracts* themselves (``ModelCapabilities``, ``LossCapabilities``,
``StrategyCapabilities``, ``AdapterCapabilities``) live in
:mod:`mriforge.models.capabilities` and are reused verbatim here so there is **one**
capability SSOT -- this IR aggregates resolved instances of them into per-axis
*profiles*, adding only the two axes capabilities.py does not cover
(:class:`DataProfile`, :class:`PhysicsProfile`).

Layering note: ``domain/`` and ``models/`` sit on the same tier (both inward of
``infrastructure/``), so importing the capability dataclasses from
``models.capabilities`` is rightward/same-tier and does not violate the
inward-only rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mriforge.models.capabilities import (
    AdapterCapabilities,
    Domain,
    LossCapabilities,
    ModelCapabilities,
    StrategyCapabilities,
)

__all__ = [
    "DataProfile",
    "PhysicsProfile",
    "ResolvedExperimentContext",
]


@dataclass(frozen=True)
class DataProfile:
    """Resolved facts about the data the experiment consumes.

    All fields are optional ("unannotated" when ``None``), mirroring the
    capability-dataclass convention so unresolved facts skip their checks
    rather than raising.
    """

    dataset_type: str | None = None
    domain: Domain | None = None
    spatial_dims: tuple[int, ...] | None = None
    paired: bool | None = None
    multi_contrast: bool | None = None
    #: e.g. "magnitude" | "real_imag_interleaved" | "complex"
    channel_layout: str | None = None
    #: acquisition metadata fields exposed to the model (TE/TR/b-value/...)
    acquisition_params_exposed: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicsProfile:
    """Resolved forward-model / acquisition-physics facts."""

    #: the domain the physics operates in ("kspace" | "image")
    operates_in: Domain | None = None
    #: e.g. "sense" | "single_coil_fft" | None
    forward_model: str | None = None
    coil_estimation: str | None = None
    data_consistency: str | None = None
    sampling: str | None = None


@dataclass(frozen=True)
class ResolvedExperimentContext:
    """The frozen, resolved-semantic picture of one experiment YAML.

    Produced once by the context resolver and threaded through reconciliation,
    compatibility rules, builders, the spec card, and the forward probe. Frozen
    so it cannot drift after resolution (the same SSOT discipline as
    ``TrainingSettings``).
    """

    # --- provenance (the chosen component names) -----------------------------
    config_version: str
    model_type: str
    strategy_name: str
    loss_names: tuple[str, ...] = ()

    # --- resolved capability profiles ---------------------------------------
    data_profile: DataProfile = field(default_factory=DataProfile)
    model_profile: ModelCapabilities = field(default_factory=ModelCapabilities)
    strategy_profile: StrategyCapabilities = field(default_factory=StrategyCapabilities)
    loss_profile: tuple[LossCapabilities, ...] = ()
    physics_profile: PhysicsProfile = field(default_factory=PhysicsProfile)
    #: declared adapters bridging domain/shape mismatches, in chain order
    adapter_chain: tuple[AdapterCapabilities, ...] = ()
    #: dotted ``strategy_profile.required_config_fields`` the resolver found
    #: unset (``None``) in the config — populated by the resolver so the
    #: ``required_config_fields`` rule stays a pure function over the IR
    #: (catches the ``training.diffusion.num_timesteps``-class bug).
    unresolved_required_fields: tuple[str, ...] = ()

    def render(self) -> str:
        """Human-readable spec-card rendering of the resolved context.

        This is the single source for the experiment spec card -- rules,
        builders and the probe consume the typed fields, while reviewers read
        this text.
        """
        lines: list[str] = [
            f"ResolvedExperimentContext (config v{self.config_version})",
            f"  model     : {self.model_type}",
            f"  strategy  : {self.strategy_name}",
            f"  losses    : {', '.join(self.loss_names) or '(none)'}",
            "  data      : "
            + f"domain={self.data_profile.domain} "
            + f"dims={self.data_profile.spatial_dims} "
            + f"paired={self.data_profile.paired} "
            + f"multi_contrast={self.data_profile.multi_contrast} "
            + f"channels={self.data_profile.channel_layout}",
            "  model.io  : "
            + f"in={self.model_profile.input_domain} "
            + f"out={self.model_profile.output_domain} "
            + f"complex={self.model_profile.accepts_complex}",
            "  physics   : "
            + f"in={self.physics_profile.operates_in} "
            + f"fwd={self.physics_profile.forward_model} "
            + f"dc={self.physics_profile.data_consistency} "
            + f"coil={self.physics_profile.coil_estimation}",
        ]
        if self.strategy_profile.supported_paradigms:
            lines.append(
                "  paradigms : " + ", ".join(sorted(self.strategy_profile.supported_paradigms))
            )
        if self.adapter_chain:
            lines.append(f"  adapters  : {len(self.adapter_chain)} in chain")
        return "\n".join(lines)
