"""Optimization configuration schema.

44 flat scalars in one class described seven unrelated jobs, so a reader could
not tell where the optimizer stopped and the memory monitor began. Phase 8 of the
config-readability work groups them into named sub-blocks:

.. code-block:: yaml

    optimization:
      optimizer: {type, learning_rate, weight_decay, betas, eps, ...}
      gradient:  {accumulation_steps, enable_checkpointing, clip: {...}}
      precision: {enabled, dtype}
      compile:   {enabled, backend, mode, ...}
      memory:    {enable_monitoring, cleanup_interval, safety_margin, ...}

Grouping is not only cosmetic: it forces one home per knob. ``use_amp`` and
``amp_dtype`` become two fields of one ``precision`` decision, and the two
spellings of gradient checkpointing collapse to ``gradient.enable_checkpointing``.

**The scheduler group is still flat, but the blocker is gone (#662, fixed
2026-08-08).** The plan called for a sixth ``scheduler:`` sub-block folding
``lr_scheduler_strategy``, ``scheduler_type``, ``T_max``, ``eta_min``,
``warmup_steps`` and ``lr_scheduler_kwargs`` into the existing ``scheduler``
dict. That fold could not land while ``resolve_scheduler_spec`` returned
``None`` — no scheduler at all — whenever ``optimization.scheduler`` was absent,
*regardless* of ``lr_scheduler_strategy``: 531 arms declared a strategy with no
``scheduler:`` dict and trained at a constant LR, so creating the dict would
have silently started annealing all of them.

The resolver now honours an **explicitly declared** family with no dict, and
still returns ``None`` for a *defaulted* one (this field carries
``default="cosine"``, and 305 arms declare neither). Declaring a strategy and
folding it therefore mean the same thing, so the fold is safe to attempt — the
sub-block itself remains future work.

Migration posture
-----------------

826 loadable arms declare ``optimization.learning_rate``, so the flat spellings
are retired as ``fold`` records (see ``renames.py``): the ``mode="before"``
validator moves the value into its sub-block, so unmigrated YAML keeps loading
while Python reads one path only. ``config.optimization.learning_rate`` raises
``AttributeError`` — there is no forwarding property.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import OPTIMIZER_ALIASES, OPTIMIZER_NAMES
from .renames import (
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
)

__all__ = [
    "CompileConfigSchema",
    "GradientClipConfigSchema",
    "GradientConfigSchema",
    "LookaheadConfigSchema",
    "MemoryConfigSchema",
    "OptimizationConfigSchema",
    "OptimizerConfigSchema",
    "ParamGroupOverrideSchema",
    "PrecisionConfigSchema",
]

#: ``torch.compile`` modes. Closed because an unknown mode used to be accepted at
#: load time, raise inside ``torch.compile``, and then be swallowed by
#: ``ModelBuilder.compile``'s blanket ``except`` into an eager-mode run that
#: reported success (#619 F1+F2).
CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

#: Shared by every sub-block below. ``forbid`` because a new block has no legacy
#: corpus to protect: a typo inside ``precision:`` has never worked, so there is
#: nothing to break by rejecting it.
_SUBBLOCK = {"extra": "forbid", "frozen": True}


class ParamGroupOverrideSchema(BaseModel):
    """Per-parameter-group optimizer overrides, keyed by ``named_parameters()`` prefix.

    A parameter joins group ``k`` iff ``name == k or name.startswith(k + ".")``.
    The resolver RAISES when a key matches zero parameters — a renamed submodule
    must not silently degrade the arm to a uniform learning rate — and when one
    parameter matches two keys.

    ``learning_rate`` and ``lr_multiplier`` are two homes for one number;
    declaring both raises.
    """

    model_config = {"extra": "forbid", "frozen": True}

    learning_rate: float | None = Field(default=None, gt=0)
    lr_multiplier: float | None = Field(default=None, gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    freeze: bool = Field(
        default=False,
        description="Set requires_grad=False and exclude the group entirely. Not "
        "the same as lr=0: a zero-LR group still accrues optimizer state and is "
        "still decoupled-weight-decayed by AdamW.",
    )

    @model_validator(mode="after")
    def _one_lr_home(self) -> "ParamGroupOverrideSchema":
        if self.learning_rate is not None and self.lr_multiplier is not None:
            raise ValueError(
                "declare either learning_rate or lr_multiplier, not both "
                f"(got learning_rate={self.learning_rate}, "
                f"lr_multiplier={self.lr_multiplier})"
            )
        return self


class LookaheadConfigSchema(BaseModel):
    """Lookahead (Zhang et al. 2019) wrapper around the resolved base optimizer.

    A wrapper, not an ``optimizer.type`` value — a Lookahead with no inner
    optimizer is meaningless, so it cannot be a registry name. Applied last, after
    the base optimizer and its param groups are built.
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = Field(default=False)
    #: Named ``sync_period`` rather than the paper's ``k``: the repo's
    #: unconsumed-knob scanner matches bare identifiers of 3+ characters
    #: (``[a-z_][a-z0-9_]{2,}``), so a field called ``k`` can never be shown to
    #: have a consumer no matter how it is wired — it would sit in the ratchet
    #: forever as a permanent false positive.
    sync_period: int = Field(
        default=5,
        ge=1,
        description="Slow-weight sync period (the paper's k / la_steps).",
    )
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Slow-weight step size (la_alpha)."
    )


# NO SAMConfigSchema here, deliberately.
#
# It existed briefly as a mounted, declarable block with THREE knobs and ZERO
# readers: `SharpnessAwareStepper` -- the consumer its own docstring named -- was
# never written, so `optimization.sam.rho: 0.1` validated, stamped into
# resolved_config.json, and changed nothing. That is pitfall #15/#16 in its
# purest form, and it was worse than the usual case because the two sibling
# blocks added alongside it (`compile`, `zenflow`) both carry a
# `_knobs_require_enabled` validator that RAISES when knobs are declared against
# a disabled mechanism, while this one had neither that guard nor an `enabled`
# field to hang it on.
#
# SAM cannot ride the existing seam: `AMPPolicy.backward_and_step` receives an
# already-computed loss and does exactly one `.backward()`, whereas SAM needs the
# closure for a second forward+backward. It returns with
# `optimizers/sharpness_aware_stepper.py` -- which must also snapshot/restore RNG
# around the pair, restore weights in a `finally`, and raise under gradient
# accumulation -- and not before. See docs/optimizer_ssot.rst.


class OptimizerConfigSchema(BaseModel):
    """Which optimizer, at what learning rate, with which hyper-parameters.

    Every key here is read by the SSOT resolver
    ``mriforge.infrastructure.training.optimizer_resolution.resolve_optimizer_spec``,
    which both OptimizationBuilder and the leaf OptimizerBuilder call. The
    resolver forwards only what the resolved optimizer's signature accepts, and
    RAISES when a knob you *explicitly declared* cannot be consumed (a knob left
    at its schema default is dropped quietly, since you never asked for it).

    Four of these docstrings used to name ``OptimizerSetupMixin._build_optimizer``
    as their consumer. That class has never existed in this repo, and ``betas``
    was in fact read by nothing at all while 27 experiment YAMLs set it.
    """

    model_config = {"protected_namespaces": (), **_SUBBLOCK}

    type: str = Field(
        default="adamw",
        description="Optimizer name. Closed vocabulary — see OptimizerType in "
        "config/schemas/enums.py. Normalised through OPTIMIZER_ALIASES; an "
        "unknown name raises at load time rather than at build time.",
    )
    learning_rate: float = Field(default=1e-5, gt=0)
    generator_learning_rate: float | None = Field(default=None, gt=0)
    discriminator_learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)

    # beta1/beta2 and betas are three spellings of two numbers. They are NOT
    # collapsed here: the ambiguity is already closed by a raise (betas wins, and
    # declaring both with different values is an error), and 579 arms set beta1
    # while 484 set beta2 -- collapsing them is a value transform over ~600 files,
    # not a key move, so it needs its own migrator record and its own PR.
    beta1: float = Field(default=0.5, ge=0, le=1)
    beta2: float = Field(default=0.999, ge=0, le=1)
    betas: tuple[float, float] | None = Field(
        default=None,
        description="Adam-family beta pair. Takes precedence over the scalar "
        "beta1/beta2 fields; declaring both with DIFFERENT values raises. Read by "
        "resolve_optimizer_spec and forwarded to any optimizer whose signature "
        "accepts `betas` (adam, adamw, adamax, nadam, radam, lamb, lion).",
    )
    eps: float = Field(
        default=1e-8,
        gt=0,
        description="Numerical-stability epsilon. Read by resolve_optimizer_spec; "
        "forwarded to optimizers whose signature accepts `eps`.",
    )
    momentum: float = Field(
        default=0.9,
        ge=0,
        description="Momentum factor. Read by resolve_optimizer_spec; forwarded to "
        "optimizers whose signature accepts `momentum` (sgd, rmsprop, lars).",
    )
    nesterov: bool | None = Field(
        default=None,
        description="Nesterov momentum (SGD/LARS). Previously reachable only "
        "through the kwargs escape hatch.",
    )
    amsgrad: bool | None = Field(
        default=None,
        description="AMSGrad variant (Adam/AdamW). Previously reachable only "
        "through the kwargs escape hatch.",
    )
    fused: bool | None = Field(
        default=None,
        description="Fused CUDA optimizer step (Adam/AdamW/SGD and others that "
        "expose it). Collapses the per-parameter elementwise update into a "
        "single multi-tensor kernel, which removes a launch-bound cost that "
        "grows with parameter-tensor COUNT rather than size — so it helps most "
        "on the deep U-Nets and cascades here, which have many small tensors. "
        "Requires CUDA params. None means 'unset' (dropped as a default); "
        "declaring it on an optimizer that has no such argument RAISES rather "
        "than silently training un-fused, because it is a throughput claim.",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch for optimizer parameters with no typed field. "
        "Validated against the resolved optimizer's signature — an unaccepted key "
        "raises. A key that duplicates a typed field is allowed only when the two "
        "values agree.",
    )
    param_groups: dict[str, ParamGroupOverrideSchema] | None = Field(
        default=None,
        description="Per-parameter-group overrides, keyed by a named_parameters() "
        "prefix. Read by resolve_optimizer_spec. A key matching zero parameters "
        "raises; so does declaring this against a model that exposes "
        "get_differential_lr_param_groups (two SSOTs for one decision).",
    )
    lookahead: LookaheadConfigSchema = Field(default_factory=LookaheadConfigSchema)

    @field_validator("betas")
    @classmethod
    def _validate_betas(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Validate the optional Adam beta pair (each in [0, 1])."""
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("betas must be a 2-tuple (beta1, beta2)")
        for b in value:
            if not (0.0 <= b <= 1.0):
                raise ValueError(f"each beta must be in [0, 1], got {b}")
        return value

    @field_validator("type")
    @classmethod
    def _canonical_optimizer_type(cls, value: str) -> str:
        """Normalise and close the optimizer vocabulary at load time.

        Deliberately a validator returning ``str`` rather than an
        ``OptimizerType`` annotation: three live sites consume this field as a
        bare string, and ``str()`` on a ``(str, Enum)`` member yields
        ``'OptimizerType.ADAMW'``, not ``'adamw'``.
        """
        key = value.strip().lower()
        key = OPTIMIZER_ALIASES.get(key, key)
        if key not in OPTIMIZER_NAMES:
            raise ValueError(
                f"Unknown optimizer type {value!r}. Valid: {sorted(OPTIMIZER_NAMES)}. "
                f"Aliases: {sorted(OPTIMIZER_ALIASES)}."
            )
        return key


class GradientClipConfigSchema(BaseModel):
    """Gradient clipping. ``enabled`` is the block's own gate, so it is bare."""

    model_config = dict(_SUBBLOCK)

    enabled: bool = Field(default=False)
    method: str = Field(default="norm", description="`norm` or `value`.")
    #: Allowed None for the configs that write `null`.
    value: float | None = Field(default=1.0, ge=0)


class GradientConfigSchema(BaseModel):
    """What happens between ``loss.backward()`` and ``optimizer.step()``.

    ``detect_anomalies`` lives here rather than in a debug block because it is a
    backward-pass switch: it flips ``torch.autograd.set_detect_anomaly`` and is
    read at ``pipelines/train.py`` into the same ``StabilityManager`` as
    ``clip.value``.
    """

    model_config = dict(_SUBBLOCK)

    accumulation_steps: int = Field(
        default=1, gt=0, description="Number of gradient accumulation steps."
    )
    enable_checkpointing: bool = Field(
        default=False,
        description="Trade compute for activation memory. Was two fields — "
        "`use_gradient_checkpointing` and the `gradient_checkpointing` alias some "
        "diffusion configs wrote — which is one knob too many.",
    )
    detect_anomalies: bool = Field(
        default=False,
        description="torch.autograd.set_detect_anomaly (NaN/Inf checks). Global "
        "and 2-4x slower. The divergence tripwire in the training loop is "
        "UNCONDITIONAL and no longer gated on this — see training_loop.py.",
    )
    clip: GradientClipConfigSchema = Field(default_factory=GradientClipConfigSchema)


class PrecisionConfigSchema(BaseModel):
    """Autocast precision. One decision, previously two unrelated-looking keys.

    ``dtype='float32'`` disables AMP even when ``enabled`` is true — a third
    state that was invisible while these were ``use_amp`` and ``amp_dtype`` at
    opposite ends of a 44-field wall.
    """

    model_config = dict(_SUBBLOCK)

    enabled: bool = Field(default=False)
    dtype: str | None = Field(
        default=None,
        description="Autocast dtype when enabled: 'float16', 'bfloat16' or "
        "'float32'. Wired through BaseTrainingStrategy via resolve_amp_precision: "
        "None/'float16' -> fp16, 'bfloat16' -> bf16, 'float32' -> AMP disabled. "
        "Prefer 'bfloat16' (no loss scaling, fp32 exponent range). See "
        "docs/troubleshooting.rst 'Mixed precision'.",
    )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str | None) -> str | None:
        """Reject illegal AMP dtypes at load time (fp8 is intentionally absent)."""
        if value is None:
            return None
        allowed = {"float16", "bfloat16", "float32"}
        if value not in allowed:
            raise ValueError(f"precision.dtype must be one of {sorted(allowed)}")
        return value


class CompileConfigSchema(BaseModel):
    """``torch.compile`` settings.

    Read by ``ModelBuilder.compile()``, wired at
    ``infrastructure/training/builders/director.py`` step 1. Compilation failure
    RAISES: when you asked for a compiled model, silently training an eager one
    is a lie about what ran. ``enabled: false`` is how you ask for eager.
    """

    model_config = dict(_SUBBLOCK)

    enabled: bool = Field(default=False)
    mode: CompileMode = Field(
        default="default",
        description="default / reduce-overhead / max-autotune / max-autotune-no-cudagraphs.",
    )
    backend: str = Field(
        default="inductor",
        description="Validated against torch._dynamo.list_backends() at load time "
        "when torch is importable.",
    )
    fullgraph: bool = Field(
        default=False,
        description="Compile the full graph (requires no dynamic shapes).",
    )
    dynamic: bool = Field(default=True, description="Allow dynamic shapes.")

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        """Reject an unknown torch.compile backend at load time.

        Guarded on torch being importable so the config layer stays loadable in
        doc builds and the torch-less CI shim (``conftest.py`` installs a
        MagicMock for torch); a mocked ``list_backends`` yields no usable set, so
        the check is skipped rather than made to reject everything.
        """
        try:
            import torch

            backends = torch._dynamo.list_backends()
        except Exception:  # pragma: no cover - torch absent or shimmed
            return value
        if not isinstance(backends, (list, tuple, set)) or not backends:
            return value
        allowed = {str(b) for b in backends}
        if value not in allowed:
            raise ValueError(f"Unknown compile backend {value!r}. Available: {sorted(allowed)}")
        return value


class MemoryConfigSchema(BaseModel):
    """Memory monitoring and mitigation. Diagnostics, not a training decision."""

    model_config = dict(_SUBBLOCK)

    enable_monitoring: bool = Field(default=False)
    monitoring_interval: int = Field(default=50, gt=0)
    enable_fragmentation_mitigation: bool = Field(default=False)
    cleanup_interval: int = Field(default=100, gt=0)
    enable_batch_size_optimization: bool = Field(default=False)
    safety_margin: float = Field(default=0.8, ge=0, le=1)


class OptimizationConfigSchema(BaseModel):
    """Optimizer, gradient, precision, compile and memory settings.

    Reached as ``config.optimization.<sub-block>.<key>`` — e.g.
    ``config.optimization.optimizer.learning_rate``. The flat spellings are
    retired through the rename SSOT and folded into place at parse time, so
    unmigrated YAML still loads while nothing in Python can read the old path.

    Strictness (H4): ``extra="forbid"`` — unknown keys are rejected so that typos
    (e.g. ``learining_rate``) fail loudly at config-load time instead of being
    silently dropped and replaced by defaults.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",  # H4: reject unknown keys so typos surface loudly
        "frozen": True,
    }

    optimizer: OptimizerConfigSchema = Field(default_factory=OptimizerConfigSchema)
    gradient: GradientConfigSchema = Field(default_factory=GradientConfigSchema)
    precision: PrecisionConfigSchema = Field(default_factory=PrecisionConfigSchema)
    compile: CompileConfigSchema = Field(default_factory=CompileConfigSchema)
    memory: MemoryConfigSchema = Field(default_factory=MemoryConfigSchema)

    # ------------------------------------------------------------------ #
    # Learning-rate scheduler — STILL FLAT, blocked on issue #662.
    #
    # Every key below is read by the SSOT resolver
    # ``mriforge.infrastructure.training.scheduler_resolution.resolve_scheduler_spec``,
    # which OptimizationBuilder.build_schedulers calls. The resolver RAISES on a
    # knob the resolved scheduler family cannot consume.
    #
    # These are NOT folded into a `scheduler:` sub-block, even though that is
    # what the readability plan asked for -- but the reason changed. The fold
    # used to be a training-behaviour change wearing a readability edit's
    # clothes, because `resolve_scheduler_spec` returned None when `scheduler`
    # was absent, BEFORE it looked at `lr_scheduler_strategy`, so 531 arms
    # trained at a constant LR and creating the dict would have started
    # annealing them. That defect is fixed (#662); the fold is now merely
    # unstarted work.
    # ------------------------------------------------------------------ #
    lr_scheduler_strategy: str = Field(
        default="cosine",
        description="Scheduler family: cosine / cosine_annealing / "
        "cosine_annealing_warm_restarts / step / linear_decay / plateau / "
        "warmup / linear_warmup / constant. Read by resolve_scheduler_spec "
        "whenever scheduler.type is absent, including when there is no "
        "`scheduler:` dict at all — but only when this key is DECLARED. Its "
        "default is never honoured, or the 305 arms that declare no scheduler "
        "would all start annealing (#662).",
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Linear-warmup length. Equivalent to scheduler.warmup_steps; "
        "declaring both with different values raises.",
    )
    lr_scheduler_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra scheduler params, merged with the scheduler dict. "
        "A disagreeing duplicate raises.",
    )
    scheduler: dict[str, Any] | None = Field(
        default=None,
        description="Scheduler parameters. Flat form is canonical "
        "(T_0/T_mult/eta_min/T_max/warmup_steps/warmup_start_lr, plus an "
        "optional 'type'); the legacy {type:, kwargs:} form still resolves. "
        "None means no scheduler is built — see #662. Deliberately an untyped "
        "mapping: resolve_scheduler_spec validates it against the resolved "
        "factory's signature, and a typed mirror here would be a second "
        "resolver that agrees until it doesn't (pitfall #13b).",
    )
    scheduler_type: str | None = Field(
        default=None,
        description="Flat scheduler family. Takes effect; conflicting with scheduler.type raises.",
    )
    T_max: int | None = Field(
        default=None,
        gt=0,
        description="CosineAnnealingLR period. Takes effect; conflicting with "
        "scheduler.T_max raises. Defaults to training.max_iterations.",
    )
    eta_min: float | None = Field(
        default=None,
        ge=0,
        description="Cosine floor LR. Takes effect; conflicting with scheduler.eta_min raises.",
    )

    # Declared, mounted, and read by NOTHING -- `rg 'optimization.num_steps'`
    # returns zero consumers across src/, scripts/, tools/ and tests/, while 4
    # arms set it. It has no sub-block above because it has no meaning to group
    # with: pitfall #15, tracked separately. Deleting it would break those 4
    # arms, so it stays visible at the top level rather than being given a home
    # that implies it works.
    num_steps: int | None = Field(
        default=None,
        ge=1,
        description="UNWIRED (#15): no consumer. Was advertised for test-time "
        "optimization / inner loops.",
    )

    #: Legacy leaf names this block still ACCEPTS as input and folds into a
    #: sub-block. Read by the execution ledger, which would otherwise see a
    #: folded key sitting in the raw YAML but absent from model_fields and
    #: report it as EXTRA_IGNORE_DROPPED at severity "error" -- up to 35
    #: spurious "the run never sees it" records on every unmigrated arm, which
    #: is how a ledger stops being read. Published as a class attribute rather
    #: than looked up from the table so `core/` need not import `config/`.
    __folded_input_keys__ = folded_input_keys("optimization")
    __folded_input_paths__ = folded_input_paths("optimization")

    # Retired flat spellings. `reject_renamed_keys` raises on the ones already
    # driven to zero; `fold_renamed_keys` moves the rest into their sub-block so
    # the corpus can migrate on its own schedule. Both read one table.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("optimization"))
    )
    _fold_renamed = model_validator(mode="before")(classmethod(fold_renamed_keys("optimization")))
