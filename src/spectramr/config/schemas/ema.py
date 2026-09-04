"""Exponential Moving Average (EMA) configuration schema.

BREAKING CHANGE (v5.1): Removed 8 redundant alias fields to enforce SSOT principle.
Legacy aliases (enable_ema, ema_decay, ema_update_frequency, ema_warmup_steps,
ema_initial_decay, ema_final_decay, ema_stability_threshold, ema_adaptation_rate)
are no longer accepted. Use primary field names.
"""

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class EMAConfigSchema(BaseModel):
    """Exponential Moving Average configuration.

    Consolidates all EMA-related hyperparameters (previously scattered in TrainingConfigSchema).

    **SSOT Principle**: Single field name per concept. No aliases.
    - Use 'enabled' not 'enable_ema'
    - Use 'decay' not 'ema_decay'
    - Use 'update_frequency' not 'ema_update_frequency'
    - Use 'warmup_steps' not 'ema_warmup_steps'
    - Use 'initial_decay' not 'ema_initial_decay'
    - Use 'final_decay' not 'ema_final_decay'
    - Use 'stability_threshold' not 'ema_stability_threshold'
    - Use 'adaptation_rate' not 'ema_adaptation_rate'

    Example:
        >>> config = EMAConfigSchema(
        ...     enabled=True,
        ...     decay=0.999,
        ...     update_frequency=1,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable exponential moving average",
    )
    decay: float = Field(
        default=0.999,
        ge=0,
        le=1,
        description="EMA decay coefficient",
    )
    update_frequency: int = Field(
        default=1,
        ge=1,
        description="Update EMA every N steps",
    )
    warmup: bool = Field(
        default=True,
        description=(
            "Ramp the EMA decay with num_updates on the standard ModelEma path "
            "(effective_decay = min(decay, (1+n)/(10+n))) so the shadow tracks "
            "the live model EARLY instead of staying ~stuck at the random init. "
            "Without it, decay=0.9999 leaves the shadow ~74% random init at "
            "iter 3000 (the Experiment-11 EMA-lag: validation graded a "
            "near-untrained shadow). Set False for the fixed-decay baseline arm."
        ),
    )

    # Adaptive EMA
    enable_adaptive_ema: bool = Field(
        default=False,
        description="Use adaptive EMA decay (adjusted during training)",
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Length of the EMA warmup period, in updates. The MECHANISM "
            "differs per path: on the standard path it is a hard gate — the "
            "training loop skips EMA updates entirely until this many "
            "iterations have passed, holding the shadow at the init. With "
            "enable_adaptive_ema it is instead the length of the soft decay "
            "ramp initial_decay -> final_decay, during which the shadow "
            "TRACKS the live model closely. The loop applies only one of the "
            "two (it reads ModelEma.adaptive), so the periods never stack. "
            "Required (> 0) when enable_adaptive_ema is set."
        ),
    )
    initial_decay: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Initial decay for adaptive EMA",
    )
    final_decay: float = Field(
        default=0.999,
        ge=0,
        le=1,
        description="Final decay for adaptive EMA",
    )

    # Stability and monitoring
    stability_threshold: float = Field(
        default=1e-6,
        ge=0,
        description=(
            "Threshold for EMA stability checks. NOT READ by any EMA code — "
            "the only `stability_threshold` in src/ is an unrelated "
            "constructor parameter of `AdaptiveWarmupScheduler` "
            "(infrastructure/training/scheduler_system.py), a learning-rate "
            "scheduler nothing constructs; the leaf names merely collide. "
            "Same status and same caveats as adaptation_rate above."
        ),
    )
    adaptation_rate: float = Field(
        default=0.01,
        ge=0,
        description=(
            "Rate of adaptation for adaptive EMA. NOT READ — nothing in src/ "
            "consumes it (KNOWN_UNCONSUMED ledger, "
            "tests/unit/config/test_schema_key_consumption.py). The adaptive "
            "decay ramp wired in #1294 is a linear interpolation over "
            "warmup_steps and takes no feedback signal, so there is no "
            "schedule for this rate to modulate. 31 reference arms declare it "
            'and this block is extra="forbid", so deleting the field would '
            "fail every one of them at load; setting it together with "
            "enable_adaptive_ema is refused instead. Wire it or retire it "
            "with a corpus migration; do not trust it today."
        ),
    )
    enable_momentum_scaling: bool = Field(
        default=False,
        description="Scale momentum based on batch size",
    )
    enable_ema_for_optimizer: bool = Field(
        default=False,
        description="Apply EMA to optimizer state (not just model weights)",
    )

    @model_validator(mode="after")
    def validate_adaptive_ema_is_fully_wired(self) -> "EMAConfigSchema":
        """Refuse an adaptive-EMA declaration this framework cannot honour.

        ``enable_adaptive_ema`` was schema-only until #1294: the implementation
        at ``models/utils/adaptive_ema.py`` was deleted in ff0efff9f and only
        its orphaned caller survived. The DETERMINISTIC half of that class (the
        ``initial_decay -> final_decay`` ramp over ``warmup_steps``) is now
        genuinely wired into
        :class:`~spectramr.infrastructure.optimization.ema.ModelEma`.

        Its stability-feedback half is NOT, and deliberately so: the historical
        ``update_stability_score`` had no caller anywhere in the tree, so its
        gradient-norm signal never arrived and the branch it fed reduced to a
        constant. Restoring it would ship a facade (pitfall #16). So rather
        than accept ``stability_threshold`` / ``adaptation_rate`` and quietly
        ignore them, we raise — an unread knob must never be advertised
        (non-negotiable 8).
        """
        if not self.enable_adaptive_ema:
            return self

        if self.warmup_steps <= 0:
            raise ValueError(
                "enable_adaptive_ema requires warmup_steps > 0 — it is the "
                "length of the decay ramp from initial_decay to final_decay. "
                f"Got warmup_steps={self.warmup_steps}, a zero-length ramp, "
                "which would silently collapse to a fixed final_decay."
            )

        unwired = [
            name
            for name in ("stability_threshold", "adaptation_rate")
            if name in self.model_fields_set
        ]
        if unwired:
            raise ValueError(
                f"ema.{unwired} is set together with enable_adaptive_ema, but "
                "the stability-feedback schedule those knobs configure is NOT "
                "implemented: the loss / gradient-norm signal it needs is "
                "never fed to the EMA. Only the deterministic ramp "
                "(warmup_steps, initial_decay, final_decay) is wired. Remove "
                "these keys to use the ramp, or leave enable_adaptive_ema "
                "off. See issue #1294."
            )
        return self

    @field_validator("final_decay")
    @classmethod
    def validate_decay_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure final_decay >= initial_decay."""
        if "initial_decay" in info.data and v < info.data["initial_decay"]:
            raise ValueError("final_decay must be >= initial_decay")
        return v

    @field_validator("decay")
    @classmethod
    def validate_decay_vs_adaptiveema(cls, v: float, info: ValidationInfo) -> float:
        """Warn if decay is very close to 0 or 1."""
        if v < 0.5:
            raise ValueError("decay < 0.5 is likely unstable; use decay >= 0.5")
        return v


__all__ = ["EMAConfigSchema"]
