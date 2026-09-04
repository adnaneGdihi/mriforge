import pytest
from pydantic import ValidationError

from spectramr.config.schemas.optimization import OptimizationConfigSchema


class TestOptimizationConfigSchema:
    def test_defaults(self):
        schema = OptimizationConfigSchema()
        assert schema.optimizer.learning_rate == 1e-5
        assert schema.optimizer.type == "adamw"
        assert schema.precision.enabled is False
        assert schema.gradient.accumulation_steps == 1
        assert schema.lr_scheduler_strategy == "cosine"

    def test_custom_values(self):
        schema = OptimizationConfigSchema(
            learning_rate=1e-4,
            optimizer_type="sgd",
            use_amp=True,
            gradient_accumulation_steps=4,
            lr_scheduler_strategy="linear",
        )
        assert schema.optimizer.learning_rate == 1e-4
        assert schema.optimizer.type == "sgd"
        assert schema.precision.enabled is True
        assert schema.gradient.accumulation_steps == 4
        assert schema.lr_scheduler_strategy == "linear"

    def test_validation(self):
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(learning_rate=0)
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(gradient_accumulation_steps=0)
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(beta1=1.1)

    def test_extra_fields_raise(self):
        # H4 (audit 2026-05-28): extra="forbid" — an unknown key (e.g. a typo
        # like ``learining_rate``) must fail loudly at load time rather than be
        # silently dropped and replaced by the default. See CLAUDE.md #9/#15.
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(learining_rate=1e-4)

    def test_scheduler_dict(self):
        scheduler_config = {"type": "cosine", "warmup": 100}
        schema = OptimizationConfigSchema(scheduler=scheduler_config)
        assert schema.scheduler == scheduler_config

    def test_param_group_overrides_accepted(self):
        # H4: param_group_overrides is used by 1 live config; declared so it
        # validates under extra=forbid.
        overrides = {"siren": {"learning_rate": 1e-4}}
        schema = OptimizationConfigSchema(optimizer={"param_groups": overrides})
        # The override is parsed into a `ParamGroupOverrideSchema` rather than
        # kept as the raw dict, so comparing to `overrides` compares a model to
        # a mapping. Assert the value arrived, which is what "accepted" means
        # here. (The flat `param_group_overrides` spelling this test used to
        # pass was promoted to `raise` on 2026-08-18.)
        assert schema.optimizer.param_groups["siren"].learning_rate == 1e-4

    def test_amp_dtype_validated(self):
        # H4: the dtype is validated against the allowed set (fail-fast on
        # typos). Declared through the canonical `precision` block -- the legacy
        # `amp_dtype` spelling was promoted to `raise` in #919, so the two cases
        # below would otherwise both raise and the test would pass for the wrong
        # reason.
        assert (
            OptimizationConfigSchema(precision={"dtype": "bfloat16"}).precision.dtype
            == "bfloat16"
        )
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(precision={"dtype": "float8"})

    def test_betas_validated(self):
        assert OptimizationConfigSchema(betas=(0.9, 0.999)).optimizer.betas == (0.9, 0.999)
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(betas=(0.9, 1.5))


class TestAccumulateGradStepsRetired:
    """`accumulate_grad_steps` was a second spelling of
    `gradient_accumulation_steps`, folded in by a hand-written before-validator
    so which one won depended on declaration order. Retired via the rename SSOT.
    """

    def test_legacy_key_raises_naming_the_replacement(self) -> None:
        import pytest

        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        with pytest.raises(Exception) as exc:
            OptimizationConfigSchema(accumulate_grad_steps=4)
        msg = str(exc.value)
        # The DESTINATION moved after this test was written: the replacement was
        # `gradient_accumulation_steps` and is now `gradient.accumulation_steps`
        # (the block decomposition). A rename test has two moving parts, and
        # pinning the old destination fails on the second move even though the
        # message is doing exactly its job.
        assert "gradient.accumulation_steps" in msg
        assert "migrate_config_keys.py" in msg

    def test_field_is_gone(self) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        assert "accumulate_grad_steps" not in OptimizationConfigSchema.model_fields

    def test_canonical_still_works(self) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        cfg = OptimizationConfigSchema(gradient_accumulation_steps=4)
        assert cfg.gradient.accumulation_steps == 4
