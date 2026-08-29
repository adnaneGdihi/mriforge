"""Extended unit tests for MultiStageConfigValidator.

PART A — Task III.3: additional coverage on top of the existing
tests/unit/config/test_multi_stage_config_validator.py.

Covers:
  - canary: single-stage sequential config normalizes cleanly.
  - parametrized: vae/diffusion/gan stage types.
  - edge: empty dependencies, missing optional fields, encoder_decoder mode.
  - expected-failure: invalid optimizer, non-positive epochs/batch_size/lr,
    unknown dependency.
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.multi_stage_config_validator import (
    DependencyList,
    MultiStageConfigValidator,
    MultiStageMode,
    MultiStageTrainingConfig,
    StageType,
    create_multi_stage_config_validator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> MultiStageConfigValidator:
    return create_multi_stage_config_validator()


@pytest.fixture
def minimal_one_stage() -> dict:
    return {
        "experiment_name": "minimal_test",
        "mode": "sequential",
        "stages": [
            {
                "name": "encode",
                "model_type": "vae",
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 1e-3,
            }
        ],
    }


# ---------------------------------------------------------------------------
# 1. CANARY
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiStageCanary:
    def test_single_stage_normalizes(
        self, validator: MultiStageConfigValidator, minimal_one_stage: dict
    ) -> None:
        cfg = validator.validate_and_normalize(minimal_one_stage)
        assert isinstance(cfg, MultiStageTrainingConfig)
        assert cfg.mode == MultiStageMode.SEQUENTIAL
        assert len(cfg.stages) == 1
        assert cfg.stages[0].type == StageType.VAE

    def test_factory_function_returns_validator(self) -> None:
        v = create_multi_stage_config_validator()
        assert isinstance(v, MultiStageConfigValidator)


# ---------------------------------------------------------------------------
# 2. PARAMETRIZED — stage types
# ---------------------------------------------------------------------------


STAGE_TYPE_CASES = [
    ("vae", StageType.VAE),
    ("vqvae", StageType.VAE),
    ("diffusion", StageType.DIFFUSION),
    ("ddpm_unet", StageType.DIFFUSION),
    ("cold_diffusion", StageType.DIFFUSION),
    ("gan_model", StageType.GAN),
    ("adversarial_net", StageType.GAN),
    ("normalizing_flow_net", StageType.FLOW),
    ("standard_unet", StageType.RECONSTRUCTION),
]


@pytest.mark.unit
@pytest.mark.parametrize("model_type,expected_type", STAGE_TYPE_CASES)
def test_stage_type_inference(
    validator: MultiStageConfigValidator,
    model_type: str,
    expected_type: StageType,
) -> None:
    inferred = validator._infer_stage_type(model_type)
    assert inferred == expected_type, (
        f"model_type={model_type!r}: expected {expected_type}, got {inferred}"
    )


# ---------------------------------------------------------------------------
# 3. EDGE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiStageEdgeCases:
    def test_no_mode_defaults_to_sequential(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "stages": [{"name": "s1", "model_type": "standard_unet"}]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.mode == MultiStageMode.SEQUENTIAL

    def test_parallel_mode_accepted(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "mode": "parallel",
            "stages": [{"name": "s1", "model_type": "vae"}],
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.mode == MultiStageMode.PARALLEL

    def test_conditional_mode_accepted(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "mode": "conditional",
            "stages": [{"name": "s1", "model_type": "vae"}],
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.mode == MultiStageMode.CONDITIONAL

    def test_empty_dependencies_normalizes(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "stages": [
                {"name": "s1", "model_type": "vae", "dependencies": []}
            ]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.stages[0].dependencies == []

    def test_stage_with_all_optional_fields_absent(
        self, validator: MultiStageConfigValidator
    ) -> None:
        """Only name + model_type required; all optional fields get defaults."""
        cfg_dict = {
            "stages": [{"name": "minimal_stage", "model_type": "diffusion"}]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        stage = cfg.stages[0]
        assert stage.epochs == 100  # default
        assert stage.batch_size == 8  # default
        assert stage.learning_rate == pytest.approx(0.001)  # default
        assert stage.optimizer == "adam"  # default

    def test_encoder_decoder_mode_inferred(
        self, validator: MultiStageConfigValidator
    ) -> None:
        """Stage with encoder/decoder sections infers mode='encoder_decoder'."""
        cfg_dict = {
            "stages": [
                {
                    "name": "ed_stage",
                    "encoder": {"model_type": "vae"},
                    "decoder": {"model_type": "vae"},
                }
            ]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.stages[0].mode == "encoder_decoder"

    def test_data_flow_empty_accepted(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "stages": [{"name": "s1", "model_type": "vae"}],
            "data_flow": {},
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert cfg.data_flow.links == []
        assert cfg.data_flow.manifest_path is None

    def test_multiple_stages_with_dependency_chain(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "stages": [
                {"name": "vae_stage", "model_type": "vae"},
                {
                    "name": "ldm_stage",
                    "model_type": "diffusion",
                    "dependencies": ["vae_stage"],
                },
            ]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert "vae_stage" in list(cfg.stages[1].dependencies)

    def test_dependency_list_is_dependency_list_type(
        self, validator: MultiStageConfigValidator
    ) -> None:
        cfg_dict = {
            "stages": [
                {"name": "a", "model_type": "vae"},
                {"name": "b", "model_type": "diffusion", "dependencies": ["a"]},
            ]
        }
        cfg = validator.validate_and_normalize(cfg_dict)
        assert isinstance(cfg.stages[1].dependencies, DependencyList)


# ---------------------------------------------------------------------------
# 4. EXPECTED-FAILURE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiStageExpectedFailures:
    def test_empty_config_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            validator.validate_and_normalize({})

    def test_empty_stages_list_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate_and_normalize({"stages": []})

    def test_invalid_mode_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            validator.validate_and_normalize(
                {
                    "mode": "bogus_mode",
                    "stages": [{"name": "s", "model_type": "vae"}],
                }
            )

    def test_stage_missing_name_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError, match="missing required field: name"):
            validator.validate_and_normalize(
                {"stages": [{"model_type": "vae"}]}
            )

    def test_stage_zero_epochs_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate_and_normalize(
                {
                    "stages": [
                        {
                            "name": "s",
                            "model_type": "vae",
                            "epochs": 0,
                        }
                    ]
                }
            )

    def test_stage_negative_batch_size_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate_and_normalize(
                {
                    "stages": [
                        {
                            "name": "s",
                            "model_type": "vae",
                            "batch_size": -4,
                        }
                    ]
                }
            )

    def test_stage_zero_learning_rate_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate_and_normalize(
                {
                    "stages": [
                        {
                            "name": "s",
                            "model_type": "vae",
                            "learning_rate": 0.0,
                        }
                    ]
                }
            )

    def test_invalid_optimizer_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError, match="Invalid optimizer"):
            validator.validate_and_normalize(
                {
                    "stages": [
                        {
                            "name": "s",
                            "model_type": "vae",
                            "optimizer": "nadam",  # not in allowed set
                        }
                    ]
                }
            )

    def test_unknown_dependency_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        with pytest.raises(ValueError, match="depends on unknown stage"):
            validator.validate_and_normalize(
                {
                    "stages": [
                        {
                            "name": "s",
                            "model_type": "diffusion",
                            "dependencies": ["ghost_stage"],
                        }
                    ]
                }
            )

    def test_stage_no_model_type_and_no_bindings_raises(
        self, validator: MultiStageConfigValidator
    ) -> None:
        """Stage must have model_type or decoder.model_type or input_bindings."""
        with pytest.raises(ValueError):
            validator.validate_and_normalize(
                {"stages": [{"name": "s"}]}
            )
