import pytest
import torch

from mriforge.config.schemas.model import ModelConfigSchema
from mriforge.config.schemas.optimization import OptimizationConfigSchema
from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
from mriforge.config.schemas.training.pipeline import (
    ActionSchema,
    MultiStageSchema,
    MultiTrainingConfigSchema,
    StageEnvironmentSchema,
    TrainingStateSchema,
)
from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)


@pytest.fixture
def multi_stage_settings() -> TrainingSettings:
    """Fixture providing a valid multi-stage TrainingSettings payload."""
    multi_cfg = MultiTrainingConfigSchema(
        stages=[
            MultiStageSchema(
                name="stage1",
                model_type="standard_unet",
                in_channels=1,
                out_channels=1,
                spatial_dims=2,
                model_kwargs={"features": [8, 8]},
                training_state=TrainingStateSchema(freeze_all=True),
            ),
            MultiStageSchema(
                name="stage2",
                model_type="standard_unet",
                in_channels=1,
                out_channels=1,
                spatial_dims=2,
                model_kwargs={"features": [8, 8]},
                training_state=TrainingStateSchema(freeze_all=False),
                stage_config=StageEnvironmentSchema(
                    optimization=OptimizationConfigSchema(
                        learning_rate=1e-3, optimizer_type="adamw", use_amp=False
                    ),
                    # We omit loss to simply test the forward pass mechanics
                ),
            ),
        ],
        routines={
            "train": [
                ActionSchema(
                    action="call",
                    node="stage1",
                    inputs={"x": "batch.input"},
                    outputs={"stage1_out": "output"},
                ),
                ActionSchema(
                    action="assign",
                    target="state.intermediary",
                    source="state.stage1_out",
                ),
                ActionSchema(
                    action="call",
                    node="stage2",
                    inputs={"x": "state.intermediary"},
                ),
            ]
        },
        end_to_end_finetune_epoch=-1,
    )

    return TrainingSettings(
        experiment_name="smoke_test_multi",
        config_version="5.0",
        training=TrainingStrategyConfigSchema(
            strategy_class="multi",
            num_epochs=1,
            device="cpu",
            multi=multi_cfg,
        ),
        optimization=OptimizationConfigSchema(
            learning_rate=1e-4, optimizer_type="adamw", use_amp=False
        ),
        model=ModelConfigSchema(
            model_type="standard_unet",
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
        ),
    )


def test_multi_stage_strategy_initialization(multi_stage_settings: TrainingSettings) -> None:
    """Test that the MultiTrainingStrategy initializes correctly."""
    strategy = MultiTrainingStrategy(
        env=multi_stage_settings,
        device=torch.device("cpu"),
    )

    assert "stage1" in strategy.multi_stages
    assert "stage2" in strategy.multi_stages
    assert len(strategy.routines) == 1
    assert "train" in strategy.routines
    assert len(strategy._stage_optimizers) == 1
    assert "stage2" in strategy._stage_optimizers


def test_multi_stage_strategy_forward(multi_stage_settings: TrainingSettings) -> None:
    """Test that the interpreter can execute a forward pass routine safely."""
    strategy = MultiTrainingStrategy(
        env=multi_stage_settings,
        device=torch.device("cpu"),
    )

    dummy_batch = {
        "input": torch.randn(2, 1, 16, 16),
        "target": torch.randn(2, 1, 16, 16),
    }

    # training_step usually does forward + loss + backward
    # But for a smoke test without explicit losses configured,
    # we can directly call _execute_flow to test the interpreter mechanics.
    state = {}
    strategy._execute_flow(
        actions=strategy.routines["train"],
        batch_data=dummy_batch,
        state=state,
        training=True
    )

    # Validate state changes
    assert "stage1_out" in state
    assert "intermediary" in state
    assert "stage2" in state

    assert state["stage1_out"].shape == (2, 1, 16, 16)
    assert state["stage2"].shape == (2, 1, 16, 16)

    # Check requires_grad flag based on frozen state
    assert state["stage1_out"].requires_grad is True  # Detached with requires_grad=True
    assert state["stage2"].requires_grad is True
