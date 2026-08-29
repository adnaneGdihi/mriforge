"""Unit tests for multi-stage training schemas (ActionSchema, routines, loops).

Tests schema validation, action types, recursive loop bodies, routines,
backward compatibility, and the full MultiTrainingConfigSchema.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.pipeline import (
    ActionSchema,
    MultiEvalMetricSchema,
    MultiEvalSchema,
    MultiStageSchema,
    MultiTrainingConfigSchema,
    RouteSchema,
    SlidingWindowSchema,
    StageEnvironmentSchema,
    TrainingStateSchema,
)

# -----------------------------------------------------------------------
# TrainingStateSchema
# -----------------------------------------------------------------------


class TestTrainingStateSchema:
    def test_defaults(self) -> None:
        state = TrainingStateSchema()
        assert state.freeze_all is False
        assert state.inject_lora is False
        assert state.lora_rank == 8

    def test_freeze_all(self) -> None:
        state = TrainingStateSchema(freeze_all=True)
        assert state.freeze_all is True

    def test_lora_config(self) -> None:
        state = TrainingStateSchema(
            freeze_all=True, inject_lora=True, lora_rank=4, lora_alpha=8
        )
        assert state.lora_rank == 4

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingStateSchema(unknown_field="x")

    def test_immutability(self) -> None:
        state = TrainingStateSchema(freeze_all=True)
        with pytest.raises(ValidationError):
            state.freeze_all = False


# -----------------------------------------------------------------------
# ActionSchema
# -----------------------------------------------------------------------


class TestActionSchema:
    def test_call_action(self) -> None:
        a = ActionSchema(
            action="call",
            node="encoder",
            inputs={"x": "batch.input"},
            outputs={"features": "output"},
        )
        assert a.action == "call"
        assert a.node == "encoder"

    def test_call_with_method(self) -> None:
        a = ActionSchema(action="call", node="scheduler", method="step")
        assert a.method == "step"

    def test_call_missing_node_raises(self) -> None:
        with pytest.raises(ValidationError, match="'call' action requires 'node'"):
            ActionSchema(action="call")

    def test_loop_action(self) -> None:
        body_action = ActionSchema(
            action="call",
            node="unet",
            inputs={"x": "state.volume"},
            outputs={"volume": "output"},
        )
        loop = ActionSchema(
            action="loop",
            iterator="batch.timesteps",
            loop_var="t",
            body=[body_action],
        )
        assert loop.action == "loop"
        assert loop.iterator == "batch.timesteps"
        assert loop.loop_var == "t"
        assert len(loop.body) == 1
        assert loop.body[0].node == "unet"

    def test_loop_missing_iterator_raises(self) -> None:
        with pytest.raises(ValidationError, match="'loop' action requires 'iterator'"):
            ActionSchema(
                action="loop",
                loop_var="t",
                body=[ActionSchema(action="call", node="unet")],
            )

    def test_loop_missing_loop_var_raises(self) -> None:
        with pytest.raises(ValidationError, match="'loop' action requires 'loop_var'"):
            ActionSchema(
                action="loop",
                iterator="batch.steps",
                body=[ActionSchema(action="call", node="unet")],
            )

    def test_loop_empty_body_raises(self) -> None:
        with pytest.raises(ValidationError, match="'loop' action requires non-empty"):
            ActionSchema(action="loop", iterator="batch.steps", loop_var="t")

    def test_assign_action(self) -> None:
        a = ActionSchema(
            action="assign",
            target="state.volume",
            source="batch.noise",
        )
        assert a.action == "assign"
        assert a.target == "state.volume"
        assert a.source == "batch.noise"

    def test_assign_missing_target_raises(self) -> None:
        with pytest.raises(ValidationError, match="'assign' action requires 'target'"):
            ActionSchema(action="assign", source="batch.noise")

    def test_assign_missing_source_raises(self) -> None:
        with pytest.raises(ValidationError, match="'assign' action requires 'source'"):
            ActionSchema(action="assign", target="state.volume")

    def test_detach_action(self) -> None:
        a = ActionSchema(action="detach", target="state.volume")
        assert a.action == "detach"
        assert a.target == "state.volume"

    def test_detach_missing_target_raises(self) -> None:
        with pytest.raises(ValidationError, match="'detach' action requires 'target'"):
            ActionSchema(action="detach")

    def test_nested_loop_body(self) -> None:
        """Loops can contain other loops (recursive)."""
        inner_call = ActionSchema(action="call", node="unet")
        inner_loop = ActionSchema(
            action="loop",
            iterator="batch.inner_steps",
            loop_var="s",
            body=[inner_call],
        )
        outer_loop = ActionSchema(
            action="loop",
            iterator="batch.outer_steps",
            loop_var="t",
            body=[inner_loop],
        )
        assert outer_loop.body[0].action == "loop"
        assert outer_loop.body[0].body[0].action == "call"

    def test_default_action_is_call(self) -> None:
        a = ActionSchema(node="encoder")
        assert a.action == "call"

    def test_loop_context_in_inputs(self) -> None:
        a = ActionSchema(
            action="call",
            node="unet",
            inputs={"t": "loop_context.timestep"},
        )
        assert a.inputs["t"] == "loop_context.timestep"


# -----------------------------------------------------------------------
# MultiStageSchema & StageEnvironmentSchema
# -----------------------------------------------------------------------


class TestStageEnvironmentSchema:
    def test_default_is_empty(self) -> None:
        cfg = StageEnvironmentSchema()
        assert cfg.training is None
        assert cfg.loss is None
        assert cfg.metrics is None
        assert cfg.early_stopping is None

    def test_with_nested_configs(self) -> None:
        # Utilizing Pydantic's dict instantiation for the nested components
        cfg = StageEnvironmentSchema(
            training={"strategy_class": "pipeline"},
            loss={"reconstruction": {"enable_l1": True, "lambda_l1": 5.0}},
        )
        assert cfg.training is not None
        assert getattr(cfg.training, "strategy_class", None) == "pipeline"
        assert cfg.loss is not None
        assert cfg.loss.reconstruction is not None
        assert cfg.loss.reconstruction.enable_l1 is True
        assert cfg.loss.reconstruction.lambda_l1 == 5.0


class TestMultiStageSchema:
    def test_required_fields(self) -> None:
        stage = MultiStageSchema(name="s1", model_type="standard_unet")
        assert stage.spatial_dims == 2
        assert stage.gradient_checkpointing is False
        assert stage.stage_config is None

    def test_3d_stage_with_stage_config(self) -> None:
        stage = MultiStageSchema(
            name="s3d",
            model_type="monai_unet",
            spatial_dims=3,
            gradient_checkpointing=True,
            in_channels=32,
            out_channels=1,
            stage_config=StageEnvironmentSchema(
                loss={"reconstruction": {"enable_l1": True}}
            ),
        )
        assert stage.spatial_dims == 3
        assert stage.stage_config is not None
        assert stage.stage_config.loss is not None
        assert stage.stage_config.loss.reconstruction is not None
        assert stage.stage_config.loss.reconstruction.enable_l1 is True

    def test_invalid_spatial_dims_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultiStageSchema(name="s", model_type="unet", spatial_dims=4)


# -----------------------------------------------------------------------
# SlidingWindowSchema & MultiEvalSchema
# -----------------------------------------------------------------------


class TestSlidingWindowSchema:
    def test_3d_config(self) -> None:
        sw = SlidingWindowSchema(roi_size=[64, 64, 64], overlap=0.5)
        assert sw.roi_size == [64, 64, 64]

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValidationError):
            SlidingWindowSchema(roi_size=[64, 64], overlap=1.0)


class TestMultiEvalSchema:
    def test_with_metrics(self) -> None:
        eval_cfg = MultiEvalSchema(
            metrics={"dice": MultiEvalMetricSchema(target="monai.metrics.DiceMetric")}
        )
        assert "dice" in eval_cfg.metrics


# -----------------------------------------------------------------------
# MultiTrainingConfigSchema
# -----------------------------------------------------------------------


class TestMultiTrainingConfigSchema:
    def test_routines_config(self) -> None:
        config = MultiTrainingConfigSchema(
            stages=[MultiStageSchema(name="unet", model_type="standard_unet")],
            routines={
                "train": [
                    ActionSchema(
                        action="call",
                        node="unet",
                        inputs={"x": "batch.input"},
                        outputs={"prediction": "output"},
                    ),
                ],
                "generate": [
                    ActionSchema(
                        action="assign",
                        target="state.x",
                        source="batch.noise",
                    ),
                    ActionSchema(
                        action="loop",
                        iterator="batch.steps",
                        loop_var="t",
                        body=[
                            ActionSchema(
                                action="call",
                                node="unet",
                                inputs={"x": "state.x", "t": "loop_context.t"},
                                outputs={"x": "output"},
                            ),
                        ],
                    ),
                ],
            },
        )
        assert "train" in config.routines
        assert "generate" in config.routines
        assert len(config.routines["generate"]) == 2
        assert config.routines["generate"][1].action == "loop"

    def test_flow_backward_compat(self) -> None:
        config = MultiTrainingConfigSchema(
            stages=[MultiStageSchema(name="enc", model_type="unet")],
            flow=[
                ActionSchema(
                    action="call",
                    node="enc",
                    inputs={"x": "batch.input"},
                    outputs={"features": "output"},
                ),
            ],
        )
        assert len(config.flow) == 1

    def test_duplicate_stage_names_raises(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate stage names"):
            MultiTrainingConfigSchema(
                stages=[
                    MultiStageSchema(name="s1", model_type="unet"),
                    MultiStageSchema(name="s1", model_type="swin"),
                ],
            )

    def test_invalid_flow_node_raises(self) -> None:
        with pytest.raises(ValidationError, match="unknown stage"):
            MultiTrainingConfigSchema(
                stages=[MultiStageSchema(name="s1", model_type="unet")],
                flow=[ActionSchema(action="call", node="nonexistent")],
            )

    def test_invalid_routine_node_raises(self) -> None:
        with pytest.raises(ValidationError, match="unknown stage"):
            MultiTrainingConfigSchema(
                stages=[MultiStageSchema(name="s1", model_type="unet")],
                routines={
                    "train": [ActionSchema(action="call", node="ghost")],
                },
            )

    def test_invalid_loop_body_node_raises(self) -> None:
        """Node references inside loop bodies must also be validated."""
        with pytest.raises(ValidationError, match="unknown stage"):
            MultiTrainingConfigSchema(
                stages=[MultiStageSchema(name="s1", model_type="unet")],
                routines={
                    "gen": [
                        ActionSchema(
                            action="loop",
                            iterator="batch.steps",
                            loop_var="t",
                            body=[ActionSchema(action="call", node="invalid")],
                        ),
                    ],
                },
            )

    def test_routing_backward_compat(self) -> None:
        config = MultiTrainingConfigSchema(
            stages=[
                MultiStageSchema(name="s1", model_type="unet"),
                MultiStageSchema(name="s2", model_type="swin"),
            ],
            routing=[RouteSchema(from_stage="s1", to_stage="s2")],
        )
        assert len(config.routing) == 1

    def test_finetune_epoch(self) -> None:
        config = MultiTrainingConfigSchema(
            stages=[MultiStageSchema(name="s1", model_type="unet")],
            end_to_end_finetune_epoch=50,
        )
        assert config.end_to_end_finetune_epoch == 50

    def test_with_evaluation(self) -> None:
        config = MultiTrainingConfigSchema(
            stages=[MultiStageSchema(name="dec", model_type="unet", spatial_dims=3)],
            evaluation=MultiEvalSchema(
                sliding_window=SlidingWindowSchema(roi_size=[64, 64, 64]),
            ),
        )
        assert config.evaluation is not None

    def test_backward_compat_aliases(self) -> None:
        from mriforge.config.schemas.training.pipeline import (
            FlowStepSchema,
            PipelineStageSchema,
            PipelineTrainingConfigSchema,
        )

        assert PipelineStageSchema is MultiStageSchema
        assert PipelineTrainingConfigSchema is MultiTrainingConfigSchema
        assert FlowStepSchema is ActionSchema
