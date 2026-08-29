"""Multi-Stage Training Configuration Schema.

Defines schemas for cascaded/stitched model pipelines with a recursive
graph interpreter supporting named routines, action-based flow
(call/loop/assign/detach), hybrid 2D/3D architectures, per-stage
freeze/LoRA control, gradient checkpointing, and sliding window inference.

The schema supports three execution modes (backward compatible):

1. **Sequential** — linear routing via ``routing`` list (simplest)
2. **DAG flow** — flat ``flow`` list of actions (supersedes routing)
3. **Routines** — named execution flows (``routines.train``, ``routines.generate_3d``)

Example YAML (Routines with loop)::

    training:
      strategy_class: 'multi'
      multi:
        stages:
          - name: unet_3d
            model_type: diffusion_unet
            spatial_dims: 3
          - name: scheduler
            model_type: ddim_scheduler
            training_state: { freeze_all: true }
        routines:
          train:
            - action: call
              node: unet_3d
              inputs: { x: "batch.noisy_volume", timestep: "batch.random_t" }
              outputs: { noise_pred: "output" }
          generate_3d:
            - action: assign
              target: "state.current_volume"
              source: "batch.pure_noise"
            - action: loop
              iterator: "batch.inference_steps"
              loop_var: "t"
              body:
                - action: call
                  node: unet_3d
                  inputs: { x: "state.current_volume", timestep: "loop_context.t" }
                  outputs: { noise_pred: "output" }
                - action: call
                  node: scheduler
                  method: step
                  inputs:
                    model_output: "state.noise_pred"
                    timestep: "loop_context.t"
                    sample: "state.current_volume"
                  outputs: { current_volume: "prev_sample" }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from ..early_stopping import EarlyStoppingConfigSchema
from ..loss import LossConfigSchema
from ..metrics import MetricsConfigSchema
from ..optimization import OptimizationConfigSchema
from ..strictness import StrictSchema

if TYPE_CHECKING:
    from .base import TrainingStrategyConfigSchema

# -----------------------------------------------------------------------
# Per-stage training state (freeze / unfreeze / LoRA)
# -----------------------------------------------------------------------


class TrainingStateSchema(BaseModel):
    """Per-stage freeze/unfreeze/LoRA configuration.

    Controls the gradient flow for a single multi-stage node:

    - ``freeze_all``: sets ``requires_grad=False`` on every parameter.
    - ``unfreeze_modules``: re-enables gradients for named-parameter patterns
      (applied *after* ``freeze_all`` so you can freeze-then-selectively-unfreeze).
    - ``inject_lora``: attaches LoRA adapters to matching Linear/Conv2d layers.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    freeze_all: bool = Field(
        default=False,
        description="Set all parameters requires_grad=False",
    )
    unfreeze_modules: list[str] = Field(
        default_factory=list,
        description="Named-parameter patterns to make trainable (e.g. ['attn', 'proj'])",
    )
    inject_lora: bool = Field(
        default=False,
        description="Attach LoRA adapters to matching layers",
    )
    lora_rank: int = Field(default=8, ge=1, description="LoRA low-rank dimension")
    lora_alpha: int = Field(default=16, ge=1, description="LoRA alpha scaling factor")
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["attn", "proj"],
        description="Module name patterns to attach LoRA to",
    )

    @model_validator(mode="after")
    def validate_unfreeze_requires_freeze(self) -> TrainingStateSchema:
        """Warn-free: unfreeze_modules only matters when freeze_all is True."""
        if self.unfreeze_modules and not self.freeze_all:
            pass  # Semantically valid but unusual
        return self


# -----------------------------------------------------------------------
# Legacy routing (backward compat)
# -----------------------------------------------------------------------


class RouteSchema(BaseModel):
    """Legacy linear connection between two stages.

    Kept for backward compatibility with simple sequential pipelines.
    For complex flows, use ``ActionSchema`` in ``routines`` instead.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    from_stage: str = Field(..., description="Source stage name")
    from_output: str = Field(default="output", description="Output key from source")
    to_stage: str = Field(..., description="Destination stage name")
    to_input: str = Field(default="input", description="Input key for destination")


# -----------------------------------------------------------------------
# Action-based flow (recursive graph interpreter)
# -----------------------------------------------------------------------


class ActionSchema(BaseModel):
    """Single instruction in the recursive graph interpreter.

    Every flow step defines an ``action`` type that determines execution:

    ``call``
        Invoke a registered model (or a specific method on it).
        Reads from ``batch.*``, ``state.*``, or ``loop_context.*``
        and writes results to ``state.*``.

    ``loop``
        Iterate over a source (e.g. diffusion timesteps) and recursively
        execute the ``body`` actions for each iteration. The current
        value is injected as ``loop_context.<loop_var>``.

    ``assign``
        Copy a value from a source namespace into the state memory bank.
        Used to initialize state (e.g. ``state.current_volume = batch.pure_noise``).

    ``detach``
        Sever the computational graph on a state tensor to prevent
        backpropagation through time (BPTT). Essential for memory-safe
        recurrent loops during training.

    Example::

        routines:
          generate:
            - action: assign
              target: "state.volume"
              source: "batch.noise"
            - action: loop
              iterator: "batch.timesteps"
              loop_var: "t"
              body:
                - action: call
                  node: unet
                  inputs: { x: "state.volume", t: "loop_context.t" }
                  outputs: { volume: "output" }
                - action: detach
                  target: "state.volume"
    """

    model_config = {"frozen": True, "extra": "forbid"}

    action: Literal["call", "loop", "assign", "detach"] = Field(
        default="call",
        description="Instruction type: call, loop, assign, or detach",
    )

    # ── call fields ──
    node: str | None = Field(
        default=None, description="Stage name to execute (required for 'call')"
    )
    method: str | None = Field(
        default=None,
        description="Specific method to call instead of forward() (e.g. 'step')",
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of model param → data source. "
            "Sources: 'batch.<key>', 'state.<key>', 'loop_context.<var>'"
        ),
    )
    outputs: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of state key → model output attribute ('output' for tensor)",
    )

    # ── loop fields ──
    iterator: str | None = Field(
        default=None,
        description="Source of iteration values (e.g. 'batch.inference_steps')",
    )
    loop_var: str | None = Field(
        default=None,
        description="Variable name injected into loop_context (e.g. 't')",
    )
    body: list[ActionSchema] = Field(
        default_factory=list,
        description="Sub-actions to execute on each loop iteration (recursive)",
    )

    # ── assign / detach fields ──
    target: str | None = Field(
        default=None,
        description="State key to write to (e.g. 'state.volume')",
    )
    source: str | None = Field(
        default=None,
        description="Source to read from (e.g. 'batch.noise', 'state.pred')",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> ActionSchema:
        """Validate that required fields are set for each action type."""
        if self.action == "call" and not self.node:
            raise ValueError("'call' action requires 'node' field")
        if self.action == "loop":
            if not self.iterator:
                raise ValueError("'loop' action requires 'iterator' field")
            if not self.loop_var:
                raise ValueError("'loop' action requires 'loop_var' field")
            if not self.body:
                raise ValueError("'loop' action requires non-empty 'body'")
        if self.action == "assign":
            if not self.target:
                raise ValueError("'assign' action requires 'target' field")
            if not self.source:
                raise ValueError("'assign' action requires 'source' field")
        if self.action == "detach" and not self.target:
            raise ValueError("'detach' action requires 'target' field")
        return self


# Rebuild model for recursive self-reference
ActionSchema.model_rebuild()

# Legacy alias
FlowStepSchema = ActionSchema


# -----------------------------------------------------------------------
# Stage definition
# -----------------------------------------------------------------------


class StageEnvironmentSchema(BaseModel):
    """Full configuration environment for a single stage.

    Replaces the global configuration (training, loss, metrics, etc.)
    for this specific stage, enabling true multi-paradigm hybrid architectures
    where each stage carries its own objectives and stopping criteria.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    training: TrainingStrategyConfigSchema | None = Field(
        default=None,
        description="Stage-specific training paradigm overrides (e.g., GAN vs Diffusion)",
    )
    loss: LossConfigSchema | None = Field(
        default=None,
        description="Stage-specific loss configuration overrides",
    )
    metrics: MetricsConfigSchema | None = Field(
        default=None,
        description="Stage-specific metrics computation overrides",
    )
    early_stopping: EarlyStoppingConfigSchema | None = Field(
        default=None,
        description="Stage-specific early stopping controls",
    )
    optimization: OptimizationConfigSchema | None = Field(
        default=None,
        description="Stage-specific optimizer and learning rate overrides",
    )


class MultiStageSchema(StrictSchema):
    """Configuration for a single stage in the multi-model graph.

    Each stage is an independently registered generator that can be
    frozen, partially trainable, or LoRA-augmented. Supports both
    2D (Conv2d) and 3D (Conv3d) spatial dimensions.

    Inherits StrictSchema (forbid + frozen + protected_namespaces=()) — the
    last of which is required because of the ``model_type`` / ``model_kwargs``
    fields below (otherwise Pydantic warns on the protected ``model_`` namespace).
    """

    name: str = Field(..., description="Unique stage identifier (e.g. 'coarse_recon')")
    model_type: str = Field(..., description="Registered generator name in ModelRegistry")
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to pre-trained weights (.safetensors or .pt)",
    )
    training_state: TrainingStateSchema = Field(
        default_factory=TrainingStateSchema,
        description="Freeze/unfreeze/LoRA configuration for this stage",
    )
    stage_config: StageEnvironmentSchema | None = Field(
        default=None,
        description="Full training environment configuration for this stage (loss, strategy, metrics, early stopping)",
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="Extra keyword arguments passed to ModelFactory.create_generator()",
    )

    # Dimensionality & architecture overrides
    spatial_dims: int = Field(
        default=2,
        ge=2,
        le=3,
        description="Spatial dimensions: 2 for Conv2d, 3 for Conv3d",
    )
    in_channels: int | None = Field(
        default=None,
        ge=1,
        description="Override global model.in_channels for this stage",
    )
    out_channels: int | None = Field(
        default=None,
        ge=1,
        description="Override global model.out_channels for this stage",
    )

    # VRAM optimization
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for this stage (trade compute for VRAM)",
    )


# -----------------------------------------------------------------------
# Evaluation & inference
# -----------------------------------------------------------------------


class SlidingWindowSchema(BaseModel):
    """Sliding window inference configuration for volumetric stages."""

    model_config = {"frozen": True, "extra": "forbid"}

    roi_size: list[int] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Patch size for inference, e.g. [64, 64, 64] for 3D",
    )
    sw_batch_size: int = Field(
        default=4, ge=1, description="Number of patches to process simultaneously"
    )
    overlap: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Overlap fraction between adjacent patches",
    )
    mode: str = Field(
        default="gaussian",
        description="Blending mode: 'gaussian' or 'constant'",
    )


class MultiEvalMetricSchema(BaseModel):
    """Configuration for a single evaluation metric."""

    model_config = {"frozen": True, "extra": "forbid"}

    target: str = Field(
        ...,
        description="Fully qualified class path (e.g. 'monai.metrics.DiceMetric')",
    )
    params: dict = Field(default_factory=dict, description="Constructor kwargs")
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of metric param → source ('state.<key>' or 'batch.<key>')",
    )


class MultiEvalSchema(BaseModel):
    """Evaluation configuration for the multi-stage pipeline."""

    model_config = {"frozen": True, "extra": "forbid"}

    sliding_window: SlidingWindowSchema | None = Field(
        default=None,
        description="Sliding window inference config for 3D validation",
    )
    metrics: dict[str, MultiEvalMetricSchema] = Field(
        default_factory=dict,
        description="Named evaluation metrics (e.g. dice_score, ssim_3d)",
    )


# -----------------------------------------------------------------------
# Top-level multi training configuration
# -----------------------------------------------------------------------


class MultiTrainingConfigSchema(BaseModel):
    """Top-level multi-stage training configuration.

    Supports three execution modes (in priority order):

    1. ``routines`` — named execution flows (train, generate, evaluate)
    2. ``flow`` — flat action list (auto-converted to ``train`` routine)
    3. ``routing`` — legacy sequential (auto-converted to ``train`` routine)

    Example (routines)::

        multi:
          stages:
            - name: unet
              model_type: standard_unet
          routines:
            train:
              - action: call
                node: unet
                inputs: { x: "batch.input" }
                outputs: { prediction: "output" }
            generate:
              - action: assign
                target: "state.x"
                source: "batch.noise"
              - action: loop
                iterator: "batch.steps"
                loop_var: "t"
                body:
                  - action: call
                    node: unet
                    inputs: { x: "state.x", t: "loop_context.t" }
                    outputs: { x: "output" }
    """

    model_config = {"frozen": True, "extra": "forbid"}

    stages: list[MultiStageSchema] = Field(
        ..., min_length=1, description="Ordered list of multi-stage nodes"
    )
    routines: dict[str, list[ActionSchema]] = Field(
        default_factory=dict,
        description="Named execution flows (e.g. train, generate_3d, evaluate)",
    )
    flow: list[ActionSchema] = Field(
        default_factory=list,
        description="Default flow actions (backward compat; becomes 'train' routine)",
    )
    routing: list[RouteSchema] = Field(
        default_factory=list,
        description="Legacy linear routing (backward compat; ignored if flow/routines set)",
    )
    evaluation: MultiEvalSchema | None = Field(
        default=None,
        description="Evaluation config (sliding window, volumetric metrics)",
    )
    end_to_end_finetune_epoch: int | None = Field(
        default=None,
        ge=1,
        description="Epoch at which all stages are unfrozen for joint fine-tuning",
    )
    evaluate_intermediates: bool = Field(
        default=True,
        description="Compute per-stage metrics during validation",
    )

    @model_validator(mode="after")
    def validate_stage_names_unique(self) -> MultiTrainingConfigSchema:
        """Ensure all stage names are unique."""
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Duplicate stage names found: {set(duplicates)}. "
                "Each stage must have a unique name."
            )
        return self

    @model_validator(mode="after")
    def validate_routing_references(self) -> MultiTrainingConfigSchema:
        """Ensure all routing references point to existing stages."""
        stage_names = {s.name for s in self.stages}
        for route in self.routing:
            if route.from_stage not in stage_names:
                raise ValueError(
                    f"Routing references unknown source stage '{route.from_stage}'. "
                    f"Available stages: {stage_names}"
                )
            if route.to_stage not in stage_names:
                raise ValueError(
                    f"Routing references unknown destination stage '{route.to_stage}'. "
                    f"Available stages: {stage_names}"
                )
        return self

    @model_validator(mode="after")
    def validate_action_node_references(self) -> MultiTrainingConfigSchema:
        """Ensure all 'call' action nodes reference existing stages."""
        stage_names = {s.name for s in self.stages}

        def _check_actions(actions: list[ActionSchema], context: str) -> None:
            for action in actions:
                if action.action == "call" and action.node not in stage_names:
                    raise ValueError(
                        f"{context}: 'call' action references unknown stage "
                        f"'{action.node}'. Available: {stage_names}"
                    )
                if action.action == "loop" and action.body:
                    _check_actions(action.body, f"{context} → loop body")

        # Validate flow actions
        if self.flow:
            _check_actions(self.flow, "flow")

        # Validate routine actions
        for routine_name, actions in self.routines.items():
            _check_actions(actions, f"routine '{routine_name}'")

        return self


# Backward compatibility aliases
PipelineStageSchema = MultiStageSchema
PipelineTrainingConfigSchema = MultiTrainingConfigSchema


__all__ = [
    "ActionSchema",
    "FlowStepSchema",
    "MultiEvalMetricSchema",
    "MultiEvalSchema",
    "MultiStageSchema",
    "MultiTrainingConfigSchema",
    "PipelineStageSchema",
    "PipelineTrainingConfigSchema",
    "RouteSchema",
    "SlidingWindowSchema",
    "TrainingStateSchema",
]
