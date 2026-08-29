import logging
from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.inference.base_inference_strategy import BaseInferenceStrategy
from mriforge.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)

logger = logging.getLogger(__name__)


class MultiInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for multi-stage pipelines.

    Delegates inference execution to the interpreter's `_execute_flow` or
    `_execute_sequential` mechanisms.
    """

    def __init__(
        self,
        strategy: MultiTrainingStrategy,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        # BaseInferenceStrategy expects a model (nn.Module).
        # MultiTrainingStrategy is not an nn.Module, so we pass None
        # and override methods to bypass direct model usage.
        # But to satisfy typing or base access, we can pass strategy.generator_model if it exists,
        # or a dummy module.
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, device=device, config=config)
        self.multi_strategy = strategy

    def infer_single(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass using the multi-stage strategy.

        This mimics the logic in MultiTrainingStrategy._apply_sliding_window
        when not using sliding window, or delegates to it.
        """
        # We rely on the MultiTrainingStrategy to do the heavy lifting.
        # It has its own _apply_sliding_window logic within validation_step,
        # but here we can just use the routines.

        batch_data = {
            "input": input_tensor,
            # We don't have targets during pure inference
            "target": torch.zeros_like(input_tensor),
            "timesteps": torch.zeros(
                input_tensor.shape[0], device=self.device
            ),  # for diffusion/etc
        }

        state: dict[str, Any] = {}

        if self.multi_strategy._use_routines:
            routine_actions = self.multi_strategy.routines.get(
                "eval"
            ) or self.multi_strategy.routines.get("train", [])
            if not routine_actions:
                raise ValueError("MultiTrainingStrategy has no valid routine for inference")

            self.multi_strategy._execute_flow(
                actions=routine_actions,
                batch_data=batch_data,
                state=state,
                training=False,
            )

            # The final output is usually the result of the last action's outputs or state assignment.
            # Look for common terminal stage keys or just last computed state.
            if len(state) > 0:
                return list(state.values())[-1]
            return input_tensor
        else:
            # Sequential fallback
            output, _ = self.multi_strategy._execute_sequential(input_tensor, training=False)
            return output
