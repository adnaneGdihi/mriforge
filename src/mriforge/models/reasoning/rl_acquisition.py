"""
RL-MRI: Diagnosis-Guided Active Acquisition.

Implements an RL agent that selects k-space trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class MRIEnv:
    """
    Simulated MRI Environment for Active Acquisition.

    State: Mask (which lines acquired) + Current Recon
    Action: Index of next line to acquire (0 to H-1)
    Reward: Improvement in SSIM/MSE or Diagnostic Confidence.
    """

    def __init__(self, shape=(32, 32), max_steps=10):
        """__init__.

        Args:
            shape (Any): Description.
            max_steps (Any): Description.
        """
        self.shape = shape
        self.max_steps = max_steps
        self.current_step = 0
        self.mask = torch.zeros(shape[0])  # 1D mask for Phase Encoding
        # Ground truth would be loaded here in real env
        self.ground_truth = torch.randn(1, 1, *shape)  # Dummy

    def reset(self) -> torch.Tensor:
        """reset.

        Returns:
            torch.Tensor: Description.
        """
        self.current_step = 0
        self.mask = torch.zeros(self.shape[0])
        self.ground_truth = torch.randn(1, 1, *self.shape)  # New sample
        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        # Observation: Current Mask state
        # In real RL, would return zero-filled recon + mask
        """_get_obs.

        Returns:
            torch.Tensor: Description.
        """
        return self.mask.clone()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """
        Acquire line 'action'.
        """
        # Update mask
        if self.mask[action] == 0:
            reward = 1.0  # Acquired new info!
            self.mask[action] = 1.0
        else:
            reward = -0.1  # Penalty for re-acquiring

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done


@register_model(name="rl_mri", training_mode="acquisition")
class ActiveScanner(nn.Module):
    """
    Policy Network for Active Acquisition.
    Uses Mamba/LSTM to process acquisition history?
    Simple Policy: Takes Mask State -> Logits over lines.
    """

    def __init__(self, num_lines: int = 32):
        """__init__.

        Args:
            num_lines (int): Description.
        """
        super().__init__()
        self.num_lines = num_lines

        # Policy Network
        # Input: Mask state [H]
        # Output: Action Logits [H]
        self.net = nn.Sequential(nn.Linear(num_lines, 64), nn.ReLU(), nn.Linear(64, num_lines))

    def forward(self, mask_state: torch.Tensor) -> torch.Tensor:
        """
        Predict logits for next line to acquire.
        Prioritize unacquired lines? The net should learn this.

        forward method for ActiveScanner.

        Executes PyTorch tensor operations.

        Args:
            mask_state (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(mask_state)

    def select_action(self, mask_state: torch.Tensor) -> int:
        """select_action.

        Args:
            mask_state (torch.Tensor): Description.
        Returns:
            int: Description.
        """
        logits = self(mask_state)
        # Mask out already acquired lines actions to force exploration?
        # Standard RL would let it learn, but we can heuristic mask.
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action


def create_rl_mri(**kwargs):
    """create_rl_mri.

    Returns:
        Any: Description.
    """
    return ActiveScanner(**kwargs)
