"""Surgery Loop Agent for Active Intervention

RL agent that plans surgical interventions based on intraoperative MRI.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class StateEncoder(nn.Module):
    """Encodes current MRI state."""

    def __init__(self, in_channels: int = 1, state_dim: int = 256):
        """__init__.

        Args:
            in_channels (int): Description.
            state_dim (int): Description.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for StateEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = self.encoder(x)
        features = features.flatten(1)
        return self.fc(features)


class ActionPredictor(nn.Module):
    """Predicts optimal intervention action."""

    def __init__(self, state_dim: int = 256, action_dim: int = 10):
        """__init__.

        Args:
            state_dim (int): Description.
            action_dim (int): Description.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            state (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ActionPredictor.

        Executes PyTorch tensor operations.

        Args:
            state (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.network(state)


class RewardModel(nn.Module):
    """Predicts outcome/reward of action."""

    def __init__(self, state_dim: int = 256, action_dim: int = 10):
        """__init__.

        Args:
            state_dim (int): Description.
            action_dim (int): Description.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            state (torch.Tensor): Description.
            action (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RewardModel.

        Executes PyTorch tensor operations.

        Args:
            state (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            action (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        combined = torch.cat([state, action], dim=-1)
        return self.network(combined)


@register_model(name="surgery_loop_agent", training_mode="reconstruction")
class SurgeryLoopAgent(nn.Module, IGenerator):
    """Active intervention agent for surgical planning.

    Uses MRI feedback to plan optimal surgical interventions in real-time.
    Implements a model-based RL approach with learned dynamics.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        state_dim: int = 256,
        action_dim: int = 10,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            state_dim (int): Description.
            action_dim (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Components
        self.state_encoder = StateEncoder(in_channels, state_dim)
        self.action_predictor = ActionPredictor(state_dim, action_dim)
        self.reward_model = RewardModel(state_dim, action_dim)

        # Outcome predictor (simulates MRI after intervention)
        self.outcome_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 16 * 16),
            nn.ReLU(inplace=True),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode_state(self, mri_image: torch.Tensor) -> torch.Tensor:
        """Encode current MRI into state representation."""
        return self.state_encoder(mri_image)

    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        """Predict optimal intervention action."""
        return self.action_predictor(state)

    def predict_outcome(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict MRI after intervention."""
        B = state.shape[0]
        combined = torch.cat([state, action], dim=-1)
        features = self.outcome_predictor(combined)
        features = features.view(B, 64, 16, 16)
        return self.spatial_decoder(features)

    def forward(self, x: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """End-to-end: encode -> predict action -> simulate outcome.

        forward method for SurgeryLoopAgent.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            action (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        state = self.encode_state(x)

        if action is None:
            action = self.predict_action(state)

        outcome = self.predict_outcome(state, action)
        return outcome

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """IGenerator implementation: generate output from input."""
        return self.forward(x, **kwargs)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "surgery_loop_agent"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape
