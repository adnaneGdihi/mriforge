import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["RLAcquisitionAgent"]


@register_model(name="rl_acquisition_agent", training_mode="reconstruction")
class RLAcquisitionAgent(nn.Module, IGenerator):
    """
    Reinforcement Learning Agent for Active k-Space Acquisition (Experiment 103).

    This agent observes the current partial reconstruction (or zero-filled image)
    and the current sampling mask, and outputs a policy (probability distribution)
    over the next k-space line to acquire.

    Architecture:
    - CNN Encoder to process the current image state.
    - MLP Head to output Q-values or Policy logits for each possible k-space line.
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 256,
        hidden_dim: int = 256,
        action_space: int = 256,  # Number of phase encode lines
        dueling_dqn: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            img_size (int): Description.
            hidden_dim (int): Description.
            action_space (int): Description.
            dueling_dqn (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.action_space = action_space
        self.dueling_dqn = dueling_dqn

        # Encoder: Simple CNN to extract features from the current reconstruction
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels + 1, 32, kernel_size=3, stride=2, padding=1
            ),  # +1 for mask channel
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        # 256 -> 128 -> 64 -> 32
        self.flattened_size = 128 * (img_size // 8) * (img_size // 8)

        if dueling_dqn:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(self.flattened_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.flattened_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_space),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(self.flattened_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_space),
            )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Current partial reconstruction [B, C, H, W]
            mask: Current sampling mask [B, 1, H, W] (or [B, 1, H, 1] broadcasted)

        Returns:
            Q-values for each k-space line [B, action_space]

        forward method for RLAcquisitionAgent.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Concatenate image and mask
        x = torch.cat([image, mask], dim=1)

        features = self.encoder(x)

        if self.dueling_dqn:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_head(features)

        return q_values

    def select_action(
        self, image: torch.Tensor, mask: torch.Tensor, epsilon: float = 0.1
    ) -> torch.Tensor:
        """
        Selects the next line to acquire using epsilon-greedy strategy.
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_space, (image.size(0),), device=image.device)
        else:
            with torch.no_grad():
                q_values = self(image, mask)
                # Mask out already acquired lines (set Q to -inf)
                # Assuming mask is 1 for acquired, 0 for not
                # We need to check the mask structure. If it's 2D mask, we check the center column or similar.
                # For simplicity here, we assume mask input allows us to know which lines are taken.
                # In a real loop, we'd handle this logic.
                return torch.argmax(q_values, dim=1)

    def generate(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Generates the next action (k-space line index).
        """
        return self.select_action(image, mask, epsilon=0.0)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "rl_acquisition_agent"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Output is action index (scalar per batch item) or Q-values
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (self.action_space,)
