import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["NCAGenerator"]


@register_model(name="nca", training_mode="reconstruction")
class NCAGenerator(nn.Module, IGenerator):
    """
    Neural Cellular Automata (NCA) Generator (Experiment 110).

    Models MRI reconstruction as a biological growth process.
    Each pixel is a cell that updates based on its neighbors.

    Uses gradient checkpointing to reduce memory consumption from O(T*S)
    to O(sqrt(T)*S) where T=num_steps and S=state size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 16,
        num_steps: int = 32,
        use_checkpointing: bool = True,
        checkpoint_segments: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_channels (int): Description.
            num_steps (int): Description.
            use_checkpointing (bool): Description.
            checkpoint_segments (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_steps = num_steps
        self.use_checkpointing = use_checkpointing
        self.checkpoint_segments = checkpoint_segments

        # State dimension: [R, G, B, A, hidden...]
        # We map input to state, evolve state, then map state to output
        self.state_dim = max(16, hidden_channels)

        # Perception: 3x3 convolution (Sobel-like filters usually learned)
        self.perception = nn.Conv2d(
            self.state_dim,
            self.state_dim * 3,
            kernel_size=3,
            padding=1,
            groups=self.state_dim,
            bias=False,
        )

        # Update rule: 1x1 convolution (MLP per pixel)
        self.update_net = nn.Sequential(
            nn.Conv2d(self.state_dim * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.state_dim, kernel_size=1),
        )

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, self.state_dim, kernel_size=1)

        # Output projection
        self.output_proj = nn.Conv2d(self.state_dim, out_channels, kernel_size=1)

        # Stochastic update mask
        self.register_buffer("update_probability", torch.tensor(0.5))

    def _run_steps(self, state: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Run a segment of NCA steps (used for checkpointing)."""
        for _ in range(num_steps):
            state = self.step(state)
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Seed image [B, C, H, W]

        forward method for NCAGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initialize state
        state = self.input_proj(x)

        # Evolve with optional gradient checkpointing
        if self.use_checkpointing and self.training and self.num_steps > 8:
            # Segment the steps for checkpointing to reduce memory
            steps_per_segment = max(1, self.num_steps // self.checkpoint_segments)
            remaining_steps = self.num_steps

            while remaining_steps > 0:
                current_steps = min(steps_per_segment, remaining_steps)
                # Use checkpoint to avoid storing intermediate activations
                state = checkpoint(
                    self._run_steps,
                    state,
                    current_steps,
                    use_reentrant=False,
                )
                remaining_steps -= current_steps
        else:
            # Standard forward without checkpointing
            for _ in range(self.num_steps):
                state = self.step(state)

        return self.output_proj(state)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        # 1. Perception
        """step.

        Args:
            state (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        perception_grid = self.perception(state)

        # 2. Update rule
        update = self.update_net(perception_grid)

        # 3. Stochastic update (asynchronous CA)
        # Random mask: each cell updates with probability 0.5
        mask = (torch.rand_like(state[:, :1, :, :]) < self.update_probability).float()

        return state + mask * update

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
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "nca_generator"
