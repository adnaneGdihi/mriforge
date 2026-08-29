import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["GFlowNetGenerator"]


@register_model(name="gflownet", training_mode="reconstruction")
class GFlowNetGenerator(nn.Module, IGenerator):
    """
    Generative Flow Network (GFlowNet) for MRI sampling mask generation (Experiment 111).

    GFlowNets learn a stochastic policy to sample objects (masks) with probability proportional
    to a reward function. Here, the reward is the reconstruction quality.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 64,
        img_size: int = 128,
        num_steps: int = 10,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_dim (int): Description.
            img_size (int): Description.
            num_steps (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.num_steps = num_steps

        # Policy network: State -> Action logits
        # Split into context encoder (static) and mask encoder (dynamic)
        # to avoid repeated concatenation and processing of static context

        # 1. Context Encoder (Static)
        self.context_encoder = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)

        # 2. Mask Encoder (Dynamic)
        self.mask_encoder = nn.Conv2d(1, hidden_dim, 3, padding=1)

        # 3. Policy Head (Shared)
        self.policy_head = nn.Sequential(
            nn.ReLU(),  # Activation after fusion
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),  # Logits for each pixel
        )

        # Flow network (for detailed balance / trajectory balance)
        # Estimating Z (partition function) or F(s)
        self.flow_net = nn.Sequential(
            nn.Linear(hidden_dim, 1)  # Simple scalar flow estimation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass generates a sampling mask.
        Args:
            x: Input image context [B, C, H, W]
        Returns:
            Generated mask [B, 1, H, W] (soft or hard)

        forward method for GFlowNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Initialize empty mask
        mask = torch.zeros((B, 1, H, W), device=x.device)

        # 1. Encode Context ONCE (Pre-computation)
        # This moves the heavy lifting out of the loop
        ctx_features = self.context_encoder(x)  # [B, hidden_dim, H, W]

        # Iterative generation
        for _ in range(self.num_steps):
            # 2. Encode Mask (Lightweight)
            mask_features = self.mask_encoder(mask)

            # 3. Fuse (Add instead of Cat)
            # Addition is cheaper than Cat + Conv on larger channels
            features = ctx_features + mask_features

            # Predict logits for adding to mask
            logits = self.policy_head(features)

            # Sample action (Bernoulli for simplicity, or categorical)
            # Here we use Gumbel-Softmax or simple sigmoid for differentiability during training if needed
            # For GFlowNet training, we usually sample actions.

            # Update mask (additive)
            # Update mask (additive/probabilistic)
            # Use differentiable update: m_new = m_old + update * (1 - m_old)
            # This allows gradients to flow back to the logits
            update = torch.sigmoid(logits)
            mask = mask + update * (1 - mask)

        return mask

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mask (alias for forward)."""
        return self.forward(x)

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

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "gflownet_generator"
