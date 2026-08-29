import torch
import torch.nn as nn

from mriforge.infrastructure.physics.dynamics.neural_ode import NeuralODEDynamics
from mriforge.infrastructure.physics.implementations.gaussian_fourier_op import (
    TemporalGaussianFourierProjector,
)
from mriforge.models.components.siren import DivergenceFreeSIREN
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(name="lans", training_mode="reconstruction")
class LaNSGenerator(nn.Module, IGenerator):
    """
    Lagrangian Neuro-Splatting (LaNS) Generator.

    Pipeline:
    1. Initialize BiophysicalGaussians (Positions, Scales, Features).
    2. Predict Velocity Field (SIREN).
    3. Advect Gaussians via NeuralODE.
    4. Project moving Gaussians to K-space over time.
    """

    def __init__(
        self,
        num_gaussians: int = 1000,
        temp_dim: int = 20,  # Number of timeframes
        in_channels: int = 2,
        out_channels: int = 2,
        siren_hidden_dim: int = 64,
        siren_layers: int = 3,
        dt: float = 0.1,
    ):
        """__init__.

        Args:
            num_gaussians (int): Description.
            temp_dim (int): Description.
            in_channels (int): Description.
            out_channels (int): Description.
            siren_hidden_dim (int): Description.
            siren_layers (int): Description.
            dt (float): Description.
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.temp_dim = temp_dim
        self.out_channels = out_channels

        # 1. Initialize Gaussians (Normally handled by PAGS, but here we can learn them or start random)
        # For simplicity, learnable parameters for now.
        self.means = nn.Parameter(torch.rand(1, num_gaussians, 3) * 2 - 1)  # [-1, 1]
        self.scales = nn.Parameter(torch.ones(1, num_gaussians, 3) * -2.0)  # log scale
        self.quats = nn.Parameter(torch.rand(1, num_gaussians, 4))
        self.features = nn.Parameter(torch.randn(1, num_gaussians, out_channels))
        self.opacity = nn.Parameter(torch.zeros(1, num_gaussians, 1))  # Logit

        # 2. Velocity Network
        self.velocity_net = DivergenceFreeSIREN(
            in_features=4,  # x,y,z,t
            hidden_features=siren_hidden_dim,
            hidden_layers=siren_layers,
            out_features=3,
        )

        # 3. ODE Solver
        # Wrapper to match ODE signature
        def dynamics(y, t):
            # y: (B, N, 3) positions
            # t: scalar time

            # Prepare inputs for SIREN: (B, N, 4) -> (x,y,z,t)
            """dynamics.

            Args:
                y (Any): Description.
                t (Any): Description.
            Returns:
                Any: Description.
            """
            B, N, _ = y.shape

        # The drift_net for NeuralODEDynamics should take (t, y) and return dy/dt.
        # Our velocity_net takes (x,y,z,t) as a single input.
        # We need a wrapper to adapt velocity_net to the drift_net signature.
        class DriftNetWrapper(nn.Module):
            """DriftNetWrapper class."""

            def __init__(self, velocity_net):
                """__init__.

                Args:
                    velocity_net (Any): Description.
                """
                super().__init__()
                self.velocity_net = velocity_net

            def forward(self, t, y):
                # y: (B, N, 3) positions
                # t: scalar time

                # Prepare inputs for SIREN: (B, N, 4) -> (x,y,z,t)
                """forward.

                        Args:
                            t (Any): Description.
                            y (Any): Description.
                        Returns:
                            Any: Description.

                forward method for DriftNetWrapper.

                Executes PyTorch tensor operations.

                Args:
                    t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
                    y (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                B, N, _ = y.shape
                t_expand = torch.ones(B, N, 1, device=y.device) * t
                coords = torch.cat([y, t_expand], dim=-1)

                vel, _ = self.velocity_net(coords)
                return vel

        self.drift_net = DriftNetWrapper(self.velocity_net)
        self.ode_dynamics = NeuralODEDynamics(drift_net=self.drift_net, solver="dopri5")

        # 4. Projector
        self.projector = TemporalGaussianFourierProjector()

    def forward(
        self,
        x: torch.Tensor | None = None,  # Input image conditioning (optional)
        k_trajectory: torch.Tensor | None = None,
        k_time_indices: torch.Tensor | None = None,
        output_type: str = "kspace",
    ) -> torch.Tensor:
        # 1. Generate Trajectory
        # Time span 0 to temp_dim * dt? Or normalized 0 to 1?
        # Let's say t goes from 0 to 1.
        """forward.

        Args:
            x (Optional[torch.Tensor]): Description.
            k_trajectory (Optional[torch.Tensor]): Description.
            k_time_indices (Optional[torch.Tensor]): Description.
            output_type (str): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LaNSGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_time_indices (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            output_type (str): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        times = torch.linspace(0, 1, self.temp_dim, device=self.means.device)

        # (B, T, N, 3)
        means_trajectory = self.ode_solver(self.means, times)

        if output_type == "trajectory":
            return means_trajectory

        if output_type == "kspace" and k_trajectory is not None and k_time_indices is not None:
            # Scales/Quats/Feats static
            scales = torch.exp(self.scales)
            quats = torch.nn.functional.normalize(self.quats, dim=-1)

            # Project
            k_data = self.projector(
                means_trajectory,
                scales,
                quats,
                self.features,
                k_trajectory,
                k_time_indices,
            )
            return k_data

        return means_trajectory

    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            noise (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "LaNSGenerator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape
