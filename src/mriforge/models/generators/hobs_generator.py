import torch
import torch.nn as nn

from mriforge.infrastructure.physics.biophysics.bloch import BlochSimulator
from mriforge.infrastructure.physics.implementations.gaussian_fourier_op import (
    GaussianCloud,
    GaussianFourierProjector,
)
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(name="hobs", training_mode="reconstruction")
class HoBSGenerator(nn.Module, IGenerator):
    """
    Holographic Bloch-Splatting (HoBS) Generator.

    Optimizes a "Biophysical Cloud" where each Gaussian has (M0, T1, T2).
    Generates k-space data for arbitrary pulse sequences by simulating the physics on-the-fly.
    """

    def __init__(
        self,
        num_gaussians: int = 1000,
        temp_dim: int = 1,  # HoBS usually static anatomy
        in_channels: int = 2,
        out_channels: int = 2,  # Complex k-space
        _learnable_biophysics: bool = True,
    ):
        """__init__.

        Args:
            num_gaussians (int): Description.
            temp_dim (int): Description.
            in_channels (int): Description.
            out_channels (int): Description.
            _learnable_biophysics (bool): Description.
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_channels = out_channels

        # 1. Initialize Biophysical Gaussians
        # Means, Scales, Quats (Geometry)
        self.means = nn.Parameter(torch.rand(1, num_gaussians, 3) * 2 - 1)
        self.scales = nn.Parameter(torch.ones(1, num_gaussians, 3) * -2.0)
        self.quats = nn.Parameter(torch.rand(1, num_gaussians, 4))
        self.opacity = nn.Parameter(torch.zeros(1, num_gaussians, 1))

        # Biophysics: M0, T1, T2
        # T1, T2 need to be positive. We optimize in log space or sigmoid space.
        # Let's use raw and enforce positivity in forward.
        # Initializing close to typical brain values?
        # T1 ~ 1000ms, T2 ~ 100ms. Normalize?
        # For prototype, we assume normalized values or learnable direct.
        self.biophysics = nn.Parameter(torch.rand(1, num_gaussians, 3))

        self.bloch = BlochSimulator()
        self.projector = GaussianFourierProjector()

    def forward(
        self,
        x: torch.Tensor | None = None,
        k_trajectory: torch.Tensor | None = None,
        output_type: str = "kspace",
        # Physics Params passed in kwargs or config?
        # For training, we need to pass them per batch item or global.
        # We'll expect them in a 'physics_params' dict in kwargs via the pipeline,
        # but standard signature doesn't have it.
        # We can accept **kwargs.
        **kwargs,
    ) -> torch.Tensor:
        # Extract physics params
        # Default to Spin Echo with some dummy vals if not provided
        # TODO: Enforce strict sequence parameter validation
        """forward.

        Args:
            x (Optional[torch.Tensor]): Description.
            k_trajectory (Optional[torch.Tensor]): Description.
            output_type (str): Description.
        Returns:
            torch.Tensor: Description.

        forward method for HoBSGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            output_type (str): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        sequence_type = kwargs.get("sequence_type", "spin_echo")
        te = kwargs.get("te", 10.0)
        tr = kwargs.get("tr", 1000.0)

        # 1. Form Biophysical Cloud
        # Enforce constraints
        scales = torch.exp(self.scales)
        quats = torch.nn.functional.normalize(self.quats, dim=-1)

        m0 = torch.sigmoid(self.biophysics[..., 0:1])  # 0-1 density
        t1 = torch.abs(self.biophysics[..., 1:2]) * 2000.0 + 10.0  # Map to ms range
        t2 = torch.abs(self.biophysics[..., 2:3]) * 500.0 + 5.0  # Map to ms range

        # 2. Bloch Simulation -> Signal Intensity (Features)
        if sequence_type == "spin_echo":
            signal = self.bloch.spin_echo(m0, t1, t2, te, tr)
        elif sequence_type == "inversion_recovery":
            ti = kwargs.get("ti", 500.0)
            signal = self.bloch.inversion_recovery(m0, t1, ti, tr)
        else:
            raise ValueError(
                f"Unknown sequence_type '{sequence_type}'. "
                "Valid options: 'spin_echo', 'inversion_recovery'."
            )

        # Signal is (B, N, 1). We need complex?
        # DFGP expects features (B, N, C). Output is (B, M, C).
        # We usually model Magnitude MRI here for simplicity in T1/T2 mapping first?
        # Or we assume phase is zero (Flat phase).
        # To output complex k-space, features should be complex?
        # Or features = real magnitude, and phase comes from Position?
        # DFGP math: sum c_j * exp(...)
        # If c_j is real, we effectively assign 0 phase to local magnetization.
        # Which is fine for magnitude T1/T2 mapping.

        # Expand signal to match out_channels if needed (e.g. 2 for real/imag or 1 for complex)
        # DFGP takes (N, C).
        # If out_channels=1 (Complex), signal is (N, 1).

        features = signal

        # 3. Project
        if output_type == "kspace" and k_trajectory is not None:
            cloud = GaussianCloud(
                means=self.means.squeeze(0),  # DFGP expects (N, 3) for now?
                # Wait, DFGP wrapper (GaussianFourierProjector) handles Forward call
                # which delegates to differentiable_gaussian_fourier using Cloud attributes.
                # Projector.forward signature: (cloud, k_traj).
                # But our Cloud struct is usually single-batch?
                # The Cloud dataclass has simple tensor containers.
                # Let's check GaussianCloud definition.
                # It accepts (means, scales...), shapes (N, 3).
                # Implementation assumes single cloud per call or batch dimension?
                # gaussian_fourier_op.py says (N, 3).
                # So we must loop over batch if B > 1.
                # For prototype B=1.
                scales=scales.squeeze(0),
                quaternions=quats.squeeze(0),
                features=features.squeeze(0),
                opacity=self.opacity.squeeze(0),
            )

            k_data = self.projector(cloud, k_trajectory)

            # k_data is (M, C). Add batch dim -> (1, M, C)
            return k_data.unsqueeze(0)

        # If output_type="biophysics", return the maps
        if output_type == "biophysics":
            return torch.cat([m0, t1, t2], dim=-1)

        return signal

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
        return "HoBSGenerator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape
