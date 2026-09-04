"""Digital Twin Simulator

Creates patient-specific digital twins that can simulate MRI responses
and predict outcomes for different scanning protocols or interventions.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["DigitalTwinSimulator", "PatientEncoder"]


class PatientEncoder(nn.Module):
    """Encodes patient-specific characteristics into an embedding."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        """__init__.

        Args:
            in_channels (int): Description.
            embedding_dim (int): Description.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PatientEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = self.network(x)
        features = features.flatten(1)
        return self.fc(features)


class PhysicsSimulator(nn.Module):
    """Simulates MRI physics given patient embedding and protocol parameters.

    Combines learned features with Bloch equation constraints for physics-informed
    simulation. Protocol params expected format: [TE, TR, flip_angle, ...rest]
    """

    def __init__(self, patient_dim: int = 128, protocol_dim: int = 16, hidden_dim: int = 256):
        """__init__.

        Args:
            patient_dim (int): Description.
            protocol_dim (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.protocol_embed = nn.Linear(protocol_dim, hidden_dim // 2)
        self.patient_proj = nn.Linear(patient_dim, hidden_dim // 2)

        # Tissue parameter predictor (T1, T2, M0) from patient embedding
        self.tissue_predictor = nn.Sequential(
            nn.Linear(patient_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # [T1, T2, M0] in normalized space
            nn.Softplus(),  # Ensure positive values
        )

        self.simulator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def bloch_signal(
        self,
        M0: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        TE: torch.Tensor,
        TR: torch.Tensor,
    ) -> torch.Tensor:
        """Steady-state Bloch equation for GRE sequence.

        Signal = M0 * (1 - exp(-TR/T1)) * exp(-TE/T2)
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        E1 = torch.exp(-TR / (T1 + eps))
        E2 = torch.exp(-TE / (T2 + eps))
        return M0 * (1 - E1) * E2

    def forward(
        self, patient_embedding: torch.Tensor, protocol_params: torch.Tensor
    ) -> torch.Tensor:
        """Simulate response given patient and protocol.

        forward method for PhysicsSimulator.

        Executes PyTorch tensor operations.

        Args:
            patient_embedding (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            protocol_params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Predict tissue parameters
        tissue_params = self.tissue_predictor(patient_embedding)
        T1 = tissue_params[..., 0:1] * 1000 + 500  # Scale to ~500-1500ms
        T2 = tissue_params[..., 1:2] * 100 + 20  # Scale to ~20-120ms
        M0 = tissue_params[..., 2:3]  # Proton density

        # Extract sequence parameters (first 2 channels: TE, TR in ms)
        TE = (protocol_params[..., 0:1] + 1) * 15  # Scale to ~0-30ms
        TR = (protocol_params[..., 1:2] + 1) * 250 + 250  # Scale to ~250-750ms

        # Physics-constrained signal modulation
        physics_signal = self.bloch_signal(M0, T1, T2, TE, TR)

        # Learned features
        patient_feat = self.patient_proj(patient_embedding)
        protocol_feat = self.protocol_embed(protocol_params)
        combined = torch.cat([patient_feat, protocol_feat], dim=-1)
        learned_features = self.simulator(combined)

        # Combine physics + learned (residual connection)
        return learned_features * physics_signal.expand_as(learned_features[:, :1]).repeat(
            1, learned_features.size(1)
        )


@register_model(name="digital_twin_simulator", training_mode="reconstruction")
class DigitalTwinSimulator(nn.Module, IGenerator):
    """Digital Twin for patient-specific MRI simulation.

    Creates a virtual representation of a patient's anatomy and physiology
    that can predict MRI responses under different acquisition protocols.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        patient_embedding_dim: int = 128,
        protocol_dim: int = 16,
        hidden_dim: int = 256,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            patient_embedding_dim (int): Description.
            protocol_dim (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patient-specific encoder
        self.patient_encoder = PatientEncoder(in_channels, patient_embedding_dim)

        # Physics-based simulator
        self.physics_simulator = PhysicsSimulator(patient_embedding_dim, protocol_dim, hidden_dim)

        # Image decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 16 * 16),
            nn.ReLU(inplace=True),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def create_twin(self, patient_image: torch.Tensor) -> torch.Tensor:
        """Create patient-specific embedding (digital twin)."""
        return self.patient_encoder(patient_image)

    def simulate(
        self,
        patient_embedding: torch.Tensor,
        protocol_params: torch.Tensor,
        spatial_size: tuple[int, int] = (128, 128),
    ) -> torch.Tensor:
        """Simulate MRI response for given protocol."""
        # Physics simulation
        physics_features = self.physics_simulator(patient_embedding, protocol_params)

        # Decode to image space
        B = physics_features.shape[0]
        spatial_features = self.image_decoder(physics_features)
        spatial_features = spatial_features.view(B, 64, 16, 16)

        return self.spatial_decoder(spatial_features)

    def forward(self, x: torch.Tensor, protocol_params: torch.Tensor | None = None) -> torch.Tensor:
        """End-to-end: create twin and simulate with default protocol.

        forward method for DigitalTwinSimulator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            protocol_params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        patient_embedding = self.create_twin(x)

        # Default protocol if not provided
        if protocol_params is None:
            B = x.shape[0]
            protocol_params = torch.zeros(B, 16, device=x.device)

        return self.simulate(patient_embedding, protocol_params)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "digital_twin_simulator"

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
