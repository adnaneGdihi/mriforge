"""q-Space Translation: Physics-Faithful Upscaling
===============================================

Implementation of Experiment 2.2 from the Research Roadmap.
Transforms VLF data to HF data by operating in the quantitative domain (q-space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.integration import BlochEquationSolver
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.reconstruction.unet import StandardUNetGenerator as StandardUNet
from mriforge.models.registry import register_model


@register_model(name="qspace_translation", training_mode="reconstruction")
class qSpaceTranslation(nn.Module, IGenerator):
    """q-Space Translation Model."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        q_map_channels: int = 2,  # T1, T2 (PD is usually assumed constant or estimated)
        b0_low: float = 0.064,  # 64mT
        b0_high: float = 3.0,  # 3T
        beta: float = 0.4,  # Empirical constant for T1 scaling
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            q_map_channels (int): Description.
            b0_low (float): Description.
            b0_high (float): Description.
            beta (float): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.b0_low = b0_low
        self.b0_high = b0_high
        self.beta = beta

        # 1. VLF q-Map Estimator
        # Estimates T1_low, T2_low from VLF image
        self.estimator = StandardUNet(
            in_channels=in_channels,
            out_channels=q_map_channels + 1,  # T1, T2, PD
            bilinear=False,
        )

        # 2. Bloch Decoder
        self.bloch_solver = BlochEquationSolver()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "qSpaceTranslation"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.estimator.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from VLF input."""
        return self.forward(z, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        This implements the full pipeline:
        VLF Image -> q-maps (Low Field) -> q-maps (High Field) -> HF Image

        forward method for qSpaceTranslation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Estimate Low-Field q-maps
        q_maps_low = self.estimator(x)

        # Split channels: T1, T2, PD
        # We assume output range is unconstrained logits, so we map to positive values
        t1_low = F.softplus(q_maps_low[:, 0:1]) * 2000.0  # Scale to ms range
        t2_low = F.softplus(q_maps_low[:, 1:2]) * 500.0
        pd = F.softplus(q_maps_low[:, 2:3])  # PD is field-independent

        # 2. Physics-Based Transformation
        # T1_high = T1_low * (B0_high / B0_low)^beta
        scaling_factor = (self.b0_high / self.b0_low) ** self.beta
        t1_high = t1_low * scaling_factor

        # T2 is largely field-independent
        # TODO: Correct T2 field dependence (relaxation mechanisms)
        t2_high = t2_low

        # 3. HF Synthesis via Bloch Simulator
        # We need sequence parameters for the target HF acquisition.
        # Since IGenerator.forward signature is fixed, we use default 3T params
        # or we could accept them if we change signature.
        # Default 3T T1w: TR=500, TE=10
        # TODO: Use protocol inputs from kwargs instead of hardcoded defaults

        # Construct m0
        m0 = torch.zeros(x.shape[0], x.shape[2], x.shape[3], 3, device=x.device)
        m0[..., 2] = pd.squeeze(1)

        self.bloch_solver.tr = 500.0
        self.bloch_solver.te = 10.0
        self.bloch_solver.t1 = t1_high.squeeze(1)
        self.bloch_solver.t2 = t2_high.squeeze(1)

        signal = self.bloch_solver(m0)
        magnitude = torch.sqrt(signal[..., 0] ** 2 + signal[..., 1] ** 2)

        return magnitude.unsqueeze(1)
