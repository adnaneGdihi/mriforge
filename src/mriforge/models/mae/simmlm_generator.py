"""SimMLM: Similarity-aware Masked Language Model with Dynamic Mixture of Experts.

Implements a dynamic expert gating mechanism for multi-modal MRI synthesis,
enabling robust imputation of missing modalities.

Key Features:
- Modality-specific expert encoders
- Dynamic gating based on input availability
- Uncertainty-aware synthesis
- Robust to corrupted inputs

Reference:
- SimMLM: Similarity-aware Masked Language Model for Multi-modal Medical Image Synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class ModalityExpert(nn.Module):
    """Expert encoder for a single modality.

    Each expert specializes in extracting features from its
    assigned modality (T1, T2, FLAIR, DWI, etc.).

    Args:
        in_channels: Input channels for this modality
        hidden_dim: Hidden feature dimension
        num_blocks: Number of residual blocks
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_dim (int): Description.
            num_blocks (int): Description.
        """
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                )
            )

        # Feature projection
        self.proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract expert features.

        Args:
            x: [B, C, H, W] input modality

        Returns:
            [B, D, H, W] expert features

        forward method for ModalityExpert.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.input_conv(x)

        for block in self.blocks:
            h = h + block(h)

        return self.proj(h)


class DynamicGatingNetwork(nn.Module):
    """Dynamic gating network for expert selection.

    Analyzes available inputs to determine expert weights,
    downweighting corrupted or missing modalities.

    Args:
        num_experts: Number of modality experts
        hidden_dim: Feature dimension
    """

    def __init__(
        self,
        num_experts: int = 4,
        hidden_dim: int = 128,
    ):
        """__init__.

        Args:
            num_experts (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()

        self.num_experts = num_experts

        # Quality estimator per expert
        self.quality_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
                for _ in range(num_experts)
            ]
        )

        # Cross-modal attention for gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_experts, num_experts * 2),
            nn.ReLU(),
            nn.Linear(num_experts * 2, num_experts),
        )

    def forward(
        self,
        expert_features: list[torch.Tensor],
        availability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dynamic gating weights.

        Args:
            expert_features: List of [B, D, H, W] features per expert
            availability_mask: [B, E] binary mask (1=available, 0=missing)

        Returns:
            [B, E] gating weights (sum to 1)

        forward method for DynamicGatingNetwork.

        Executes PyTorch tensor operations.

        Args:
            expert_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            availability_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = expert_features[0].shape[0]
        device = expert_features[0].device

        # Estimate quality of each expert's input
        quality_scores = []
        for i, (feat, qnet) in enumerate(zip(expert_features, self.quality_nets, strict=False)):
            score = qnet(feat)  # [B, 1]
            quality_scores.append(score)

        quality = torch.cat(quality_scores, dim=1)  # [B, E]

        # Apply availability mask
        if availability_mask is not None:
            quality = quality * availability_mask + (1 - availability_mask) * (-1e9)

        # Compute gates via MLP + softmax
        gate_logits = self.gate_mlp(quality)
        gates = F.softmax(gate_logits, dim=1)

        return gates


class UncertaintyEstimator(nn.Module):
    """Epistemic uncertainty estimator via dropout.

    Provides uncertainty maps for synthesized modalities
    to warn clinicians about unreliable regions.

    Args:
        hidden_dim: Feature dimension
    """

    def __init__(self, hidden_dim: int = 128, dropout_rate: float = 0.1):
        """__init__.

        Args:
            hidden_dim (int): Description.
            dropout_rate (float): Description.
        """
        super().__init__()

        self.dropout = nn.Dropout2d(dropout_rate)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(),  # Positive uncertainty
        )

    def forward(
        self,
        features: torch.Tensor,
        num_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty via MC dropout.

        Args:
            features: [B, D, H, W] fused features
            num_samples: Number of MC samples

        Returns:
            mean: [B, 1, H, W] uncertainty map (mean)
            var: [B, 1, H, W] uncertainty map (variance)

        forward method for UncertaintyEstimator.

        Executes PyTorch tensor operations.

        Args:
            features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            num_samples (int): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        samples = []

        # MC sampling
        for _ in range(num_samples):
            h = self.dropout(features)
            u = self.uncertainty_head(h)
            samples.append(u)

        samples = torch.stack(samples, dim=0)  # [S, B, 1, H, W]
        mean = samples.mean(dim=0)
        var = samples.var(dim=0)

        return mean, var


@register_model(name="simmlm", training_mode="reconstruction")
class SimMLMGenerator(nn.Module, IGenerator):
    """SimMLM: Dynamic Mixture of Modality Experts for MRI Synthesis.

    Handles missing modality imputation with:
    1. Modality-specific expert encoders
    2. Dynamic gating based on input quality
    3. Uncertainty quantification

    Args:
        modalities: List of modality names (e.g., ['t1', 't2', 'flair', 'dwi'])
        hidden_dim: Feature dimension
        num_blocks: Blocks per expert
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modalities: list[str] = None,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        enable_uncertainty: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            modalities (list[str]): Description.
            hidden_dim (int): Description.
            num_blocks (int): Description.
            enable_uncertainty (bool): Description.
        """
        super().__init__()

        if modalities is None:
            modalities = ["t1", "t2", "flair", "dwi"]

        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.hidden_dim = hidden_dim
        self.enable_uncertainty = enable_uncertainty

        # Expert encoders
        self.experts = nn.ModuleDict(
            {mod: ModalityExpert(in_channels, hidden_dim, num_blocks) for mod in modalities}
        )

        # Gating network
        self.gating = DynamicGatingNetwork(self.num_modalities, hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
        )

        # Decoder (shared across target modalities)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + self.num_modalities, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

        # Uncertainty estimator
        if enable_uncertainty:
            self.uncertainty = UncertaintyEstimator(hidden_dim)

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode available modalities.

        Args:
            inputs: Dict mapping modality name to tensor [B, C, H, W]

        Returns:
            fused: [B, D, H, W] fused features
            gates: [B, E] gating weights
        """
        B = next(iter(inputs.values())).shape[0]
        H, W = next(iter(inputs.values())).shape[2:]
        device = next(iter(inputs.values())).device

        # Extract features from each expert
        expert_features = []
        availability_mask = []

        for mod in self.modalities:
            if mod in inputs and inputs[mod] is not None:
                feat = self.experts[mod](inputs[mod])
                expert_features.append(feat)
                availability_mask.append(1.0)
            else:
                # Zero features for missing modality
                feat = torch.zeros(B, self.hidden_dim, H, W, device=device)
                expert_features.append(feat)
                availability_mask.append(0.0)

        availability_mask = (
            torch.tensor(availability_mask, device=device).unsqueeze(0).expand(B, -1)
        )

        # Compute gating weights
        gates = self.gating(expert_features, availability_mask)  # [B, E]

        # Weighted fusion
        fused = torch.zeros(B, self.hidden_dim, H, W, device=device)
        for i, feat in enumerate(expert_features):
            fused = fused + gates[:, i : i + 1, None, None] * feat

        # Additional fusion processing
        fused = self.fusion(fused)

        return fused, gates

    def decode(
        self,
        fused: torch.Tensor,
        target_modality: str,
    ) -> torch.Tensor:
        """Decode to target modality.

        Args:
            fused: [B, D, H, W] fused features
            target_modality: Name of target modality

        Returns:
            [B, C, H, W] synthesized target
        """
        B, D, H, W = fused.shape
        device = fused.device

        # One-hot encoding of target modality
        target_idx = self.modalities.index(target_modality)
        target_onehot = torch.zeros(B, self.num_modalities, H, W, device=device)
        target_onehot[:, target_idx, :, :] = 1.0

        # Condition on target
        conditioned = torch.cat([fused, target_onehot], dim=1)

        return self.decoder(conditioned)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        target_modality: str = None,
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Synthesize missing modality.

        Args:
            inputs: Dict of available modalities
            target_modality: Which modality to synthesize (if None, synthesize all missing)
            return_uncertainty: Whether to return uncertainty maps

        Returns:
            Dict with 'output', optionally 'uncertainty', 'gates'

        forward method for SimMLMGenerator.

        Executes PyTorch tensor operations.

        Args:
            inputs (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_modality (str): Expected input tensor.
            return_uncertainty (bool): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode available modalities
        fused, gates = self.encode(inputs)

        results = {"gates": gates}

        # Determine target(s)
        if target_modality is not None:
            targets = [target_modality]
        else:
            # Synthesize all missing modalities
            targets = [m for m in self.modalities if m not in inputs]

        # Decode each target
        for target in targets:
            synth = self.decode(fused, target)
            results[f"output_{target}"] = synth

            if return_uncertainty and self.enable_uncertainty:
                u_mean, u_var = self.uncertainty(fused)
                results[f"uncertainty_{target}"] = u_mean
                results[f"uncertainty_var_{target}"] = u_var

        # For compatibility, also set 'output' to first target
        if len(targets) > 0:
            results["output"] = results[f"output_{targets[0]}"]

        return results

    def generate(
        self,
        x: torch.Tensor,
        target_modality: str = "t2",
        source_modality: str = "t1",
        **kwargs,
    ) -> torch.Tensor:
        """Generate interface for compatibility.

        Args:
            x: [B, C, H, W] source modality
            target_modality: Modality to synthesize
            source_modality: Name of input modality

        Returns:
            [B, C, H, W] synthesized target
        """
        inputs = {source_modality: x}
        results = self.forward(inputs, target_modality)
        return results["output"]


__all__ = [
    "DynamicGatingNetwork",
    "ModalityExpert",
    "SimMLMGenerator",
    "UncertaintyEstimator",
]
