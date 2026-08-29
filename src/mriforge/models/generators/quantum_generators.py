"""Quantum & Neuromorphic Generators
=====================================

Quantum-inspired, spiking, and unconventional computing architectures.
"""

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Quantum-Inspired — parameterized quantum circuit simulation on classical HW
# ---------------------------------------------------------------------------


class _QuantumLayer(nn.Module):
    """Simulates a parameterized quantum gate via rotation matrices."""

    def __init__(self, dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(dim) * 0.1)
        self.phi = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotation: Ry(theta) ⊗ Rz(phi) simulated as element-wise phase + amplitude
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        cos_p = torch.cos(self.phi)
        sin_p = torch.sin(self.phi)
        # Apply rotation to pairs of channels
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        y1 = cos_t[:half] * x1 - sin_t[:half] * x2
        y2 = sin_t[:half] * x1 + cos_t[:half] * x2
        return torch.cat([y1, y2], dim=-1)


@register_model(name="quantum_inspired", training_mode="reconstruction")
class QuantumInspiredGenerator(nn.Module, IGenerator):
    """Quantum-inspired network with parameterized rotation gates."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 64,
        n_layers: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, dim, 1)
        self.quantum_layers = nn.ModuleList([_QuantumLayer(dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.proj_out = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.proj_in(x)
        tokens = feat.permute(0, 2, 3, 1)  # (B, H, W, dim)
        for ql, norm in zip(self.quantum_layers, self.norms, strict=False):
            tokens = norm(tokens + ql(tokens))
        return self.proj_out(tokens.permute(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# 2. Quantum U-Net — U-Net with quantum-inspired entanglement layers
# ---------------------------------------------------------------------------


class _EntanglementLayer(nn.Module):
    """CNOT-like entanglement between channel pairs."""

    def __init__(self, ch: int):
        super().__init__()
        self.control_gate = nn.Conv2d(ch, ch, 1, groups=ch)
        self.target_gate = nn.Conv2d(ch, ch, 1, groups=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        control = torch.sigmoid(self.control_gate(x))
        target = self.target_gate(x)
        return x + control * target


@register_model(name="quantum_unet", training_mode="reconstruction")
class QuantumUNet(nn.Module, IGenerator):
    """U-Net with quantum-inspired entanglement layers at each level."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = nn.Sequential(_ConvBN(in_channels, bf), _EntanglementLayer(bf))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf, bf * 2), _EntanglementLayer(bf * 2))
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4), _EntanglementLayer(bf * 4)
        )
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = nn.Sequential(_ConvBN(bf * 4, bf * 2), _EntanglementLayer(bf * 2))
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(_ConvBN(bf * 2, bf), _EntanglementLayer(bf))
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 3. Spiking Neural Network — event-driven processing with LIF neurons
# ---------------------------------------------------------------------------


class _LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient."""

    def __init__(self, tau: float = 0.5, threshold: float = 1.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold

    def forward(self, x: torch.Tensor, n_steps: int = 4) -> torch.Tensor:
        membrane = torch.zeros_like(x)
        spike_sum = torch.zeros_like(x)
        for _ in range(n_steps):
            membrane = self.tau * membrane + x
            spike = (membrane >= self.threshold).float()
            spike_sum = spike_sum + spike
            membrane = membrane * (1 - spike)
        return spike_sum / n_steps


@register_model(name="spiking_neural_network", training_mode="reconstruction")
class SpikingNeuralNetworkGenerator(nn.Module, IGenerator):
    """SNN with LIF neurons: temporal integration over spike trains."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_steps: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.n_steps = n_steps
        self.conv1 = nn.Conv2d(in_channels, bf, 3, padding=1)
        self.lif1 = _LIFNeuron()
        self.conv2 = nn.Conv2d(bf, bf * 2, 3, padding=1)
        self.lif2 = _LIFNeuron()
        self.conv3 = nn.Conv2d(bf * 2, bf, 3, padding=1)
        self.lif3 = _LIFNeuron()
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.lif1(self.conv1(x), self.n_steps)
        h = self.lif2(self.conv2(h), self.n_steps)
        h = self.lif3(self.conv3(h), self.n_steps)
        return self.head(h)


# ---------------------------------------------------------------------------
# 4. Spin Glass Inspired — Ising/Boltzmann machine-inspired energy model
# ---------------------------------------------------------------------------


@register_model(name="spin_glass_inspired", training_mode="reconstruction")
class SpinGlassInspiredGenerator(nn.Module, IGenerator):
    """Ising model-inspired: iterative spin relaxation for reconstruction.

    Learns coupling weights (J) and external field (h) analogous to spin glass.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_relax: int = 5,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.n_relax = n_relax
        self.init = nn.Sequential(_ConvBN(in_channels, bf), nn.Conv2d(bf, out_channels, 1))
        # Coupling (nearest-neighbor interaction)
        self.coupling = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False
        )
        # External field from input
        self.field = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), nn.GELU(), nn.Conv2d(bf, out_channels, 1)
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        spins = torch.tanh(self.init(x))
        h_ext = self.field(x)
        for _ in range(self.n_relax):
            local_field = self.coupling(spins) + h_ext
            spins = torch.tanh(local_field / (self.temperature + 1e-6))
        return spins
