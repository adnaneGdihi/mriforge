"""Multi-Task & Meta-Learning Generators
=========================================

Models for joint objectives, knowledge transfer, and few-shot adaptation.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 1. Multi-Task Shared — shared encoder with multiple task-specific heads
# ---------------------------------------------------------------------------


@register_model(name="multi_task_shared", training_mode="reconstruction")
class MultiTaskSharedGenerator(nn.Module, IGenerator):
    """Shared encoder with separate reconstruction and segmentation heads."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_classes: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.encoder = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.recon_decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, n_classes, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encoder(x)
        recon = self.recon_decoder(feat)
        return F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 2. Joint Recon-Seg — simultaneous reconstruction and segmentation
# ---------------------------------------------------------------------------


@register_model(name="joint_recon_seg", training_mode="reconstruction")
class JointReconSegGenerator(nn.Module, IGenerator):
    """Jointly optimizes reconstruction and segmentation with cross-task attention."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_classes: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.enc = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.cross_attn = nn.MultiheadAttention(bf * 4, num_heads=4, batch_first=True)
        self.recon_head = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, n_classes, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.enc(x)
        fH, fW = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).permute(0, 2, 1)
        tokens = tokens + self.cross_attn(tokens, tokens, tokens)[0]
        feat = tokens.permute(0, 2, 1).reshape(B, -1, fH, fW)
        return F.interpolate(
            self.recon_head(feat), size=(H, W), mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# 3. Knowledge Distillation Generator — student model for distillation
# ---------------------------------------------------------------------------


@register_model(name="knowledge_distillation", training_mode="reconstruction")
class KnowledgeDistillationGenerator(nn.Module, IGenerator):
    """Lightweight student network for teacher-student distillation."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 16, **kwargs
    ):
        super().__init__()
        bf = base_features  # intentionally small
        self.net = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.interpolate(self.net(x), size=x.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 4. Lottery Ticket Network — prunable network for lottery ticket hypothesis
# ---------------------------------------------------------------------------


@register_model(name="lottery_ticket_network", training_mode="reconstruction")
class LotteryTicketNetwork(nn.Module, IGenerator):
    """Dense network with magnitude pruning support (lottery ticket hypothesis)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 64, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.net = nn.Sequential(
            _ConvBN(in_channels, bf),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )
        # Store original init for rewinding
        self._initial_state = None

    def save_initial_weights(self) -> None:
        self._initial_state = copy.deepcopy(self.state_dict())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. Lottery Ticket U-Net — U-Net variant with structured pruning
# ---------------------------------------------------------------------------


@register_model(name="lottery_ticket_unet", training_mode="reconstruction")
class LotteryTicketUNet(nn.Module, IGenerator):
    """U-Net with structured pruning masks for the lottery ticket hypothesis."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = _ConvBN(in_channels, bf)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf, bf * 2))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4))
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = _ConvBN(bf * 4, bf * 2)
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = _ConvBN(bf * 2, bf)
        self.head = nn.Conv2d(bf, out_channels, 1)
        # Learnable pruning masks
        self.masks = nn.ParameterDict(
            {
                f"mask_{i}": nn.Parameter(torch.ones(bf * (2 ** min(i, 2))), requires_grad=False)
                for i in range(5)
            }
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 6. MAML Meta-Learning — model-agnostic meta-learning architecture
# ---------------------------------------------------------------------------


@register_model(name="maml_meta_learning", training_mode="reconstruction")
class MAMLMetaLearning(nn.Module, IGenerator):
    """MAML-compatible generator with parameter-free forward for inner-loop updates."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.InstanceNorm2d(bf),
            nn.ReLU(),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.InstanceNorm2d(bf * 2),
            nn.ReLU(),
            nn.Conv2d(bf * 2, bf, 3, padding=1),
            nn.InstanceNorm2d(bf),
            nn.ReLU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# 7. Hypernetwork U-Net — hypernetwork generates U-Net weights
# ---------------------------------------------------------------------------


@register_model(name="hypernetwork_unet", training_mode="reconstruction")
class HypernetworkUNet(nn.Module, IGenerator):
    """A hypernetwork generates per-sample weights for a small target U-Net."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 16, **kwargs
    ):
        super().__init__()
        bf = base_features
        target_params = bf * bf * 9 + bf  # rough param budget for target conv
        self.hyper = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 128),
            nn.GELU(),
            nn.Linear(128, target_params),
        )
        self.target_net = nn.Sequential(
            _ConvBN(in_channels, bf), _ConvBN(bf, bf), nn.Conv2d(bf, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Hypernetwork generates a modulation vector
        modulation = self.hyper(x)
        out = self.target_net(x)
        # Apply modulation as channel-wise scaling
        scale = modulation[:, : out.shape[1]].unsqueeze(-1).unsqueeze(-1)
        return out * (1 + scale * 0.1)
