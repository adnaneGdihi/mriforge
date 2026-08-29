import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fft_ops import ifft2c
from mriforge.infrastructure.physics.virtual_coil_dc import VirtualCoilDataConsistency
from mriforge.models.blocks.siren import SIREN
from mriforge.models.generators.complex_unet import ComplexUNet


class SIRENVarNetGenerator(nn.Module):
    """
    Distributed Implicit Fiducial Framework (DIFF) - Stage 2: SIREN-Modulated VarNet.

    Combines a Coordinate-MLP (SIREN) Field Estimator with an Unrolled
    Variational Network (VarNet).
    The SIREN predicts the global continuous complex field map from sparse boundaries,
    which is injected as a 'virtual sensitivity map' into the VarNet's Data Consistency blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 8,
        siren_features: int = 256,
        siren_layers: int = 3,
        unet_features: int = 32,
        init_alpha: float = 0.5,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_cascades (int): Description.
            siren_features (int): Description.
            siren_layers (int): Description.
            unet_features (int): Description.
            init_alpha (float): Description.
        """
        super().__init__()
        self.num_cascades = num_cascades
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Phase 1: SIREN Field Interpolator
        self.siren = SIREN(
            in_features=2,
            hidden_features=siren_features,
            hidden_layers=siren_layers,
            out_features=2,
        )

        # Phase 2: VarNet Cascades
        self.cascades = nn.ModuleList()
        # For each cascade, we have a VirtualCoilDC block and an Image Regularizer (e.g. UNet)
        for _ in range(num_cascades):
            # Using ComplexUNet as the regularizer (which inherently handles 2-channel real/imag input)
            regularizer = ComplexUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                init_features=unet_features,
                num_levels=4,
            )
            dc_block = VirtualCoilDataConsistency(init_alpha=init_alpha)

            self.cascades.append(nn.ModuleDict({"regularizer": regularizer, "dc_block": dc_block}))

        # Grid buffer to prevent regenerating spatial coordinates every forward pass
        self.register_buffer("_grid", None)

    def _get_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create or retrieve normalized [-1, 1] spatial coordinate grid."""
        if self._grid is not None and self._grid.shape[:2] == (H, W):
            return self._grid

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # SIREN expects [N, 2] coordinates
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self._grid = grid
        return self._grid

    def forward(
        self,
        y_measured: torch.Tensor,
        mask: torch.Tensor,
        x_initial: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            y_measured: Sampled k-space [B, 2, H, W]
            mask: Undersampling mask [B, 1, H, W]
            x_initial: Initial image guess [B, 2, H, W], e.g., zero-filled IFFT
                       If None, computes zero-filled IFFT.
        Returns:
            Reconstructed image [B, 2, H, W]
            (For Curriculum Training, SIREN map is evaluated internally)

        forward method for SIRENVarNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            y_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_initial (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = y_measured.shape
        device = y_measured.device

        # 1. Evaluate SIREN to get full-grid Virtual Sensitivity Map
        coords = self._get_grid(H, W, device)  # [H*W, 2]

        # SIREN requires independent forward per batch if mapping per-patient dynamically.
        # But for zero-shot per-subject training (Alternative A),
        # or amortized training with a conditional SIREN, we predict a single field.
        # Assuming batched evaluation of a global map for this experiment:
        S_pred_flat = self.siren(coords)  # [H*W, 2]
        S_pred = (
            S_pred_flat.view(H, W, 2).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        )  # [B, 2, H, W]

        # Keep S_pred for PINN loss computation outside if needed
        self.last_S_pred = S_pred

        # 2. Get initial image guess if not provided
        if x_initial is None:
            # y_measured is assumed real/imag stacked or interleaved. VirtualCoilDC handles complex conversions.
            y_c = torch.complex(y_measured[:, 0], y_measured[:, 1])
            x_zf_c = ifft2c(y_c)
            x_est = torch.stack([x_zf_c.real, x_zf_c.imag], dim=1)
        else:
            x_est = x_initial

        # 3. Unrolled Cascades
        for cascade in self.cascades:
            # Regularization in image space
            res = cascade["regularizer"](x_est)
            # Add residual (identity skip connection)
            x_reg = x_est + res

            # Virtual-Coil Data Consistency
            x_est = cascade["dc_block"](x_reg, y_measured, mask, S_map=S_pred)

        return x_est
