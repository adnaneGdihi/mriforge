import torch
import torch.nn as nn

from mriforge.models.layers.spatial_transformer import SpatialTransformer
from mriforge.models.registry import register_model


@register_model("b0_estimator", training_mode="reconstruction")
class B0Estimator(nn.Module):
    """
    Physics-Informed network that estimates B0 inhomogeneity from paired data.
    Input: Concatenated [64mT, 3T]
    Output: Displacement Field (Flow) and Corrected 64mT Image
    """

    def __init__(self, input_shape=(160, 160), input_channels=2, **kwargs):
        """__init__.

        Args:
            input_shape (Any): Description.
            input_channels (Any): Description.
        """
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(input_channels, 32)
        self.enc2 = self._conv_block(32, 64, stride=2)
        self.enc3 = self._conv_block(64, 128, stride=2)
        self.enc4 = self._conv_block(128, 256, stride=2)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = self._conv_block(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = self._conv_block(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self._conv_block(64 + 32, 32)

        # Flow Head: Predicts (dy, dx) or (dz, dy, dx)
        # We initialize weights to zero so training starts with Identity transform
        self.flow_head = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.flow_head.weight.data.normal_(0, 1e-5)
        self.flow_head.bias.data.zero_()

        self.transformer = SpatialTransformer(input_shape)

    def _conv_block(self, in_c, out_c, stride=1):
        """_conv_block.

        Args:
            in_c (Any): Description.
            out_c (Any): Description.
            stride (Any): Description.
        Returns:
            Any: Description.
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, moving_64mT, fixed_3T=None, **kwargs):
        # Concatenate inputs: [B, 2, H, W]
        """forward.

        Args:
            moving_64mT: Either the moving 64mT image [B, 1, H, W] or a
                concatenated [B, 2, H, W] tensor when called via the
                standard training pipeline (single-arg forward(x)).
            fixed_3T: The fixed 3T reference [B, 1, H, W].  When *None*,
                ``moving_64mT`` is assumed to be pre-concatenated and is
                split along the channel dimension.
        Returns:
            Tuple of (corrected_image, flow_field).

        forward method for B0Estimator.

        Executes PyTorch tensor operations.

        Args:
            moving_64mT (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            fixed_3T (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # B0 estimation needs paired (moving, fixed) input because B0
        # inhomogeneity is inferred from the geometric distortion between
        # the two scans. There is no single-image fallback that produces
        # a physically meaningful flow.
        #
        # Accept either:
        # * the standard pipeline form ``forward(concatenated)`` where
        #   the input is already ``[B, 2, H, W]`` ([moving|fixed] cat),
        # * or the explicit kwarg form ``forward(moving, fixed=...)``,
        # * or — added 2026-05-11 to fix exp_51's 1-channel YAML —
        #   the ``forward(moving, target=fixed_image)`` form that
        #   strategies which propagate ``batch_context["hr"]`` via
        #   ``forward_kwargs`` can use without a strategy-side branch.
        if fixed_3T is None:
            fixed_3T = kwargs.get("target")
        if fixed_3T is None:
            if moving_64mT.shape[1] >= 2:
                x = moving_64mT
                moving_64mT = x[:, :1]
            else:
                raise ValueError(
                    f"B0Estimator requires paired (moving_64mT, fixed_3T) "
                    f"input but got a single-channel ({moving_64mT.shape[1]}-ch) "
                    "tensor with no ``fixed_3T`` kwarg and no ``target`` "
                    "kwarg.  Configure the experiment YAML to (a) emit a "
                    "[B, 2, H, W] concatenated tensor as the model input, "
                    "(b) call ``forward(moving, fixed=...)`` explicitly "
                    "from a registration strategy, or (c) propagate "
                    "``batch_context['hr']`` via ``forward_kwargs['target']``. "
                    "A single-image fallback would produce no meaningful "
                    "B0 estimate (CLAUDE.md #9)."
                )
        else:
            x = torch.cat([moving_64mT, fixed_3T], dim=1)

        # UNet Pass
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Predict Flow
        flow = self.flow_head(d1)

        # Apply Correction
        corrected = self.transformer(moving_64mT, flow)

        return corrected, flow
