"""Low-Rank Adaptation (LoRA) Implementation
===========================================

This module provides a generic implementation of Low-Rank Adaptation (LoRA)
for efficient fine-tuning of neural networks. It supports applying LoRA to
Linear and Conv2d layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Base class for LoRA layers."""

    def __init__(
        self,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        """__init__.

        Args:
            r (int): Description.
            lora_alpha (int): Description.
            lora_dropout (float): Description.
            merge_weights (bool): Description.
        """
        super().__init__()
        self.init_lora_parameters(r, lora_alpha, lora_dropout, merge_weights)

    def init_lora_parameters(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        """init_lora_parameters.

        Args:
            r (int): Description.
            lora_alpha (int): Description.
            lora_dropout (float): Description.
            merge_weights (bool): Description.
        Returns:
            Any: Description.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights_option = merge_weights


class LoRAAdapter(nn.Module):
    """Standalone LoRA adapter for any linear operation.

    This computes the LoRA branch only: output = scale * (x @ A.T @ B.T)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            r (int): Description.
            lora_alpha (int): Description.
            lora_dropout (float): Description.
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # result = (dropout(x) @ A.T) @ B.T * scaling
        # x: [..., in]
        # A: [r, in]
        # B: [out, r]

        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LoRAAdapter.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.lora_dropout(x)
        # x @ A.T -> [..., r]
        x = F.linear(x, self.lora_A)
        # x @ B.T -> [..., out]
        x = F.linear(x, self.lora_B)

        return x * self.scaling


class LoRALinear(nn.Linear, LoRALayer):
    """LoRA implemented in a dense layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            r (int): Description.
            lora_alpha (int): Description.
            lora_dropout (float): Description.
            merge_weights (bool): Description.
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.init_lora_parameters(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """train.

        Args:
            mode (bool): Description.
        Returns:
            Any: Description.
        """
        nn.Linear.train(self, mode)
        # No state mutation needed

    def merge_weights(self):
        """Explicitly merge weights for export/inference optimization."""
        if self.r > 0 and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # Base output
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            Any: Description.
        """
        out = F.linear(x, self.weight, bias=self.bias)

        # Add LoRA branch (stateless)
        if self.r > 0 and not self.merged:
            # Optimize: (x @ A.T) @ B.T * scale
            lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
            out = out + lora_out * self.scaling

        return out


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """LoRA implemented in a Conv2d layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (Union[int, tuple[int, int]]): Description.
            r (int): Description.
            lora_alpha (int): Description.
            lora_dropout (float): Description.
            merge_weights (bool): Description.
        """
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        self.init_lora_parameters(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros(
                    (
                        r * kernel_size[0] * kernel_size[1],
                        in_channels * kernel_size[0] * kernel_size[1],
                    )
                )
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (
                        out_channels * kernel_size[0] * kernel_size[1],
                        r * kernel_size[0] * kernel_size[1],
                    )
                )
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """train.

        Args:
            mode (bool): Description.
        Returns:
            Any: Description.
        """
        nn.Conv2d.train(self, mode)
        # No state mutation needed

    def merge_weights(self):
        """Explicitly merge weights for export/inference optimization."""
        if self.r > 0 and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            Any: Description.
        """
        if self.r > 0 and not self.merged:
            # Standard conv
            base_out = F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            # LoRA delta
            # This is slightly inefficient as we construct the full delta kernel
            # But for Conv2d, doing it "layer-wise" is tricky without unfolding.
            # Constructing delta weight is standard for LoRA Conv2d.
            delta_w = (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            lora_out = F.conv2d(
                x,
                delta_w,
                None,  # Bias handled in base_out
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return base_out + lora_out

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all parameters except LoRA parameters and optionally bias.

    Args:
        model: The model to modify.
        bias: 'none', 'all', or 'lora_only'.
    """
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Unsupported bias option: {bias}")


def inject_lora_adapters(
    model: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Inject LoRA adapters into matching Linear and Conv2d layers.

    Scans the model's named modules and replaces each ``nn.Linear`` or
    ``nn.Conv2d`` whose name matches any pattern in *target_modules* with
    the corresponding ``LoRALinear`` or ``LoRAConv2d`` wrapper.  The
    original pre-trained weights are copied into the new module and frozen;
    only the low-rank A/B matrices are trainable.

    Args:
        model: The model to inject LoRA adapters into.
        target_modules: Module name patterns to match (e.g. ``["attn", "proj"]``).
            If ``None``, defaults to ``["attn", "proj"]``.
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability on LoRA branch.

    Returns:
        Number of modules replaced.

    Example::

        >>> from mriforge.models.lora import inject_lora_adapters
        >>> count = inject_lora_adapters(model, target_modules=["attn"], rank=4, alpha=8)
        >>> print(f"Injected LoRA into {count} modules")
    """
    if target_modules is None:
        target_modules = ["attn", "proj"]

    replaced = 0
    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            # Check if this module matches any target pattern
            if not any(pattern in full_name for pattern in target_modules):
                continue

            if isinstance(child_module, nn.Linear):
                lora_module = LoRALinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=child_module.bias is not None,
                )
                # Copy pre-trained weights
                lora_module.weight.data.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    lora_module.bias.data.copy_(child_module.bias.data)
                replacements.append((parent_module, child_name, lora_module))
                replaced += 1

            elif isinstance(child_module, nn.Conv2d):
                kernel_size = child_module.kernel_size
                # Normalize to tuple
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)

                lora_module = LoRAConv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=kernel_size,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                )
                # Copy pre-trained weights
                lora_module.weight.data.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    lora_module.bias.data.copy_(child_module.bias.data)
                replacements.append((parent_module, child_name, lora_module))
                replaced += 1

    # Apply replacements (separate loop to avoid modifying during iteration)
    for parent_module, child_name, lora_module in replacements:
        setattr(parent_module, child_name, lora_module)

    return replaced
