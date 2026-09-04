import torch
import torch.nn as nn

from spectramr.models.registry import register_model


@register_model(name="stable_diffusion_adapter", training_mode="diffusion")
class StableDiffusionAdapterGenerator(nn.Module):
    """
    Generator that mimics a Stable Diffusion U-Net with Adapter support.
    Designed to be used with the StableDiffusion process wrapper.
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels
        out_channels: int = 4,
        _base_model_path: str = None,
        adapter_type: str = None,
        ckpt_gradients: bool = False,
        image_size: int = 256,
        hidden_dim: int = 320,  # SD v1.5 base dim
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            _base_model_path (str): Description.
            adapter_type (str): Description.
            ckpt_gradients (bool): Description.
            image_size (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.adapter_type = adapter_type

        # Proper SD U-Net Structure using PyTorch Primitives

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )

        # Cross attention context mapping
        context_dim = 768  # Assuming standard text/clip context dim

        class ResBlock(nn.Module):
            def __init__(self, in_c, out_c, time_emb_dim):
                super().__init__()
                self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c))
                self.conv1 = nn.Sequential(
                    nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, 3, padding=1)
                )
                self.conv2 = nn.Sequential(
                    nn.GroupNorm(32, out_c), nn.SiLU(), nn.Conv2d(out_c, out_c, 3, padding=1)
                )
                self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

            def forward(self, x, t):
                """forward method for ResBlock.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
                    t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                h = self.conv1(x)
                t_emb = self.time_mlp(t)[..., None, None]
                h = h + t_emb
                h = self.conv2(h)
                return h + self.shortcut(x)

        class SpatialCrossAttention(nn.Module):
            def __init__(self, dim, context_dim, heads=8):
                super().__init__()
                self.heads = heads
                self.scale = (dim // heads) ** -0.5
                self.norm = nn.GroupNorm(32, dim)
                self.to_q = nn.Conv2d(dim, dim, 1)
                self.to_k = nn.Linear(context_dim, dim)
                self.to_v = nn.Linear(context_dim, dim)
                self.to_out = nn.Conv2d(dim, dim, 1)

            def forward(self, x, context=None):
                """forward method for SpatialCrossAttention.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
                    context (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                if context is None:
                    return x
                b, c, h, w = x.shape
                q = (
                    self.to_q(self.norm(x))
                    .view(b, self.heads, c // self.heads, h * w)
                    .transpose(-1, -2)
                )
                k = self.to_k(context).view(b, -1, self.heads, c // self.heads).transpose(1, 2)
                v = self.to_v(context).view(b, -1, self.heads, c // self.heads).transpose(1, 2)

                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                attn = dots.softmax(dim=-1)
                out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
                return x + self.to_out(out)

        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)

        # Down blocks: Extract features at 2 resolutions
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "res": ResBlock(hidden_dim, hidden_dim, hidden_dim * 4),
                        "attn": SpatialCrossAttention(hidden_dim, context_dim),
                    }
                ),
                nn.ModuleDict(
                    {
                        "down": nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                        "res": ResBlock(hidden_dim, hidden_dim * 2, hidden_dim * 4),
                        "attn": SpatialCrossAttention(hidden_dim * 2, context_dim),
                    }
                ),
            ]
        )

        # Middle block
        self.mid_block = nn.ModuleDict(
            {
                "res1": ResBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim * 4),
                "attn": SpatialCrossAttention(hidden_dim * 2, context_dim),
                "res2": ResBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim * 4),
            }
        )

        # Up blocks
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "res": ResBlock(hidden_dim * 4, hidden_dim * 2, hidden_dim * 4),
                        "attn": SpatialCrossAttention(hidden_dim * 2, context_dim),
                        "up": nn.ConvTranspose2d(
                            hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "res": ResBlock(hidden_dim * 3, hidden_dim, hidden_dim * 4),
                        "attn": SpatialCrossAttention(hidden_dim, context_dim),
                    }
                ),
            ]
        )

        self.norm_out = nn.GroupNorm(32, hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)

        if ckpt_gradients:
            self.set_grad_checkpointing(True)

    def set_grad_checkpointing(self, enable: bool):
        """set_grad_checkpointing.

        Args:
            enable (bool): Description.
        Returns:
            Any: Description.
        """
        self.gradient_checkpointing = enable

    def forward(self, x, timesteps, context=None):
        """
        Args:
            x: Latents [B, C, H, W]
            timesteps: [B]
            context: Conditioning [B, L, D] (optional)

        forward method for StableDiffusionAdapterGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Sinusoidal time embedding
        t_emb = self.time_embed(self._get_time_embedding(timesteps, self.hidden_dim))

        h = self.conv_in(x)
        skips = [h]

        # Downpass
        for block in self.down_blocks:
            if "down" in block:
                h = block["down"](h)
            h = block["res"](h, t_emb)
            h = block["attn"](h, context)
            skips.append(h)

        # Midpass
        h = self.mid_block["res1"](h, t_emb)
        h = self.mid_block["attn"](h, context)
        h = self.mid_block["res2"](h, t_emb)

        # Uppass
        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block["res"](h, t_emb)
            h = block["attn"](h, context)
            if "up" in block:
                h = block["up"](h)

        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.conv_out(h)

    def _get_time_embedding(self, timesteps, dim):
        # Simple sinusoidal embedding
        """_get_time_embedding.

        Args:
            timesteps (Any): Description.
            dim (Any): Description.
        Returns:
            Any: Description.
        """
        half_dim = dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
