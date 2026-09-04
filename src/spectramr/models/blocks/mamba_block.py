import importlib.util
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _mamba_fallback_allowed() -> bool:
    """Whether the (non-SSM) Gated-Conv+GRU fallback may be used.

    Opt-in ONLY, via ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` — intended for CPU/CI
    wiring tests on boxes without the ``mamba_ssm`` CUDA kernel. Real Mamba
    experiments must use the official kernel; a missing kernel otherwise fails
    loud (pitfall #9) so a GRU is never silently trained under the "Mamba"
    label (pitfall #16).
    """
    return os.environ.get("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mamba_ssm_importable() -> bool:
    """Lightweight check that the ``mamba_ssm`` package is present.

    Uses :func:`importlib.util.find_spec` (no side effects, no kernel load) —
    suitable for the Tier-1 audit. A present-but-broken CUDA kernel still passes
    here but fails loud at :class:`MambaBlock` construction.
    """
    return importlib.util.find_spec("mamba_ssm") is not None


class MambaBlock(nn.Module):
    """
    Mamba block implementation.
    Tries to use the official `mamba_ssm` implementation if available.
    Otherwise, falls back to a simplified Gated CNN/RNN approximation for compatibility.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        **kwargs,
    ):
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
            d_conv (int): Description.
            expand (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.use_official = False
        try:
            from mamba_ssm import Mamba

            self.mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs
            )
            self.use_official = True
        except ImportError:
            # Try alternative import path (mamba_ssm v2.x restructured modules)
            try:
                from mamba_ssm.modules.mamba_simple import Mamba

                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs,
                )
                self.use_official = True
                logger.info("Loaded Mamba from mamba_ssm.modules.mamba_simple.")
            except ImportError as e2:
                # No official selective-scan kernel. The Gated-Conv+GRU fallback
                # is NOT an SSM, so by default we FAIL LOUD (pitfall #9): an
                # experiment must never silently train a GRU under the "Mamba"
                # label (pitfall #16). The fallback is available ONLY via the
                # explicit SPECTRAMR_ALLOW_MAMBA_FALLBACK opt-in (CPU/CI wiring).
                if importlib.util.find_spec("mamba_ssm") is not None:
                    reason = (
                        f"mamba_ssm is installed but its Mamba CUDA kernel failed to "
                        f"import ({e2}); rebuild it against your CUDA toolchain "
                        f"(`pip install mamba-ssm causal-conv1d --no-build-isolation`)."
                    )
                else:
                    reason = (
                        "mamba_ssm is not installed; it is REQUIRED for Mamba models. "
                        "Install the optional extra: `pip install -e '.[mamba]'` "
                        "(needs CUDA + nvcc)."
                    )
                if not _mamba_fallback_allowed():
                    raise ImportError(
                        f"MambaBlock requires the official mamba_ssm selective-scan "
                        f"kernel. {reason} The Gated-Conv+GRU fallback is NOT "
                        f"equivalent to an SSM and is disabled by default; set "
                        f"SPECTRAMR_ALLOW_MAMBA_FALLBACK=1 ONLY for CPU/CI wiring tests "
                        f"(such runs are not scientifically 'Mamba')."
                    ) from e2

                logger.warning(
                    "[MambaBlock] %s Falling back to Gated-Conv+GRU (NOT equivalent "
                    "to an SSM) because SPECTRAMR_ALLOW_MAMBA_FALLBACK is set — wiring "
                    "tests only; do NOT use for Mamba experiments.",
                    reason,
                )
                self.use_official = False

                # Fallback layers: Project -> Conv1d -> Activation -> GRU -> Project.
                # Built ONLY on the opt-in fallback path (so the official-kernel
                # path no longer constructs unused GRU layers).
                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=True,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d_conv - 1,
                )
                self.activation = nn.SiLU()
                # Note: batch_first=True is critical for cuDNN compatibility
                self.gru = nn.GRU(
                    input_size=self.d_inner, hidden_size=self.d_inner, batch_first=True
                )
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: shape (B, L, D)
        """
        if self.use_official:
            # Disable AMP autocast for official Mamba SSM — its internal
            # GRU layers crash with cuDNN under float16 inputs.
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.mamba(x.float()).to(x.dtype)

        # Fallback implementation
        batch, seq_len, dim = x.shape

        # 1. Project
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 2. Conv1d (requires B, C, L)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2).contiguous()  # Ensure contiguity for GRU

        x_conv = self.activation(x_conv)
        # Activation can break contiguity - ensure it's contiguous before GRU
        if not x_conv.is_contiguous():
            x_conv = x_conv.contiguous()

        # Double-check: Force contiguous for cuDNN GRU
        x_conv = x_conv.contiguous()

        # 3. SSM (GRU approximation) - requires contiguous float32 input
        # cuDNN GRU triggers CUDNN_STATUS_NOT_SUPPORTED with inputs from
        # Conv1d→transpose even after .contiguous(). Disable cuDNN entirely
        # and use the native PyTorch GRU backend (universally compatible).
        with (
            torch.amp.autocast(device_type="cuda", enabled=False),
            torch.backends.cudnn.flags(enabled=False),
        ):
            x_ssm, _ = self.gru(x_conv.float())
        x_ssm = x_ssm.to(z.dtype)  # Cast back to original dtype

        # 4. Gating
        z = self.activation(z)
        out = x_ssm * z

        # 5. Output Project
        out = self.out_proj(out)
        out = self.dropout(out)

        return out
