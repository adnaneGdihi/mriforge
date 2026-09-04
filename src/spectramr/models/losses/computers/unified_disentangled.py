"""Unified Disentangled Loss Computer - SSOT Implementation.

Computes all disentanglement-related losses in standardized LossOutput format.
Supports self-reconstruction, style/content consistency, pinning/bloch prior,
and multi-scale perceptual/structural losses.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.computers.base import BaseLossComputer, LossOutput

logger = logging.getLogger(__name__)


KNOWN_LOSS_COMPONENTS = [
    "recon",
    "recon_a",
    "recon_b",
    "recon_ba",
    "l1",
    "l2",
    "style_consistency",
    "content_consistency",
    "bloch_regression",
    "kl",
    "mind_ssc",
    "ssim",
    "ms_ssim",
    "perceptual",
    "hfen",
    "ffl",
    "hist",
    "tv_bias_field",
    "tv",
    "explicit_gradient",
    "patch_nce",
]


class UnifiedDisentangledLossComputer(BaseLossComputer):
    """Unified Disentangled loss computer implementing SSOT pattern.

    Handles the complex loss landscape of Disentangled Representation Learning (DRL).
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cpu")):
        """__init__.

        Args:
            config (Any): Description.
            device (torch.device): Description.
        """
        super().__init__(config, device)
        # Store log variances for homoscedastic uncertainty weighting
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
                for name in KNOWN_LOSS_COMPONENTS
            }
        )

    def _initialize_losses(self) -> None:
        """Initialize all loss functions using LossBuilder or manual fallback."""
        from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

        if self.config is not None:
            # Build errors with a real config MUST propagate (pitfall #9). The
            # old `except Exception: logger.debug(...)` swallowed the failure at
            # DEBUG (filtered out by default) and silently collapsed the
            # reconstruction/adversarial/physics/latent stack to a bare
            # L1+MSE+KL — review 2026-07-01. Fail loud instead.
            builder = LossBuilder(self.config, self.device)
            self.losses = (
                builder.build_reconstruction_losses()
                .build_adversarial_losses()
                .build_physics_losses()
                .build_latent_losses()
                .build()
            )
            return

        # No config supplied (direct construction in a test/script) — the ONLY
        # sanctioned minimal fallback.
        self.losses = {
            "l1": nn.L1Loss(),
            "mse": nn.MSELoss(),
            "kl": nn.KLDivLoss(reduction="batchmean"),
        }

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute all disentanglement losses."""
        device = pred.device
        components = {}

        # 1. Primary Reconstruction Loss
        recon_dict = self._compute_recon_losses(pred, target, **kwargs)
        components.update(recon_dict)

        recon_type = getattr(self.config.losses.reconstruction, "loss_type", "l1")
        if "recon_a" in recon_dict:
            components[recon_type] = recon_dict["recon_a"]
            components["recon"] = recon_dict["recon_a"]  # Alias for backward compatibility

        # 2. Style/Content Latent Consistency & PatchNCE
        components.update(self._compute_latent_losses(**kwargs))

        # 3. Physics/Bloch Prior
        components.update(self._compute_physics_losses(**kwargs))

        # 4. Structural/Perceptual Losses
        components.update(self._compute_structural_losses(pred, target, **kwargs))

        # 5. KL Divergence (Controlled Capacity)
        components.update(self._compute_kl_losses(iteration=iteration, **kwargs))

        # Weight and assembly
        total = self._stack_components(components, epoch=epoch, iteration=iteration)

        # [FIX] Detach components for logging metrics
        metrics = {
            k: v.detach().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in components.items()
        }

        return LossOutput(total=total, components=components, metrics=metrics)

    def _compute_recon_losses(self, pred, target, **kwargs) -> dict[str, torch.Tensor]:
        """Compute L1/L2 reconstruction losses."""
        l1_fn = self.losses.get("l1", nn.L1Loss())
        recon_components = {}

        recon_a = l1_fn(pred, target)
        recon_components["recon_a"] = recon_a

        if kwargs.get("is_paired"):
            x_ab = kwargs.get("x_ab")
            x_b = kwargs.get("x_b")
            if x_ab is not None and x_b is not None:
                recon_b = l1_fn(x_ab, x_b)
                recon_components["recon_b"] = recon_b

            x_ba = kwargs.get("x_ba")
            x_a = kwargs.get("x_a")
            if x_ba is not None and x_a is not None:
                recon_ba = l1_fn(x_ba, x_a)
                recon_components["recon_ba"] = recon_ba

        return recon_components

    @staticmethod
    def _align_spatial(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bilinearly interpolate pred to match target spatial dims if needed."""
        if pred.shape == target.shape:
            return pred
        if pred.ndim == 4 and target.ndim == 4 and pred.shape[2:] != target.shape[2:]:
            import torch.nn.functional as F

            pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=False)
        return pred

    def _compute_latent_losses(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute MSE losses on style and content codes."""
        components = {}
        mse_fn = self.losses.get("mse", nn.MSELoss())

        s_a, s_b = kwargs.get("s_a"), kwargs.get("s_b")
        c_a, c_b = kwargs.get("c_a"), kwargs.get("c_b")
        s_a_recon, s_b_recon = kwargs.get("s_a_recon"), kwargs.get("s_b_recon")
        c_a_recon, c_b_recon = kwargs.get("c_a_recon"), kwargs.get("c_b_recon")

        if s_a_recon is not None and s_a is not None:
            components["style_consistency"] = mse_fn(self._align_spatial(s_a_recon, s_a), s_a)
        if s_b_recon is not None and s_b is not None:
            components["style_consistency"] = components.get("style_consistency", 0.0) + mse_fn(
                self._align_spatial(s_b_recon, s_b), s_b
            )

        if c_a_recon is not None and c_a is not None:
            # PatchNCE on Content Features (Replaces pixel-wise Cycle Consistency)
            # c_a: [B, C, H, W] features from real
            # c_a_recon: [B, C, H, W] features from fake
            feat_real = c_a
            feat_fake = self._align_spatial(c_a_recon, c_a)

            b, c, h, w = feat_real.shape
            # Flatten spatial dimensions: [B, C, H*W] -> [B, H*W, C]
            feat_real_flat = feat_real.view(b, c, -1).transpose(1, 2)
            feat_fake_flat = feat_fake.view(b, c, -1).transpose(1, 2)

            # L2 Normalize features for cosine similarity
            import torch.nn.functional as F

            feat_real_norm = F.normalize(feat_real_flat, p=2, dim=-1)
            feat_fake_norm = F.normalize(feat_fake_flat, p=2, dim=-1)

            tau = 0.07  # Temperature parameter
            # Maximize Mutual Information between corresponding spatial locations.
            # A full [B, H*W, H*W] logits matrix would OOM at typical resolutions
            # (256*256 -> a 65536x65536 matrix), so we sample patches per batch
            # element below and build the small [N, N] logits there.
            num_patches = feat_real_flat.shape[1]
            max_patches = 256

            patch_nce_total = 0.0
            for i_b in range(b):
                if num_patches > max_patches:
                    # Randomly sample patches to avoid OOM
                    indices = torch.randperm(num_patches, device=feat_real.device)[:max_patches]
                else:
                    indices = torch.arange(num_patches, device=feat_real.device)

                cur_real = feat_real_norm[i_b : i_b + 1, indices, :]  # [1, N, C]
                cur_fake = feat_fake_norm[i_b : i_b + 1, indices, :]  # [1, N, C]

                cur_logits = torch.bmm(cur_fake, cur_real.transpose(1, 2)) / tau  # [1, N, N]
                cur_logits = cur_logits.squeeze(0)  # [N, N]
                cur_labels = torch.arange(cur_logits.shape[0], device=cur_logits.device)

                patch_nce_total += F.cross_entropy(cur_logits, cur_labels)

            components["patch_nce"] = patch_nce_total / b

        if c_b_recon is not None and c_b is not None:
            feat_real_b = c_b
            feat_fake_b = self._align_spatial(c_b_recon, c_b)

            b, c, h, w = feat_real_b.shape
            feat_real_flat_b = feat_real_b.view(b, c, -1).transpose(1, 2)
            feat_fake_flat_b = feat_fake_b.view(b, c, -1).transpose(1, 2)

            feat_real_norm_b = F.normalize(feat_real_flat_b, p=2, dim=-1)
            feat_fake_norm_b = F.normalize(feat_fake_flat_b, p=2, dim=-1)

            num_patches = feat_real_flat_b.shape[1]
            max_patches = 256

            patch_nce_total_b = 0.0
            for i_b in range(b):
                if num_patches > max_patches:
                    indices = torch.randperm(num_patches, device=feat_real_b.device)[:max_patches]
                else:
                    indices = torch.arange(num_patches, device=feat_real_b.device)

                cur_real = feat_real_norm_b[i_b : i_b + 1, indices, :]
                cur_fake = feat_fake_norm_b[i_b : i_b + 1, indices, :]

                cur_logits = torch.bmm(cur_fake, cur_real.transpose(1, 2)) / 0.07
                cur_logits = cur_logits.squeeze(0)
                cur_labels = torch.arange(cur_logits.shape[0], device=cur_logits.device)

                patch_nce_total_b += F.cross_entropy(cur_logits, cur_labels)

            if "patch_nce" in components:
                components["patch_nce"] = components["patch_nce"] + (patch_nce_total_b / b)
            else:
                components["patch_nce"] = patch_nce_total_b / b

        return components

    def _compute_physics_losses(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute Bloch regression and bias field penalty."""
        components = {}
        mse_fn = self.losses.get("mse", nn.MSELoss())

        p_a, p_a_pred = kwargs.get("p_a"), kwargs.get("p_a_pred")
        if p_a is not None and p_a_pred is not None:
            components["bloch_regression"] = mse_fn(p_a_pred, p_a)

        tissue_params_a = kwargs.get("tissue_params_a")
        if tissue_params_a is not None:
            if (
                hasattr(self.config.losses, "physics")
                and getattr(self.config.losses.physics, "lambda_tv_bias_field", 0.0) > 0
            ):
                if "bias_field" in tissue_params_a:
                    bf = tissue_params_a["bias_field"]
                    if bf.ndim == 4:
                        tv_loss = torch.mean(
                            (bf[:, :, :, :-1] - bf[:, :, :, 1:]) ** 2
                        ) + torch.mean((bf[:, :, :-1, :] - bf[:, :, 1:, :]) ** 2)
                        components["tv_bias_field"] = tv_loss

        return components

    def _compute_structural_losses(self, pred, target, **kwargs) -> dict[str, torch.Tensor]:
        """Compute structural and perceptual losses."""
        components = {}
        for name in [
            "ssim",
            "ms_ssim",
            "perceptual",
            "mind_ssc",
            "hfen",
            "ffl",
            "explicit_gradient",
            "hist",
        ]:
            fn = self.losses.get(name)
            if fn:
                components[name] = fn(pred, target)
        return components

    def _compute_kl_losses(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute KL divergence with capacity scheduling."""
        components = {}
        kl_div = kwargs.get("kl_div")

        if kl_div is None:
            mu_a, logvar_a = kwargs.get("mu_a"), kwargs.get("logvar_a")
            if mu_a is not None and logvar_a is not None:
                # [FIX] Constrain logvar to prevent e^logvar gradient singularities.
                # e^20 ≈ 4.85e8 → catastrophic. e^6 ≈ 403 → safe upper bound.
                logvar_a_c = torch.clamp(logvar_a, min=-10.0, max=6.0)
                mu_a_c = torch.clamp(mu_a, min=-50.0, max=50.0)
                kl_div = (
                    -0.5
                    * torch.sum(1 + logvar_a_c - mu_a_c.pow(2) - logvar_a_c.exp(), dim=1).mean()
                )

            mu_b, logvar_b = kwargs.get("mu_b"), kwargs.get("logvar_b")
            if mu_b is not None and logvar_b is not None:
                logvar_b_c = torch.clamp(logvar_b, min=-10.0, max=6.0)
                mu_b_c = torch.clamp(mu_b, min=-50.0, max=50.0)
                kl_b = (
                    -0.5
                    * torch.sum(1 + logvar_b_c - mu_b_c.pow(2) - logvar_b_c.exp(), dim=1).mean()
                )
                kl_div = (kl_div + kl_b) / 2 if kl_div is not None else kl_b

        if kl_div is not None:
            latent_cfg = self.config.losses.latent if self.config.losses else None
            if latent_cfg and getattr(latent_cfg, "use_capacity_scheduling", False):
                c_max = getattr(latent_cfg, "kl_capacity_target", 20.0)
                warmup = getattr(latent_cfg, "kl_capacity_warmup_steps", 10000)
                current_c = c_max * min(kwargs.get("iteration", 0) / (warmup + 1e-6), 1.0)
                kl_div = torch.abs(kl_div - current_c)
            components["kl"] = kl_div
        return components

    def _stack_components(
        self,
        components: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
        epoch: int = 0,
        iteration: int = 0,
    ) -> torch.Tensor:
        """Stack components with optional homoscedastic uncertainty weighting."""
        if not components:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total = torch.tensor(0.0, device=self.device, requires_grad=True)
        use_uncertainty = False
        latent_cfg = self.config.losses.latent if self.config.losses else None
        if latent_cfg:
            use_uncertainty = getattr(latent_cfg, "use_homoscedastic_uncertainty", False)

        for name, loss_val in components.items():
            if loss_val is None:
                continue
            base_w = weights.get(name) if weights else self._get_loss_weight(name, epoch, iteration)

            if use_uncertainty and name in self.log_vars:
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                total = total + base_w * (precision * loss_val + log_var)
            else:
                total = total + base_w * loss_val
        return total

    def _get_loss_weight(
        self, loss_name: str, epoch: int = 0, iteration: int = 0, **kwargs
    ) -> float:
        """Get DRL-specific weight mapping with Curriculum Loss Scheduling."""
        recon_cfg = (
            getattr(self.config.losses, "reconstruction", None)
            if self.config and hasattr(self.config, "losses")
            else None
        )

        if recon_cfg and getattr(recon_cfg, "use_curriculum_scheduling", False):
            warmup_epochs = getattr(recon_cfg, "curriculum_warmup_epochs", 20.0)
            max_epochs = getattr(recon_cfg, "curriculum_max_epochs", 100.0)

            l1_start = getattr(recon_cfg, "lambda_recon", getattr(recon_cfg, "lambda_l1", 5.0))
            l1_end = getattr(recon_cfg, "curriculum_l1_end_weight", 1.0)

            if epoch < warmup_epochs:
                if loss_name in ["recon", "l1"]:
                    return l1_start
            elif epoch <= max_epochs:
                p = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1e-6)
                decay = 0.5 * (1.0 + math.cos(math.pi * p))
                if loss_name in ["recon", "l1"]:
                    return l1_end + (l1_start - l1_end) * decay

        # Mapping
        mapping = {
            "recon": "lambda_recon",
            "recon_a": "lambda_recon",
            "recon_b": "lambda_recon",
            "recon_ba": "lambda_recon",
            "style_consistency": "lambda_style",
            "content_consistency": "lambda_content_consistency",
            "bloch_regression": "lambda_bloch",
            "kl": "lambda_kl",
            "mind_ssc": "lambda_mind_ssc",
            "ssim": "lambda_ssim",
            "perceptual": "lambda_perceptual",
            "hfen": "lambda_hfen",
            "ffl": "lambda_ffl",
            "hist": "lambda_hist",
            "explicit_gradient": "lambda_explicit_gradient",
            "ms_ssim": "lambda_ms_ssim",
            "patch_nce": "lambda_patch_nce",
        }
        lambda_key = mapping.get(loss_name, f"lambda_{loss_name}")

        if not hasattr(self.config, "losses") or self.config.losses is None:
            return 1.0

        for section in [self.config.losses.reconstruction, self.config.losses.latent]:
            if section and hasattr(section, lambda_key):
                val = getattr(section, lambda_key)
                if val is not None:
                    return float(val)
        return 1.0
