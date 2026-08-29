"""Source-Target Adaptation for Diffusion (SoTa-DiT).

Implements Test-Time Adaptation by optimizing affine parameters of normalization
layers in the score network to minimize data consistency loss.
"""

import torch
import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor

from mriforge.infrastructure.physics.fft_ops import fftnc, ifftnc
from mriforge.models.diffusion.score_sde.score_network import ScoreNetwork
from mriforge.models.registry import register_model


@register_model(name="sota_adapter", training_mode="adaptation")
class SoTaAdapter(nn.Module):
    """Test-Time Adapter for Score-Based Diffusion Models."""

    def __init__(
        self,
        score_net: ScoreNetwork,
        learning_rate: float = 1e-4,
        num_steps: int = 10,
    ):
        """__init__.

        Args:
            score_net (ScoreNetwork): Description.
            learning_rate (float): Description.
            num_steps (int): Description.
        """
        super().__init__()
        self.score_net = score_net
        self.learning_rate = learning_rate
        self.num_steps = num_steps

        # Identify adaptable parameters (Norm Layers)
        self.adaptable_params = []
        for name, module in self.score_net.named_modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
                # We optimize weight and bias
                has_affine = False
                if hasattr(module, "affine"):
                    has_affine = module.affine
                elif hasattr(module, "elementwise_affine"):
                    has_affine = module.elementwise_affine

                if has_affine:
                    self.adaptable_params.append(module.weight)
                    self.adaptable_params.append(module.bias)

    def adapt(
        self,
        k_obs: Complex[Tensor, "B C D H W"],
        mask: Float[Tensor, "B 1 D H W"],
        device: torch.device,
    ):
        """Perform test-time adaptation on the current observation."""

        from mriforge.infrastructure.training.builders import OptimizationBuilder

        optimizer = OptimizationBuilder.create_single_optimizer(
            self.adaptable_params, learning_rate=self.learning_rate
        )

        # We need a forward operator A(x) -> k
        # Assuming simple FFT
        def A(x):
            # x is real [B, C, D, H, W]? ScoreNet usually outputs noise/score.
            # We need to adapt such that the *denoised* image is consistent.
            # Estimating x0 from xt?
            # For VE-SDE: x0 = x + sigma^2 score
            # But we don't have x_sigma. We have y.
            # We can sample noise, generate x_t, predict x_0, compare to y.

            # Problem: Predicting x_0 from random noise x_T is hard in one step.
            # Adaptation usually runs *during* sampling or before?
            # "SoTa" usually implies adapting on the specific instance input.
            # If we have y, we can try to find x such that A(x)=y and Prior(x) is high.
            # Prior(x) is implicit in score net.

            # Strategy: Use DSM-like objective?
            # || S(x_t) - (x_0 - x_t)/sigma^2 ||? We don't have x_0.

            # Robust Strategy for MRI:
            # Generative Reconstruction usually does:
            # x* = argmin ||Ax - y|| s.t. x \in Prior

            # TTA approach: Update theta to minimize || A(Denoise(x_t, theta)) - y ||
            # for some intermediate t? or t near 0?

            # Let's pick a few timesteps and enforce consistency on Tweedie estimate.
            """A.

            Args:
                x (Any): Description.
            Returns:
                Any: Description.
            """
            return fftnc(x)

        for _ in range(self.num_steps):
            optimizer.zero_grad()

            # Sample random time and noise
            # We want small t where signal is preserved, to update mostly fine details?
            # Or large t?
            # Consistency is best enforced when x_t contains info.
            # But x_t = x_0 + n.
            # Tweedie x_0_hat = x_t + sigma^2 s(x_t).

            # Let's sample t \in [0.1, 0.5] (mid range)
            t = torch.rand(k_obs.shape[0], device=device) * 0.4 + 0.1

            # Generate noisy estimate based on y (measurement guided initialization)
            # x_ref = A^T y (Zero-filled)
            x_ref = ifftnc(k_obs).real

            # Perturb
            # sigma(t) approximation (assuming VE SDE stats from Phase 2)
            # sigma_min=0.01, sigma_max=50.
            # sigma_t = 0.01 * (50/0.01)^t
            sigma_t = 0.01 * (5000.0) ** t
            sigma_t = sigma_t.view(-1, 1, 1, 1, 1)

            noise = torch.randn_like(x_ref)
            x_t = x_ref + sigma_t * noise

            # Predict score
            score = self.score_net(x_t, t)

            # Tweedie Denoising
            # x_0_hat = x_t + sigma^2 * score
            x_0_hat = x_t + (sigma_t**2) * score

            # Consistency Loss
            k_pred = A(x_0_hat)
            # Masked consistency
            # k_obs is masked.
            diff = (k_pred - k_obs) * mask
            loss = torch.norm(diff)

            loss.backward()
            optimizer.step()

        return self.score_net
