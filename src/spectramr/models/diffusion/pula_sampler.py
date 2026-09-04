"""Advanced Diffusion Samplers for MRI Reconstruction.

Implements:
- pULA: Preconditioned Unadjusted Langevin Algorithm
- MCG: Manifold Constrained Gradient

These samplers accelerate diffusion-based MRI reconstruction
by accounting for the ill-conditioned forward operator.

References:
- Chung et al. "Improving Diffusion Models for Inverse Problems using Manifold Constraints"
- Laumont et al. "Bayesian imaging using plug & play priors: when Langevin meets Tweedie"
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.models.diffusion.samplers import register_sampler


@register_sampler(name="pula", aliases=["PULA", "PULASampler"])
class PULASampler(nn.Module):
    """Preconditioned Unadjusted Langevin Algorithm (pULA) Sampler.

    Addresses the ill-conditioning of the MRI forward operator
    by using a preconditioner that normalizes the gradient curvature.

    The update rule is:
        x_{k+1} = x_k + γ M_t (∇log p(x) + ∇log p(y|x)) + √(2γ M_t) z

    where M_t ≈ (A^H A + σ_t^{-2} I)^{-1} is the preconditioner.

    Args:
        score_model: Pre-trained score network s_θ(x, t)
        forward_op: Forward operator A (e.g., NUFFT)
        adjoint_op: Adjoint operator A^H
        num_steps: Number of sampling steps
        step_size: Base step size γ
    """

    def __init__(
        self,
        score_model: nn.Module,
        forward_op: Callable | None = None,
        adjoint_op: Callable | None = None,
        num_steps: int = 1000,
        step_size: float = 1e-5,
        noise_schedule: str = "cosine",
    ):
        """__init__.

        Args:
            score_model (nn.Module): Description.
            forward_op (Optional[Callable]): Description.
            adjoint_op (Optional[Callable]): Description.
            num_steps (int): Description.
            step_size (float): Description.
            noise_schedule (str): Description.
        """
        super().__init__()

        self.score_model = score_model
        self.forward_op = forward_op
        self.adjoint_op = adjoint_op
        self.num_steps = num_steps
        self.step_size = step_size

        # Setup noise schedule
        self._setup_schedule(noise_schedule)

    def _setup_schedule(self, schedule: str):
        """Setup noise schedule for diffusion."""
        if schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, 1, self.num_steps + 1)
            alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        else:  # linear
            betas = torch.linspace(0.0001, 0.02, self.num_steps)
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigma_t", torch.sqrt(1 - alphas_cumprod))

    def compute_preconditioner(
        self,
        x: torch.Tensor,
        sigma: float,
        cg_iterations: int = 10,
    ) -> Callable:
        """Compute preconditioner M_t via conjugate gradient.

        M_t ≈ (A^H A + σ^{-2} I)^{-1}

        This is computed implicitly via CG rather than forming the matrix.

        Args:
            x: Current iterate (for shape)
            sigma: Current noise level
            cg_iterations: Number of CG iterations

        Returns:
            Preconditioner function M(v)
        """
        reg = 1.0 / (sigma**2 + 1e-8)

        def precond(v: torch.Tensor) -> torch.Tensor:
            """Apply preconditioner to vector v using CG."""

            # A^H A v + reg * v
            def matvec(u):
                """matvec.

                Args:
                    u (Any): Description.
                Returns:
                    Any: Description.
                """
                if self.forward_op is not None and self.adjoint_op is not None:
                    return self.adjoint_op(self.forward_op(u)) + reg * u
                return reg * u  # Identity if no operator

            # CG solve: (A^H A + reg I) z = v
            z = torch.zeros_like(v)
            r = v - matvec(z)
            p = r.clone()
            rsold = (r * r).sum()

            for _ in range(cg_iterations):
                Ap = matvec(p)
                alpha = rsold / ((p * Ap).sum() + 1e-10)
                z = z + alpha * p
                r = r - alpha * Ap
                rsnew = (r * r).sum()
                if rsnew < 1e-10:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew

            return z

        return precond

    def compute_likelihood_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Compute gradient of log-likelihood: ∇_x log p(y|x).

        For Gaussian noise: ∇_x log p(y|x) = -A^H (Ax - y) / σ²

        Args:
            x: Current image estimate
            y: Measurements
            sigma: Noise level

        Returns:
            Likelihood gradient
        """
        if self.forward_op is None or self.adjoint_op is None:
            return torch.zeros_like(x)

        residual = self.forward_op(x) - y
        grad = -self.adjoint_op(residual) / (sigma**2 + 1e-8)

        return grad

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        x_init: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Sample from posterior p(x|y) using pULA.

        Args:
            y: Measurements [B, ...]
            x_init: Initial sample (optional)
            shape: Output shape if x_init not provided

        Returns:
            Posterior sample x
        """
        device = y.device

        # Initialize
        if x_init is not None:
            x = x_init
        elif shape is not None:
            x = torch.randn(shape, device=device)
        else:
            # Assume adjoint gives right shape
            x = self.adjoint_op(y) if self.adjoint_op else torch.randn_like(y)

        for t in range(self.num_steps - 1, -1, -1):
            sigma = self.sigma_t[t].item()

            # Time tensor
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Prior score (from score model)
            prior_score = self.score_model(x, t_tensor)

            # Likelihood score
            likelihood_score = self.compute_likelihood_gradient(x, y, sigma)

            # Total score
            total_score = prior_score + likelihood_score

            # Preconditioner
            precond = self.compute_preconditioner(x, sigma)

            # Preconditioned score
            precond_score = precond(total_score)

            # Langevin step
            step_size = self.step_size * (sigma**2)
            noise = torch.randn_like(x)

            x = x + step_size * precond_score + torch.sqrt(2 * step_size) * precond(noise)

        return x


@register_sampler(name="mcg", aliases=["MCG", "MCGSampler"])
class MCGSampler(nn.Module):
    """Manifold Constrained Gradient (MCG) Sampler.

    Projects the data consistency gradient onto the tangent space
    of the data manifold to prevent hallucinations.

    Key insight: The score function ∇log p(x) is orthogonal to the
    manifold in high-probability regions. MCG projects gradients
    to stay on the manifold.

    Args:
        score_model: Pre-trained score network
        forward_op: Forward operator A
        adjoint_op: Adjoint operator A^H
        num_steps: Number of sampling steps
        mcg_weight: Weight for manifold constraint
    """

    def __init__(
        self,
        score_model: nn.Module,
        forward_op: Callable | None = None,
        adjoint_op: Callable | None = None,
        num_steps: int = 1000,
        mcg_weight: float = 1.0,
    ):
        """__init__.

        Args:
            score_model (nn.Module): Description.
            forward_op (Optional[Callable]): Description.
            adjoint_op (Optional[Callable]): Description.
            num_steps (int): Description.
            mcg_weight (float): Description.
        """
        super().__init__()

        self.score_model = score_model
        self.forward_op = forward_op
        self.adjoint_op = adjoint_op
        self.num_steps = num_steps
        self.mcg_weight = mcg_weight

        self._setup_schedule()

    def _setup_schedule(self):
        """Setup cosine noise schedule."""
        s = 0.008
        t = torch.linspace(0, 1, self.num_steps + 1)
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, 0.0001, 0.999)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod[:-1])
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

    def tweedie_estimate(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Tweedie's formula: estimate x_0 from x_t and score.

        x̂_0 = (x_t + σ_t² ∇log p(x_t)) / α_t

        Args:
            x_t: Noisy sample at time t
            t: Time indices
            score: Score ∇log p(x_t)

        Returns:
            Estimated clean image x̂_0
        """
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_hat = (x_t + sigma_t**2 * score) / alpha_t

        return torch.clamp(x0_hat, -1, 1)

    def project_to_manifold(
        self,
        gradient: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Project gradient onto manifold tangent space.

        The score function is approximately orthogonal to the manifold.
        We project out the component along the score:

        g_proj = g - (g · s / ||s||²) * s

        Args:
            gradient: Data consistency gradient
            score: Manifold normal (score function)

        Returns:
            Projected gradient on tangent space
        """
        # Flatten for dot product
        g_flat = gradient.view(gradient.shape[0], -1)
        s_flat = score.view(score.shape[0], -1)

        # Project out score component
        dot_product = (g_flat * s_flat).sum(dim=1, keepdim=True)
        s_norm_sq = (s_flat**2).sum(dim=1, keepdim=True) + 1e-8

        proj_coef = dot_product / s_norm_sq
        g_proj = g_flat - proj_coef * s_flat

        return g_proj.view_as(gradient)

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        x_init: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Sample from posterior using MCG.

        Args:
            y: Measurements
            x_init: Initial sample
            shape: Output shape

        Returns:
            Reconstructed image
        """
        device = y.device

        # Initialize
        if x_init is not None:
            x = x_init
        elif shape is not None:
            x = torch.randn(shape, device=device)
        else:
            x = self.adjoint_op(y) if self.adjoint_op else torch.randn_like(y)

        for t in range(self.num_steps - 1, -1, -1):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Get prior score
            score = self.score_model(x, t_tensor)

            # Tweedie estimate of x_0
            x0_hat = self.tweedie_estimate(x, t_tensor, score)

            # Data consistency gradient
            if self.forward_op is not None and self.adjoint_op is not None:
                residual = self.forward_op(x0_hat) - y
                dc_grad = self.adjoint_op(residual)
            else:
                dc_grad = torch.zeros_like(x)

            # MCG: Project DC gradient onto manifold
            dc_grad_proj = self.project_to_manifold(dc_grad, score)

            # Standard reverse diffusion step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            # Mean prediction
            noise_pred = -self.sqrt_one_minus_alphas_cumprod[t] * score
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # Apply MCG correction to x0
            x0_corrected = x0_pred - self.mcg_weight * dc_grad_proj

            # Reverse step
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_cumprod) * x0_corrected + torch.sqrt(1 - alpha_cumprod) * noise
            else:
                x = x0_corrected

        return x


@register_sampler(name="dps_mri", aliases=["DPS", "DPSMRI", "DPSMRISampler"])
class DPSMRISampler(nn.Module):
    """Diffusion Posterior Sampling (DPS) specialized for MRI.

    Combines pULA preconditioning with MCG manifold constraints
    for robust MRI reconstruction.

    Args:
        score_model: Pre-trained unconditional score model
        forward_op: MRI forward operator (Sens × NUFFT)
        adjoint_op: Adjoint operator
        sampler_type: 'pula', 'mcg', or 'hybrid'
    """

    def __init__(
        self,
        score_model: nn.Module,
        forward_op: Callable | None = None,
        adjoint_op: Callable | None = None,
        num_steps: int = 1000,
        sampler_type: str = "hybrid",
    ):
        """__init__.

        Args:
            score_model (nn.Module): Description.
            forward_op (Optional[Callable]): Description.
            adjoint_op (Optional[Callable]): Description.
            num_steps (int): Description.
            sampler_type (str): Description.
        """
        super().__init__()

        self.sampler_type = sampler_type

        if sampler_type == "pula":
            self.sampler = PULASampler(score_model, forward_op, adjoint_op, num_steps)
        elif sampler_type == "mcg":
            self.sampler = MCGSampler(score_model, forward_op, adjoint_op, num_steps)
        elif sampler_type == "hybrid":
            self.pula = PULASampler(score_model, forward_op, adjoint_op, num_steps)
            self.mcg = MCGSampler(score_model, forward_op, adjoint_op, num_steps)
        else:
            raise ValueError(
                f"Unknown sampler_type {sampler_type!r}; expected one of 'pula', 'mcg', 'hybrid'."
            )

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        x_init: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Sample from posterior.

        Args:
            y: k-space measurements
            x_init: Initial image estimate
            shape: Output shape

        Returns:
            Reconstructed MRI image
        """
        if self.sampler_type == "hybrid":
            # First pass: pULA for fast convergence
            x_pula = self.pula.sample(y, x_init, shape)
            # Second pass: MCG for manifold refinement
            x_final = self.mcg.sample(y, x_pula)
            return x_final
        else:
            return self.sampler.sample(y, x_init, shape)


__all__ = [
    "DPSMRISampler",
    "MCGSampler",
    "PULASampler",
]
