"""Information-Bottleneck active acquisition strategy — Idea 4.

The training procedure is standard self-supervised reconstruction with
a randomly sampled k-space mask; the novel logic lives in the
inference-time policy ``select_next_line(observed_mask, current_estimate)``
that picks which line to acquire next so as to maximise the
information-bottleneck objective

    L_IB(ℓ) = I(X; Y_ℓ | Y_obs) − β · I(Y_ℓ; M_ℓ).

For the linear-Gaussian regime ``L_IB`` admits the closed form quoted
in TODO/audit/audit_plan_novel.md (Idea 4). The implementation in this
file ships the linear-Gaussian path; the nonlinear (Hutchinson) path is
gated on a registered diffusion score model and is exposed but marked
``TODO(phase-2-refinement)``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from mriforge.infrastructure.training.builders.environment import TrainingEnvironment
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.infrastructure.training.strategies.ib_active_acquisition_nonlinear import (
    ib_score_nonlinear,
    posterior_covariance_woodbury_update,
)

logger = logging.getLogger(__name__)


class IBActiveAcquisitionStrategy(BaseTrainingStrategy):
    """SSL-style training with an IB line-selection policy at inference.

    Training simply minimises ``‖f_θ(M⊙y) − x‖`` over random masks (the
    classical SSDU recipe). The information-bottleneck logic appears
    only at policy-rollout time via :meth:`select_next_line`.
    """

    #: Advertised line-selection regimes. An unknown ``mode`` must raise
    #: rather than silently default to the linear-Gaussian path (pitfall #9).
    VALID_MODES: frozenset[str] = frozenset({"linear_gaussian", "nonlinear"})

    def __init__(
        self,
        env: TrainingEnvironment,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, **kwargs)
        ib = getattr(self.config.training, "ib_active_acquisition", None)
        self.beta = float(getattr(ib, "beta", 1.0)) if ib is not None else 1.0
        self.mode = (
            str(getattr(ib, "mode", "linear_gaussian")) if ib is not None else "linear_gaussian"
        )
        # No silent fallback: an unrecognised mode (e.g. a typo
        # "linear-gaussian") would otherwise slip through to the
        # linear-Gaussian path unnoticed (pitfall #9). Validate at
        # construction against the advertised set.
        if self.mode not in self.VALID_MODES:
            msg = (
                f"IBActiveAcquisitionStrategy: unknown mode {self.mode!r}; "
                f"expected one of {sorted(self.VALID_MODES)}."
            )
            raise ValueError(msg)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Standard reconstruction loss; IB policy is inference-only."""
        del epoch
        pred = self.generator_model(input_batch)
        l1 = F.l1_loss(pred, target_batch)
        l2 = F.mse_loss(pred, target_batch)
        total = l1 + 0.1 * l2
        return {
            "g_total_loss": total,
            "loss_l1": l1.detach(),
            "loss_l2": l2.detach(),
        }

    # ---- IB policy -------------------------------------------------- #

    @staticmethod
    def select_next_line(
        observed_mask: torch.Tensor,
        current_estimate: torch.Tensor,
        *,
        posterior_cov_inv: torch.Tensor | None = None,
        noise_var: float = 1.0,
        beta: float = 1.0,
        mode: str = "linear_gaussian",
        score_fn: callable | None = None,
    ) -> int:
        """Score every unobserved line and return the argmax.

        This is a *fast* implementation of the linear-Gaussian closed form
        L_IB(ℓ) = ½ log det(I + Σ_X|Y_obs A_ℓᵀ Σ_N,ℓ⁻¹ A_ℓ)
                − β · ½ log det(I + Σ_N,ℓ⁻¹).

        Args:
            observed_mask: 1-D bool tensor of shape [H] marking acquired
                phase-encode lines.
            current_estimate: Last reconstruction (used for shape only
                in the linear-Gaussian path).
            posterior_cov_inv: Posterior precision Σ_X|Y_obs⁻¹ for the
                current observed state. If None, defaults to the
                identity — the score then reduces to a no-prior baseline.
            noise_var: Per-coil noise variance σ_N².
            beta: IB tradeoff.

        Returns:
            Integer index ``ℓ★`` of the unobserved line to acquire.
        """
        H = observed_mask.shape[-1]
        unobserved_mask = ~observed_mask.bool()
        unobserved = unobserved_mask.nonzero(as_tuple=False).squeeze(-1)
        if unobserved.numel() == 0:
            return 0
        d = current_estimate.shape[-1]
        # Default to identity prior for tractability.
        if posterior_cov_inv is None:
            posterior_cov_inv = torch.eye(d, device=current_estimate.device)

        # No silent fallthrough: an unknown mode must raise here rather than
        # quietly running the linear-Gaussian branch (pitfall #9).
        if mode not in IBActiveAcquisitionStrategy.VALID_MODES:
            msg = (
                f"select_next_line: unknown mode {mode!r}; "
                f"expected one of {sorted(IBActiveAcquisitionStrategy.VALID_MODES)}."
            )
            raise ValueError(msg)

        if mode == "nonlinear":
            if score_fn is None:
                raise ValueError(
                    "IB nonlinear-mode line selection requires a callable score_fn "
                    "(typically the diffusion model's ∇log p_σ(x))."
                )
            # Solve sigma_inv x = I to obtain the posterior covariance
            # diagonal; for diagonal-dominant precision matrices the linear
            # solve is well-conditioned.
            try:
                sigma_post = torch.linalg.solve(
                    posterior_cov_inv,
                    torch.eye(d, device=posterior_cov_inv.device),
                )
            except RuntimeError as exc:
                logger.warning(
                    "IB nonlinear posterior solve failed (%s); falling back to "
                    "identity posterior covariance — the IB score loses its "
                    "prior signal for this step.",
                    exc,
                )
                sigma_post = torch.eye(d, device=posterior_cov_inv.device)
            scores = ib_score_nonlinear(
                score_fn=score_fn,
                current_estimate=current_estimate,
                candidate_lines=unobserved_mask,
                sigma_post=sigma_post,
                noise_var=noise_var,
                beta=beta,
            )
            best_idx = int(unobserved[scores.argmax()].item())
            return best_idx

        # ---- linear-Gaussian path (default) ---------------------- #
        # Diagonal of Σ_post is needed for the per-line determinant
        # update. We get it by one linear solve (Woodbury-equivalent on
        # the dense form); for very large d the caller should pass a
        # pre-computed factorisation via ``posterior_cov_inv``.
        try:
            diag = torch.linalg.solve(
                posterior_cov_inv,
                torch.eye(d, device=posterior_cov_inv.device),
            ).diagonal()
        except RuntimeError as exc:
            logger.warning(
                "IB linear-Gaussian posterior solve failed (%s); falling back "
                "to a no-prior identity diagonal — the IB score loses its prior "
                "signal for this step.",
                exc,
            )
            diag = torch.ones(d, device=current_estimate.device)
        log_det = 0.5 * torch.log(torch.clamp(1.0 + diag / (noise_var + 1e-12), min=1e-12))
        # Constant per-line noise cost (β-weighted), retained for
        # documentation completeness.
        cost = (
            beta
            * 0.5
            * torch.log(
                torch.tensor(1.0 + 1.0 / (noise_var + 1e-12), device=current_estimate.device)
            )
        )
        scores = log_det[unobserved] - cost
        best_idx = int(unobserved[scores.argmax()].item())
        return best_idx

    @staticmethod
    def update_posterior_precision(
        sigma_inv: torch.Tensor,
        A_ell: torch.Tensor,
        noise_var: float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper around :func:`posterior_covariance_woodbury_update`."""
        return posterior_covariance_woodbury_update(sigma_inv, A_ell, noise_var)


__all__ = ["IBActiveAcquisitionStrategy"]
