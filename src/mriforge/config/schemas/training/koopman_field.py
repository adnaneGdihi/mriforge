"""Config schema for the Koopman linear field-propagator strategy (B-3.10)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class KoopmanFieldConfig(StrictSchema):
    """Knobs for the continuous Koopman cross-field propagator (B-3.10).

    The Koopman semigroup is STRUCTURAL on the model
    (``model_kwargs.use_koopman_semigroup`` — the one-knob ablation, and
    ``model_kwargs.koopman_dim`` — the observable dimension); this block carries the loss
    weights only.
    """

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 loss on the field-propagated prediction vs target.",
    )
    lambda_recon: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Weight of the autoencoder-consistency loss (dtau=0 -> psi(phi(x_s)) ~ x_s), a soft "
            "consistency regulariser that pushes the encoder/decoder toward APPROXIMATE "
            "invertibility (NOT a hard guarantee); raise toward 1.0 for stricter consistency."
        ),
    )
    lambda_stability: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the operator-stability penalty relu(mu2(A)) on the learned infinitesimal "
            "generator A, where mu2(A)=lambda_max(0.5(A+A^T)) is the numerical abscissa — a "
            "DIFFERENTIABLE upper bound on the spectral abscissa max Re(lambda(A)). Drives the "
            "continuous semigroup exp(dtau*A) toward non-expansive (mu2<=0). Structurally N/A "
            "for the nonlinear ablation (use_koopman_semigroup=false, no operator A) -> the term "
            "is skipped there, so it may carry the same value in both arms without breaking the "
            "one-knob contract. 0 disables (default)."
        ),
    )
