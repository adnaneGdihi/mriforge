"""Field-conditioned stochastic bridge (MICCAI MRIxFields2026, B-3.3).

A single velocity field, conditioned on the field coordinate along the path,
transports between any two field marginals (any-to-any). Distinct from the
deterministic field-flow (B-3.1) by its **stochastic (Brownian-bridge) interpolant**:
during training the interpolant carries a noise term
``gamma(s) = sigma_bridge * sqrt(s(1-s))`` (zero at both endpoints), and inference
uses a stochastic posterior-mean re-noising sampler. ``sigma_bridge = 0`` recovers
the deterministic flow, so the knob directly tests the stochastic bridge against it.

This is bridge-MATCHING (an I2SB-style fixed Brownian-bridge reference / stochastic
interpolant), NOT a true entropic-OT Schrödinger bridge — there is no
Sinkhorn/iterative-proportional-fitting coupling refinement. The honest claim is a
field-conditioned stochastic bridge that models translation uncertainty the
deterministic flow cannot; we do not claim entropic optimality.

The velocity target is the per-unit-tau secant ``(x_t - x_s)/(tau_t - tau_s)``
(``tau = log10 B0``), reusing the field-flow conditioning and the registered
``FieldFlowVelocityLoss`` (DRY); the bridge adds the interpolant noise + stochastic
sampling. Pure helpers hold the testable math.
"""

from __future__ import annotations

from typing import Any

import torch

from .field_flow_strategy import _TAU_SPAN_EPS, _log10
from .reconstruction import ReconstructionTrainingStrategy


def compute_field_bridge_losses(
    model: Any,
    batch: dict[str, Any],
    sigma_bridge: float = 0.1,
    velocity_norm: str = "l2",
    lambda_endpoint_l1: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Velocity matching on a stochastic (Brownian-bridge) field interpolant.

    When ``lambda_endpoint_l1 > 0`` an ENDPOINT L1 anchor is added: from the sampled
    interpolant point the model's drift gives a one-step endpoint estimate
    ``x_hat_t = x_path + v * (tau_t - tau_now)``; anchoring ``|x_hat_t - x_t|`` trains the
    DENOISER property the stochastic sampler relies on (``bridge_sample`` assumes
    ``x_hat_t ≈ x_t`` for a noisy path point). This is the anti-degeneracy guard (#20) that
    keeps the stochastic bridge from drifting to a noise-dominated / measurement-independent
    output; upweighting it relative to the pure velocity (bridge) term grounds the transport
    in the true target. Applied identically in both arms so the one-knob stays ``sigma_bridge``.
    """
    from mriforge.models.losses.cross_field_losses import FieldFlowVelocityLoss

    x_s = batch["input"]
    x_t = batch["target"]
    tau_s = _log10(batch["field_strength"].float()).reshape(-1, 1, 1, 1)
    tau_t = _log10(batch["field_strength_target"].float()).reshape(-1, 1, 1, 1)
    span = tau_t - tau_s
    eps_signed = torch.copysign(
        torch.full_like(span, _TAU_SPAN_EPS),
        torch.where(span == 0, torch.ones_like(span), span),
    )
    span = torch.where(span.abs() < _TAU_SPAN_EPS, eps_signed, span)

    s = torch.rand(x_s.shape[0], 1, 1, 1, device=x_s.device, dtype=x_s.dtype)
    gamma = sigma_bridge * torch.sqrt((s * (1.0 - s)).clamp_min(0.0))
    x_path = (1.0 - s) * x_s + s * x_t + gamma * torch.randn_like(x_s)
    field_now = torch.pow(10.0, tau_s + s * span).reshape(-1)
    u = (x_t - x_s) / span

    # Thread the per-sample contrast id (mrixfields emits it); the widened velocity
    # model uses it only when use_contrast_conditioning is on, else ignores it (raises
    # if on but absent, #15). Mirrors compute_field_flow_losses.
    v = model(x_path, field_strength=field_now, contrast_id=batch.get("contrast_id"))
    loss_velocity = FieldFlowVelocityLoss(norm=velocity_norm)(prediction=v, target=u)
    out = {"loss_total": loss_velocity, "loss_velocity": loss_velocity}
    if lambda_endpoint_l1 > 0.0:
        remaining = (1.0 - s) * span  # tau_t - tau_now, per-sample [B,1,1,1]
        x_hat_t = x_path + v * remaining  # one-step endpoint estimate via the learned drift
        loss_endpoint = torch.mean(torch.abs(x_hat_t - x_t))
        out["loss_endpoint_l1"] = loss_endpoint
        out["loss_total"] = loss_velocity + lambda_endpoint_l1 * loss_endpoint
    return out


@torch.no_grad()
def bridge_sample(
    model: Any,
    x_start: torch.Tensor,
    b_s: float | torch.Tensor,
    b_t: float | torch.Tensor,
    n_steps: int = 20,
    sigma_bridge: float = 0.1,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stochastic bridge sampler via posterior-mean re-noising (Bansal/I2SB-style).

    ``contrast_id`` (optional) is threaded to the velocity model's FiLM conditioning
    (the widened FieldVelocityUNet uses it only when ``use_contrast_conditioning``).

    At each step the model's per-unit-tau drift gives an endpoint estimate
    ``x_hat_t = x + v·(tau_t - tau_k)``; we then re-noise it onto the same
    Brownian-bridge interpolant used in training:
    ``x = (1-s')·x_start + s'·x_hat_t + sigma·sqrt(s'(1-s'))·z``. Because the noise
    schedule ``gamma(s)=sigma·sqrt(s(1-s))`` is identically zero at the final ``s'=1``,
    the output equals the model's last-step endpoint estimate ``x_hat_t`` with NO
    terminal noise injection (unlike the noise-accumulating integrate-then-add-noise
    scheme, which carries the penultimate noise into the output). Pinning to the TRUE
    target additionally requires the model to act as a denoiser (``x_hat_t≈x_t`` for a
    noisy ``x``), which a trained bridge learns. ``sigma_bridge=0`` is deterministic and
    recovers the field-flow endpoint. This keeps the sampler's marginals consistent
    with the training interpolant.
    """
    dev, dt = x_start.device, x_start.dtype
    b = x_start.shape[0]

    def _as_tau(value: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(value, device=dev, dtype=dt)
        if t.ndim == 0:
            t = t.expand(b)
        return _log10(t.float()).to(dt)

    tau_s = _as_tau(b_s)
    tau_t = _as_tau(b_t)
    span = (tau_t - tau_s).reshape(-1, 1, 1, 1)

    x = x_start
    for k in range(n_steps):
        s_k = k / n_steps
        tau_k = tau_s + s_k * (tau_t - tau_s)
        v = model(x, field_strength=torch.pow(10.0, tau_k), contrast_id=contrast_id)
        remaining = (1.0 - s_k) * span  # tau_t - tau_k, per-sample
        x_hat_t = x + v * remaining  # endpoint estimate via the learned drift
        s_next = (k + 1) / n_steps
        x = (1.0 - s_next) * x_start + s_next * x_hat_t
        if sigma_bridge > 0.0 and s_next < 1.0:
            gamma = sigma_bridge * (s_next * (1.0 - s_next)) ** 0.5
            x = x + gamma * torch.randn_like(x)
    return x


class FieldBridgeStrategy(ReconstructionTrainingStrategy):
    """Train a field-conditioned stochastic bridge (B-3.3, bridge-matching).

    Not a true entropic-OT Schrödinger bridge (no Sinkhorn/IPF coupling) — see the
    module docstring. ``sigma_bridge=0`` is the deterministic field-flow control.
    """

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_bridge", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_bridge", None)
        self._fb_sigma = float(getattr(cfg, "sigma_bridge", 0.1)) if cfg is not None else 0.1
        self._fb_n_steps = int(getattr(cfg, "n_steps", 20)) if cfg is not None else 20
        self._fb_norm = str(getattr(cfg, "velocity_norm", "l2")) if cfg is not None else "l2"
        # Anti-degeneracy endpoint L1 anchor (#20): trains the denoiser property so the
        # stochastic bridge cannot drift to a noise-dominated output.
        self._fb_lambda_endpoint = (
            float(getattr(cfg, "lambda_endpoint_l1", 0.0)) if cfg is not None else 0.0
        )
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"[B-3.3] field bridge: sigma_bridge={self._fb_sigma}, "
                f"lambda_endpoint_l1={self._fb_lambda_endpoint}"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "FieldBridgeStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        return compute_field_bridge_losses(
            self.env.generator,
            batch,
            sigma_bridge=getattr(self, "_fb_sigma", 0.1),
            velocity_norm=getattr(self, "_fb_norm", "l2"),
            lambda_endpoint_l1=getattr(self, "_fb_lambda_endpoint", 0.0),
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold per-sample fields into batch_context (the pipeline gated seam only
        forwards fields the signature declares)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = stochastic-bridge sampling from the source to the target field."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "FieldBridgeStrategy validation requires per-sample 'field_strength' "
                "and 'field_strength_target' (set data.expose_field_strength on the "
                f"mrixfields dataset). Got {b_s!r}, {b_t!r}."
            )
        return bridge_sample(
            self.env.generator,
            x0,
            b_s,
            b_t,
            n_steps=getattr(self, "_fb_n_steps", 20),
            sigma_bridge=getattr(self, "_fb_sigma", 0.1),
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        )
