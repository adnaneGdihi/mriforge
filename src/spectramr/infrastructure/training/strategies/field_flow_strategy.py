"""Neural-ODE field-flow translation (MICCAI MRIxFields2026, B-3.1).

Translation is transport along the physical field coordinate ``tau = log10(B0)``: a
single velocity field ``v_theta(x, B0)`` predicts ``dx/dtau``, and translating from
``b_s`` to ``b_t`` integrates that field over ``[tau_s, tau_t]``. Because the field
is parameterised per-unit-tau and integration is over tau, the continuous flow
obeys the semigroup law ``Phi_{tau_s->tau_t} = Phi_{tau_m->tau_t} o Phi_{tau_s->tau_m}``
(Picard-Lindelof). The fixed-step Euler/Heun integrator is a *consistent*
discretisation: composition holds up to integration error in general, and EXACTLY
for a constant velocity field (the property the oracle semigroup test verifies).

The strategy subclasses :class:`ReconstructionTrainingStrategy` (a non-diffusion
sibling) so it inherits the train/optimizer plumbing without the diffusion
``timesteps`` coupling (which would otherwise be a pitfall-#20 facade). The pure
helpers hold the testable math.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)

_TAU_SPAN_EPS = 1.0e-3  # guard against a degenerate equal-field pair (span -> 0)


def _log10(b: torch.Tensor) -> torch.Tensor:
    return torch.log10(b.clamp_min(1.0e-6))


def compute_field_flow_losses(
    model: Any,
    batch: dict[str, Any],
    lambda_straightness: float = 0.0,
    velocity_norm: str = "l2",
) -> dict[str, torch.Tensor]:
    """Conditional-flow-matching velocity regression along the field axis.

    Args:
        model: velocity field ``v(x, field_strength=…) -> dx/dtau`` (same shape as x).
        batch: dict with ``input`` (x_s), ``target`` (x_t), ``field_strength`` (b_s),
            ``field_strength_target`` (b_t).
        lambda_straightness: weight of the transport-cost penalty ``mean(||v||^2)``.
        velocity_norm: ``'l2'`` (default) or ``'l1'`` — the registered
            :class:`FieldFlowVelocityLoss` norm (so the YAML/field_flow knob is live
            and the registered loss is the one actually invoked, not a facade).

    Returns:
        dict with ``loss_total``, ``loss_velocity``, ``loss_straightness``.
    """
    # Lazy import avoids a models<-infrastructure import cycle at module load.
    from spectramr.models.losses.cross_field_losses import FieldFlowVelocityLoss

    velocity_loss_fn = FieldFlowVelocityLoss(norm=velocity_norm)
    x_s = batch["input"]
    x_t = batch["target"]
    tau_s = _log10(batch["field_strength"].float()).reshape(-1, 1, 1, 1)
    tau_t = _log10(batch["field_strength_target"].float()).reshape(-1, 1, 1, 1)
    span = tau_t - tau_s
    # Guard a near-degenerate equal-field pair (|span| -> 0) while PRESERVING sign
    # (a descending b_t < b_s pair has negative span; forcing +eps would flip the
    # velocity direction). copysign keeps the original direction.
    eps_signed = torch.copysign(
        torch.full_like(span, _TAU_SPAN_EPS), torch.where(span == 0, torch.ones_like(span), span)
    )
    span = torch.where(span.abs() < _TAU_SPAN_EPS, eps_signed, span)

    s = torch.rand(x_s.shape[0], 1, 1, 1, device=x_s.device, dtype=x_s.dtype)
    x_path = (1.0 - s) * x_s + s * x_t
    field_now = torch.pow(10.0, tau_s + s * span).reshape(-1)  # current B0 (Tesla)
    u = (x_t - x_s) / span  # per-unit-tau conditional velocity (secant slope)

    # Thread the per-sample contrast id (mrixfields emits it); the model uses it only
    # when use_contrast_conditioning is on, else ignores it (raises if on but absent, #15).
    v = model(x_path, field_strength=field_now, contrast_id=batch.get("contrast_id"))
    loss_velocity = velocity_loss_fn(prediction=v, target=u)
    if lambda_straightness > 0.0:
        loss_straightness = lambda_straightness * torch.mean(v**2)
    else:
        loss_straightness = v.new_zeros(())
    return {
        "loss_total": loss_velocity + loss_straightness,
        "loss_velocity": loss_velocity,
        "loss_straightness": loss_straightness,
    }


@torch.no_grad()
def integrate_field_flow(
    model: Any,
    x0: torch.Tensor,
    b_s: float | torch.Tensor,
    b_t: float | torch.Tensor,
    n_steps: int = 20,
    solver: str = "heun",
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate ``dx/dtau = v(x, 10^tau)`` from field ``b_s`` to ``b_t``.

    ``b_s``/``b_t`` may be scalars or per-sample ``[B]`` tensors (Tesla).
    ``contrast_id`` (optional) is threaded to the velocity model's FiLM conditioning.
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"solver must be 'euler' or 'heun'; got {solver!r}")
    dev, dt = x0.device, x0.dtype
    b = x0.shape[0]

    def _as_tau(value: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(value, device=dev, dtype=dt)
        if t.ndim == 0:
            t = t.expand(b)
        return _log10(t.float()).to(dt)

    tau_s = _as_tau(b_s)
    tau_t = _as_tau(b_t)
    dtau = (tau_t - tau_s) / n_steps  # [B]
    dtau_v = dtau.reshape(-1, 1, 1, 1)

    x = x0
    for k in range(n_steps):
        tau_k = tau_s + k * dtau
        field_k = torch.pow(10.0, tau_k)
        v = model(x, field_strength=field_k, contrast_id=contrast_id)
        if solver == "euler":
            x = x + v * dtau_v
        else:  # heun: predictor + corrector
            x_pred = x + v * dtau_v
            field_next = torch.pow(10.0, tau_s + (k + 1) * dtau)
            v_next = model(x_pred, field_strength=field_next, contrast_id=contrast_id)
            x = x + 0.5 * (v + v_next) * dtau_v
    return x


class FieldFlowStrategy(ReconstructionTrainingStrategy):
    """Train a field-axis velocity field by CFM velocity regression (B-3.1)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"field_flow_velocity"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_flow", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_flow", None)
        self._lambda_straightness = (
            float(getattr(cfg, "lambda_straightness", 0.0)) if cfg is not None else 0.0
        )
        self._ff_n_steps = int(getattr(cfg, "n_steps", 20)) if cfg is not None else 20
        self._ff_solver = str(getattr(cfg, "solver", "heun")) if cfg is not None else "heun"
        self._ff_velocity_norm = (
            str(getattr(cfg, "velocity_norm", "l2")) if cfg is not None else "l2"
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
                "FieldFlowStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        lam = getattr(self, "_lambda_straightness", 0.0)
        return compute_field_flow_losses(
            self.env.generator,
            batch,
            lambda_straightness=lam,
            velocity_norm=getattr(self, "_ff_velocity_norm", "l2"),
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
        """Fold the per-sample source/target fields into ``batch_context``.

        Declaring ``field_strength`` / ``field_strength_target`` / ``contrast_id``
        here makes the pipeline's gated validation seam forward them (it only forwards
        fields the strategy's ``validation_step`` signature declares); they are then
        available to :meth:`_validation_forward`.
        """
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
        """Validation prediction = the *integrated* field-axis translation.

        A velocity-field forward is not an image, so comparing it to the target
        would be a metric mismatch (pitfall #18). Integrate dx/dtau from b_s to b_t
        to obtain the translated image the image metrics (SSIM/PSNR/LPIPS) score.
        Reads the fields from ``batch_context`` (folded in by :meth:`validation_step`)
        or ``kwargs``; raises loudly if absent (no silent fallback, pitfall #9).
        """
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "FieldFlowStrategy validation requires per-sample 'field_strength' "
                "and 'field_strength_target'. Set data.expose_field_strength on the "
                "mrixfields dataset (which emits both). Got "
                f"field_strength={b_s!r}, field_strength_target={b_t!r}."
            )
        return integrate_field_flow(
            self.env.generator,
            x0,
            b_s,
            b_t,
            n_steps=getattr(self, "_ff_n_steps", 20),
            solver=getattr(self, "_ff_solver", "heun"),
            contrast_id=batch_context.get("contrast_id", kwargs.get("contrast_id")),
        )
