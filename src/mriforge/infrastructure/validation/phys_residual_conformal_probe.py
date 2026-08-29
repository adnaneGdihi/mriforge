r"""Tier-2 probe for PR-CC (Physics-Residual Conformal hallucination certificate).

The probe is the locally-runnable, scientifically-rigorous proof that PR-CC
*fires* (pitfall #16 guard). On a phantom with KNOWN (PD, T1, T2) it:

1. **Mechanism check** -- calibrates the certificate on noisy copies of the
   rendered contrast, verifies empirical coverage meets the :math:`1-\alpha`
   target, then injects a high-residual "hallucination" block and asserts it
   lands in the flagged set. Because the parameters are known, this residual is
   physically meaningful (unlike on single-field magnitude data).
2. **Contract check** -- builds the configured model and asserts it actually
   outputs a (rho, T1, T2) parameter map (via ``predict_parameters`` or a
   3-channel forward); otherwise the residual would be ill-defined.

Returns a :class:`ProbeResult`; never raises (the audit CLI maps it to an exit
code). A coverage/flagging violation or a non-param-map model returns
``passed=False, severity="error"`` -- a hard gate.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import torch

from mriforge.infrastructure.calibration.phys_residual_certificate import (
    PhysResidualConformalCertificate,
)
from mriforge.infrastructure.validation.forward_probe import ProbeResult

_CATEGORY = "phys_residual_conformal_coverage"
_YAML_KEYS = ["training.phys_residual_conformal"]


def _phantom_params(device: Any, h: int = 24, w: int = 24) -> torch.Tensor:
    """A ``[1, 3, h, w]`` phantom with known smooth (rho, T1_ms, T2_ms)."""
    yy, xx = torch.meshgrid(torch.linspace(0.4, 1.0, h), torch.linspace(0.5, 1.0, w), indexing="ij")
    rho = (0.6 + 0.4 * xx).unsqueeze(0).unsqueeze(0)
    t1 = (600.0 + 800.0 * yy).unsqueeze(0).unsqueeze(0)
    t2 = (60.0 + 80.0 * xx).unsqueeze(0).unsqueeze(0)
    return torch.cat([rho, t1, t2], dim=1).to(device)


def _render(
    params: torch.Tensor, contrast_type: str, acquisition: dict[str, float]
) -> torch.Tensor:
    from mriforge.infrastructure.calibration.scores import _CONTRAST_TO_SEQ
    from mriforge.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    seq = _CONTRAST_TO_SEQ.get(str(contrast_type).lower(), 0)
    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False).to(params.device)
    return layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {
            "TR": float(acquisition["TR"]),
            "TE": float(acquisition["TE"]),
            "TI": float(acquisition.get("TI", 0.0)),
            "contrast_type": int(seq),
        },
    )


def _check_param_map_model(config: Any, device: Any) -> tuple[bool, str]:
    """Build the configured model and verify it outputs a (rho, T1, T2) map."""
    try:
        from mriforge.infrastructure.validation.equivariant_imaging_probe import (
            _build_model,
        )

        model = _build_model(config).to(device).eval()
    except Exception as exc:
        return False, f"could not build the configured model: {exc}"

    x = torch.randn(1, int(getattr(config.model, "in_channels", 1) or 1), 16, 16, device=device)
    with torch.no_grad():
        if hasattr(model, "predict_parameters"):
            try:
                pd, t1, t2 = model.predict_parameters(x)
            except Exception as exc:
                return False, f"predict_parameters() failed: {exc}"
            if pd.shape[1] == t1.shape[1] == t2.shape[1] == 1:
                return True, "model exposes predict_parameters() -> (rho, T1, T2)"
            return False, "predict_parameters() did not return single-channel (rho, T1, T2)"
        try:
            out = model(x)
        except Exception as exc:
            return False, f"forward() failed (and no predict_parameters): {exc}"
        if out.ndim >= 2 and out.shape[1] == 3:
            return True, "model forward outputs a 3-channel (rho, T1, T2) map"
        return False, (
            f"model outputs shape {tuple(out.shape)}, not a (rho, T1, T2) parameter "
            "map; PR-CC requires predict_parameters() or a 3-channel forward"
        )


def phys_residual_conformal_probe(config: Any, device: str = "cpu") -> ProbeResult:
    """Run the PR-CC Tier-2 mechanism + contract probe."""
    t0 = time.perf_counter()

    def _fail(message: str, fix_hint: str | None = None, tb: str | None = None) -> ProbeResult:
        return ProbeResult(
            passed=False,
            category=_CATEGORY,
            message=message,
            severity="error",
            yaml_keys=_YAML_KEYS,
            fix_hint=fix_hint,
            device=str(device),
            elapsed_seconds=time.perf_counter() - t0,
            traceback=tb,
        )

    try:
        cfg = getattr(getattr(config, "training", None), "phys_residual_conformal", None)
        if cfg is None:
            return _fail(
                "no training.phys_residual_conformal block to probe.",
                fix_hint="add a phys_residual_conformal block with tr_ms/te_ms.",
            )
        if isinstance(cfg, dict):
            from mriforge.config.schemas.training.phys_residual_conformal import (
                PhysResidualConformalConfig,
            )

            cfg = PhysResidualConformalConfig(**cfg)

        acquisition = {"TR": cfg.tr_ms, "TE": cfg.te_ms, "TI": cfg.ti_ms}
        alpha = cfg.alpha

        # --- mechanism check (known phantom) ---
        torch.manual_seed(0)
        params = _phantom_params(device)
        clean = _render(params, cfg.contrast_type, acquisition)
        sigma = 0.03

        cert = PhysResidualConformalCertificate(
            alpha=alpha, acquisition=acquisition, contrast_type=cfg.contrast_type
        )
        cert.calibrate([(params, clean + sigma * torch.randn_like(clean), 0.3) for _ in range(40)])
        covs = [
            cert.certify(params, clean + sigma * torch.randn_like(clean)).coverage
            for _ in range(40)
        ]
        coverage = sum(covs) / len(covs)
        target = 1.0 - alpha
        if coverage < target - 0.07:
            return _fail(
                f"PR-CC empirical coverage {coverage:.3f} is below the {target:.2f} "
                "target on a known phantom -- the certificate is mis-calibrated.",
                fix_hint="check the score / quantile path in PhysResidualConformalCertificate.",
            )

        corrupted = clean.clone()
        corrupted[..., :6, :6] += 0.6  # injected hallucination block
        res = cert.certify(params, corrupted)
        if not bool(res.flagged[..., :6, :6].all().item()):
            return _fail(
                "PR-CC did not flag an injected hallucination block -- the flagged "
                "set does not cover a known high-residual perturbation.",
                fix_hint="verify the flagged map s(v) > q in the certificate.",
            )

        # --- contract check (configured model) ---
        model_ok, model_msg = _check_param_map_model(config, device)
        if not model_ok:
            return _fail(
                f"PR-CC model contract failed: {model_msg}.",
                fix_hint="use a base model with predict_parameters() or a 3-channel "
                "(rho, T1, T2) output (e.g. bloch_field_bottleneck, qmap_generator).",
            )

        return ProbeResult(
            passed=True,
            category=_CATEGORY,
            message=(
                f"PR-CC fires: phantom coverage={coverage:.3f} (target {target:.2f}), "
                f"injected hallucination flagged; {model_msg}."
            ),
            severity="info",
            device=str(device),
            elapsed_seconds=time.perf_counter() - t0,
        )
    except Exception as exc:
        return _fail(f"PR-CC probe crashed: {exc}", tb=traceback.format_exc())


__all__ = ["phys_residual_conformal_probe"]
