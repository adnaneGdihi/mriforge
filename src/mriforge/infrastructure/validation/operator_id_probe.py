r"""Tier-2 forward probe for the operator-identification paradigm (Proposal 1).

The generic :func:`mriforge.infrastructure.validation.forward_probe.synthetic_forward_probe`
cannot meaningfully exercise an ``operator_id`` arm: the ``bch_operator_conditioner``
returns a *parameter dictionary* (severities, PSD, gains), not an image, so the
generic image-shape contract does not apply. This module supplies the
paradigm-specific probe the audit ladder needs.

On a Shepp-Logan phantom with the configured ``mode_dictionary`` and a random
severity vector, it asserts the four correctness properties from the
implementation plan:

* **(i) operator action** — :math:`\exp(\Omega(s))x` has the same shape as
  :math:`x` and contains only finite values;
* **(ii) trainable likelihood** — the gradient of the structured-Gaussian NLL
  with respect to the conditioner parameters is non-zero (the operator is
  actually fitted, not a dead pass-through);
* **(iii) adjoint identity** — every generator satisfies
  :math:`\langle L_k u, v\rangle = \langle u, L_k^{H} v\rangle` to ``1e-4``
  (the gating correctness risk for ``gradcheck``);
* **(iv) identity at zero severity** — at :math:`s\equiv 0` the operator
  reduces to the identity, the consistency check from the proof.

These are the same properties verified off-line in
``tests/physics/test_operator_id_physics.py``; running them here promotes the
guarantee from "CI tests pass" to "the audit refuses a misconfigured arm
before training starts".
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from mriforge.infrastructure.validation.forward_probe import ProbeResult

__all__ = ["operator_id_forward_probe"]

# Adjoint dot-product test tolerance (complex64 single precision).
_ADJOINT_ATOL = 1.0e-4
# Identity-at-zero-severity tolerance.
_IDENTITY_ATOL = 1.0e-5
# Cap the probe grid so the conditioner basis + Krylov action stay cheap.
_MAX_PROBE_SIDE = 24


def _resolve_op_cfg(config: Any):
    from mriforge.config.schemas.training.operator_id import OperatorIDConfig

    raw = getattr(getattr(config, "training", None), "operator_id", None)
    if raw is None:
        return None
    if isinstance(raw, OperatorIDConfig):
        return raw
    if isinstance(raw, dict):
        return OperatorIDConfig.model_validate(raw)
    if hasattr(raw, "model_dump"):
        return OperatorIDConfig.model_validate(raw.model_dump())
    return OperatorIDConfig.model_validate(dict(raw))


def _probe_grid(config: Any) -> tuple[int, int]:
    patch = list(
        getattr(getattr(getattr(config, "data", None), "sampling", None), "patch_size", None) or []
    )
    while len(patch) > 2 and patch[-1] == 1:
        patch = patch[:-1]
    if len(patch) >= 2 and all(isinstance(s, int) and s > 0 for s in patch[:2]):
        h, w = patch[0], patch[1]
    else:
        h, w = 16, 16
    return min(h, _MAX_PROBE_SIDE), min(w, _MAX_PROBE_SIDE)


def operator_id_forward_probe(config: Any, device: str = "cpu") -> ProbeResult:
    """Run the operator-identification Tier-2 probe; return a :class:`ProbeResult`.

    Args:
        config: A loaded ``TrainingSettings`` whose ``training.operator_id``
            block is present.
        device: ``"cpu"`` (default, CI-safe) or ``"cuda"``.
    """
    t0 = time.perf_counter()

    def _fail(category: str, message: str, fix_hint: str | None = None, tb: str | None = None):
        return ProbeResult(
            passed=False,
            category=category,
            message=message,
            severity="error",
            yaml_keys=["training.operator_id.mode_dictionary", "model.model_type"],
            fix_hint=fix_hint,
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=tb,
        )

    try:
        import torch
    except ImportError:
        return ProbeResult(
            passed=True,
            category="no_probe",
            message="PyTorch unavailable — operator_id Tier-2 probe skipped.",
            severity="info",
            elapsed_seconds=time.perf_counter() - t0,
        )

    op_cfg = _resolve_op_cfg(config)
    if op_cfg is None:
        return ProbeResult(
            passed=True,
            category="no_probe",
            message="No training.operator_id block — operator_id probe not applicable.",
            severity="info",
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
        )

    try:
        from mriforge.infrastructure.physics.degradation_generators import build_generators
        from mriforge.infrastructure.physics.magnus_exponential import (
            assemble_omega,
            expm_multiply,
        )
        from mriforge.infrastructure.physics.structured_covariance import radial_to_grid
        from mriforge.infrastructure.validation.phantom_builder import synthetic_phantom
        from mriforge.models.losses.complex.structured_gaussian_nll import StructuredGaussianNLL
        from mriforge.models.registry import get_model_class
    except Exception as exc:
        return _fail(
            "instantiation", f"operator_id probe imports failed: {exc}", tb=traceback.format_exc()
        )

    h, w = _probe_grid(config)
    cdtype = torch.complex64

    # ── Build the deterministic generator basis ───────────────────────
    try:
        generators, det_modes, _noise = build_generators(
            list(op_cfg.mode_dictionary), h, w, device=device, dtype=cdtype
        )
    except Exception as exc:
        return _fail(
            "operator_basis",
            f"build_generators failed for mode_dictionary={op_cfg.mode_dictionary}: {exc}",
            fix_hint="Every deterministic mode must resolve to a registered generator adapter.",
            tb=traceback.format_exc(),
        )
    if not generators:
        return _fail(
            "operator_basis",
            "mode_dictionary has no deterministic generators; exp(Ω) cannot be formed.",
            fix_hint="Add at least one non-noise mode (e.g. motion_rigid, b0_phase).",
        )
    k = len(generators)

    # ── (iii) Per-generator adjoint identity ⟨L u, v⟩ = ⟨u, L^H v⟩ ─────
    torch.manual_seed(0)
    u_vec = torch.randn(1, 1, h, w, dtype=cdtype, device=device)
    v_vec = torch.randn(1, 1, h, w, dtype=cdtype, device=device)
    for name, g in zip(det_modes, generators, strict=True):
        try:
            lhs = (g(u_vec).conj() * v_vec).sum()
            rhs = (u_vec.conj() * g.adjoint(v_vec)).sum()
        except Exception as exc:
            return _fail(
                "adjoint_identity",
                f"generator {name!r} failed to apply/adjoint: {exc}",
                tb=traceback.format_exc(),
            )
        if not torch.allclose(lhs, rhs, atol=_ADJOINT_ATOL, rtol=1e-3):
            return _fail(
                "adjoint_identity",
                f"generator {name!r} violates ⟨Lu,v⟩=⟨u,Lᴴv⟩ "
                f"(|Δ|={float((lhs - rhs).abs()):.2e} > {_ADJOINT_ATOL:g}). "
                "gradcheck through the Magnus action will be unreliable.",
                fix_hint=f"Fix the rmatvec (adjoint) of the {name!r} generator adapter.",
            )

    # ── Build the conditioner exactly as the strategy would ───────────
    try:
        # Ensure the registry is populated even when the probe is invoked
        # standalone (the CLI's from_yaml path populates it in production).
        from mriforge.models.init_registry import populate_model_registry

        populate_model_registry()
        cls = get_model_class("bch_operator_conditioner")
        conditioner = cls(
            num_modes=k,
            num_contrasts=int(op_cfg.num_contrasts),
            covariance_rank=int(op_cfg.covariance_rank),
            rho_bins=int(op_cfg.rho_bins),
            channels=1,
            height=h,
            width=w,
        ).to(device)
        conditioner.train()
    except Exception as exc:
        return _fail(
            "instantiation",
            f"bch_operator_conditioner construction failed: {exc}",
            tb=traceback.format_exc(),
        )

    # ── Phantom + conditioning coordinate ─────────────────────────────
    try:
        x = synthetic_phantom((1, 1, h, w), device=device, dtype="complex64", add_phase=True)
    except Exception:
        x = torch.complex(
            torch.rand(1, 1, h, w, device=device), torch.zeros(1, 1, h, w, device=device)
        )
    contrast_idx = torch.zeros(1, dtype=torch.long, device=device)
    beta = float(torch.log10(torch.tensor(0.3)))
    field_coord = torch.full((1,), beta, device=device)

    out = conditioner(contrast_idx=contrast_idx, field_coord=field_coord)
    severities = out["severities"]  # [1, K]
    rho_radial = out["rho_radial"]
    gains = out["gains"]

    # ── (iv) Identity at zero severity: exp(Ω(0)) x ≈ x ───────────────
    try:
        omega0 = assemble_omega(
            generators, torch.zeros_like(severities), order=int(op_cfg.bch_order)
        )
        x0 = expm_multiply(omega0, x, krylov_dim=int(op_cfg.krylov_dim))
    except Exception as exc:
        return _fail(
            "operator_action", f"exp(Ω(0)) action failed: {exc}", tb=traceback.format_exc()
        )
    if not torch.allclose(x0, x, atol=_IDENTITY_ATOL, rtol=1e-4):
        return _fail(
            "identity_consistency",
            f"exp(Ω(0)) is not the identity (max|Δ|={float((x0 - x).abs().max()):.2e}). "
            "The operator does not reduce to identity at zero severity.",
            fix_hint="Check assemble_omega scaling and the Krylov action at s≡0.",
        )

    # ── (i) Operator action at the predicted severity: shape + finite ─
    try:
        omega = assemble_omega(generators, severities, order=int(op_cfg.bch_order))
        x_deg = expm_multiply(omega, x, krylov_dim=int(op_cfg.krylov_dim))
    except Exception as exc:
        return _fail(
            "operator_action", f"exp(Ω(s)) action failed: {exc}", tb=traceback.format_exc()
        )
    if x_deg.shape != x.shape:
        return _fail(
            "operator_action",
            f"exp(Ω)x shape {tuple(x_deg.shape)} != input {tuple(x.shape)}.",
        )
    if not torch.isfinite(x_deg.abs()).all():
        return _fail(
            "operator_action",
            "exp(Ω)x contains NaN/Inf — likely a stiff generator; raise krylov_dim "
            "or enable scaling-and-squaring.",
            fix_hint="Increase training.operator_id.krylov_dim or reduce severity scale.",
        )

    # ── (ii) NLL gradient w.r.t. the conditioner is non-zero ──────────
    try:
        rho_grid = radial_to_grid(rho_radial.to(torch.float32), h, w)
        low_u = conditioner.build_lowrank(gains)
        residual = x - x_deg  # no observed y at probe time; -x_deg is a valid residual
        nll = StructuredGaussianNLL()(residual, rho_grid, low_u, channels=1, height=h, width=w)
        conditioner.zero_grad(set_to_none=True)
        nll.backward()
    except Exception as exc:
        return _fail("nll_backward", f"NLL backward failed: {exc}", tb=traceback.format_exc())

    grad_sq = 0.0
    for p in conditioner.parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.detach().pow(2).sum())
    if grad_sq <= 0.0:
        return _fail(
            "nll_backward",
            "NLL gradient w.r.t. the conditioner is exactly zero — the operator "
            "is not being fitted (dead conditioning path).",
            fix_hint="Check that severities/rho/gains flow into the loss with requires_grad.",
        )

    return ProbeResult(
        passed=True,
        category="operator_id_probe",
        message=(
            f"operator_id: {k} generators on {h}x{w} — adjoint identity, "
            f"exp(Ω(0))=I, exp(Ω(s)) finite, and NLL grad-norm "
            f"{grad_sq**0.5:.2e} all OK on {device}."
        ),
        severity="info",
        device=device,
        elapsed_seconds=time.perf_counter() - t0,
    )
