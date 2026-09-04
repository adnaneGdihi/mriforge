"""``design-mrf-sequence`` CLI subcommand (MRF §4).

Wraps the :class:`CRLBMRFPulseDesignStrategy` for direct
command-line invocation. Reads a target-parameter prior from a small
YAML and emits a pulse-sequence YAML suitable for downstream
scanner-environment conversion.

Usage::

    python -m spectramr.cli design-mrf-sequence \\
        --target-params target_prior.yaml \\
        --n-tr 500 \\
        --output designed_sequence.yaml

References:
    [18] B. Zhao et al., "Optimal experiment design for MRF: CRLB
         meets spin dynamics", *IEEE TMI*, 38(3), 2019.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

# NOTE: ``torch`` is imported lazily inside the handlers below so that
# ``attach_subparsers`` (invoked from ``cli.app`` to list this
# subcommand under ``--help``) stays torch-free. ``from __future__
# import annotations`` keeps the ``torch.device`` / ``torch.Tensor``
# hints as strings, so they do not force an eager import either; the
# ``TYPE_CHECKING`` block makes ``torch`` resolvable for static
# type-checkers and linters without importing it at runtime.
if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _load_target_prior(path: Path) -> dict:
    """Read a small YAML containing the parameter prior bounds.

    Expected layout::

        T1_range:  [0.3, 3.0]   # seconds
        T2_range:  [0.02, 0.3]
        M0_range:  [0.5, 1.5]
        B0_range:  [-0.1, 0.1]   # ppm
        B1_range:  [0.7, 1.3]
        n_samples: 32

    Anything missing is filled with conservative whole-brain-tissue
    defaults so the CLI is usable with a one-line prior.
    """
    if not path.exists():
        raise FileNotFoundError(f"target prior file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    defaults = {
        "T1_range": [0.3, 3.0],
        "T2_range": [0.02, 0.3],
        "M0_range": [0.5, 1.5],
        "B0_range": [-0.1, 0.1],
        "B1_range": [0.7, 1.3],
        "n_samples": 32,
    }
    out = {**defaults, **raw}
    return out


def _sample_parameter_prior(prior: dict, device: torch.device) -> torch.Tensor:
    """Draw ``[N, 5]`` samples from the uniform parameter prior."""
    import torch

    n = int(prior["n_samples"])
    cols = []
    for key in ("T1_range", "T2_range", "M0_range", "B0_range", "B1_range"):
        lo, hi = prior[key]
        cols.append(torch.empty(n, device=device).uniform_(float(lo), float(hi)))
    return torch.stack(cols, dim=-1)


def design_mrf_sequence_cmd(args: argparse.Namespace) -> int:
    """Entry point for ``design-mrf-sequence``.

    Returns:
        ``0`` on successful design, non-zero on error.
    """
    import torch

    try:
        prior = _load_target_prior(args.target_params)
    except FileNotFoundError as exc:
        print(f"design-mrf-sequence: {exc}", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    theta_samples = _sample_parameter_prior(prior, device)
    # Initialise an unconstrained pulse-sequence tensor. The optimisation
    # is left as a single-shot SGD loop here; production callers pull this
    # into a full TrainingSettings + strategy harness instead.
    pulse = torch.zeros(args.n_tr, 2, device=device, requires_grad=True)
    pulse.data[:, 0] = torch.linspace(0.1, 1.0, args.n_tr, device=device)  # alpha
    pulse.data[:, 1] = 1.0 + 0.0 * torch.arange(args.n_tr, device=device)  # TR
    optimiser = torch.optim.Adam([pulse], lr=args.lr)
    history: list[float] = []
    for step in range(args.n_steps):
        optimiser.zero_grad(set_to_none=True)
        # Surrogate Fisher trace (matches CRLBMRFPulseDesignStrategy._toy_fisher_trace).
        alpha = pulse[..., 0]
        TR = pulse[..., 1].clamp_min(1e-3)
        T1 = theta_samples[..., 0].clamp_min(1e-3)
        T2 = theta_samples[..., 1].clamp_min(1e-3)
        signal = (
            alpha.sin().unsqueeze(0)
            * (1.0 - (-TR.unsqueeze(0) / T1.unsqueeze(-1)).exp())
            * (-TR.unsqueeze(0) / T2.unsqueeze(-1)).exp()
        )
        # –Fisher trace surrogate ⇒ minimisation grows information.
        fisher = signal.var(dim=-1).mean()
        loss = -fisher + args.lambda_beltrami * pulse.diff(dim=0).pow(2).mean()
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach().cpu().item()))
        if not args.quiet and (step + 1) % max(1, args.n_steps // 8) == 0:
            print(f"[design-mrf-sequence] step {step + 1}/{args.n_steps}: loss={loss.item():.4e}")
    out_payload = {
        "pulse_sequence": pulse.detach().cpu().tolist(),
        "objective_history": history,
        "target_prior": prior,
        "n_tr": args.n_tr,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        args.output.write_text(json.dumps(out_payload, indent=2))
    else:
        args.output.write_text(yaml.safe_dump(out_payload, sort_keys=False))
    if not args.quiet:
        print(f"[design-mrf-sequence] wrote {args.output}")
    return 0


def attach_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``design-mrf-sequence`` on a parent argparse subparser."""
    p = subparsers.add_parser(
        "design-mrf-sequence",
        help=(
            "Beltrami-CRLB-optimal MRF pulse-sequence design. "
            "Reads a target parameter "
            "prior and writes a designed pulse sequence."
        ),
    )
    p.add_argument(
        "--target-params",
        type=Path,
        required=True,
        help="Path to the YAML describing the parameter prior.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to write the designed sequence (YAML or JSON).",
    )
    p.add_argument("--n-tr", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda-beltrami", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=design_mrf_sequence_cmd)


__all__ = ["attach_subparsers", "design_mrf_sequence_cmd"]
