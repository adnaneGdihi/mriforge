"""Regression tests for F7/E27 (smoke_audit_20260521) — diffusion
generator wrappers must accept and forward the ``timesteps`` /
``t`` kwarg to their inner UNet.

Root cause: ``LaplaceDiffusionGenerator.forward(self, x)`` and
``RicianDiffusionGenerator.forward(self, x)`` both wrapped an
``EnhancedDeepDiffusionUNet`` whose forward signature is
``forward(self, x, timesteps, cond=None)``. Calling
``self.denoising_model(x)`` from the wrapper crashed with
``TypeError: EnhancedDeepDiffusionUNet.forward() missing 1 required
positional argument: 'timesteps'`` in 2 experiments
(experiment_73_laplace_diffusion +
experiment_74_rician_diffusion_for_magnitude_mri).

A third experiment (idea_4_strict_sar_bound) hit the same class of
bug with ``MRFTangentScore.forward(self, theta, t, ...)`` called
from ``mrf_acquisition_strategies._compute_losses_impl`` without
``t``.

Fix:
- Both generator wrappers' ``forward`` signatures now expose
  ``timesteps`` (None default → zeros vector) so the diffusion
  strategy's ``_callable_accepts_kwarg`` introspection sees it and
  threads through.
- ``MRFTangentScore.forward`` makes ``t`` optional with the same
  zero default for the acquisition-strategy path that has no
  diffusion-time context.

These tests:
1. Verify the wrappers accept ``timesteps`` without crashing.
2. Verify the inner UNet was called with the threaded ``timesteps``.
3. Verify the None-default fallback works.

Note: we use a mock for the inner UNet to avoid pulling the full
heavy Enhanced UNet stack in unit tests.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


# ── Static textual checks (cheap, no torch needed) ─────────────────────────


def test_laplace_generator_forward_has_timesteps_kwarg() -> None:
    """Pin the public ``forward`` signature — the diffusion strategy's
    ``_callable_accepts_kwarg`` only threads ``timesteps`` if the
    signature exposes it."""
    text = (
        REPO / "src" / "mriforge" / "models" / "generators"
        / "laplace_diffusion_generator.py"
    ).read_text()
    # The post-F7 signature spans a few lines; match relaxed.
    assert re.search(
        r"def forward\(\s*self,\s*x[^)]*timesteps[^)]*\)",
        text,
        re.DOTALL,
    ), (
        "LaplaceDiffusionGenerator.forward must accept ``timesteps`` "
        "or the diffusion strategy's introspection won't thread it "
        "through. See TODO/audit/smoke_audit_20260521.md §F7."
    )


def test_rician_generator_forward_has_timesteps_kwarg() -> None:
    text = (
        REPO / "src" / "mriforge" / "models" / "generators"
        / "rician_diffusion_generator.py"
    ).read_text()
    assert re.search(
        r"def forward\(\s*self,\s*x[^)]*timesteps[^)]*\)",
        text,
        re.DOTALL,
    ), "RicianDiffusionGenerator.forward must accept ``timesteps``."


def test_mrf_tangent_score_t_kwarg_is_optional() -> None:
    """The MRF acquisition strategy uses ``MRFTangentScore`` without
    threading ``t`` (no diffusion context). Making ``t`` optional
    with a zero default unblocks the strategy without changing the
    diffusion-time path's semantics."""
    text = (
        REPO / "src" / "mriforge" / "models" / "generators" / "mrf_models.py"
    ).read_text()
    # Match ``t: torch.Tensor | None = None`` in the MRFTangentScore class block.
    assert re.search(
        r"class MRFTangentScore[\s\S]*?def forward\([\s\S]*?t:\s*torch\.Tensor\s*\|\s*None\s*=\s*None",
        text,
    ), (
        "MRFTangentScore.forward must declare ``t`` as "
        "``Optional[torch.Tensor] = None`` so the acquisition strategy "
        "path can call it without a diffusion timestep."
    )


# ── Behavioral checks ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("mriforge.models.generators.laplace_diffusion_generator", "LaplaceDiffusionGenerator"),
        ("mriforge.models.generators.rician_diffusion_generator", "RicianDiffusionGenerator"),
    ],
)
def test_generator_forward_threads_timesteps_to_inner_unet(
    module_path: str, class_name: str
) -> None:
    """When the wrapper's forward is called with an explicit
    ``timesteps`` arg, the inner ``denoising_model`` must be called
    WITH that timesteps tensor (not dropped silently)."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    # Build with a mock denoising_model + mock diffusion to avoid the
    # heavy constructor work.
    mock_unet = MagicMock(name="inner_unet")
    mock_unet.return_value = torch.zeros(2, 1, 8, 8)

    # The wrappers' constructors take (denoising_model, diffusion).
    # We construct manually to avoid the heavy diffusion setup.
    inst = cls.__new__(cls)
    inst.denoising_model = mock_unet
    inst.diffusion = MagicMock()
    inst.diffusion.device = torch.device("cpu")
    # Initialize the nn.Module base
    torch.nn.Module.__init__(inst)
    # Re-add the mocked submodule after Module.__init__ resets state
    inst.denoising_model = mock_unet
    inst.diffusion = MagicMock(device=torch.device("cpu"))

    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0, 100], dtype=torch.long)

    # Call with explicit timesteps — must be threaded to inner UNet.
    inst.forward(x, timesteps=t)
    assert mock_unet.call_count == 1
    call_args, call_kwargs = mock_unet.call_args
    # Inner UNet was called with (x, t) positionally — the signature
    # of EnhancedDeepDiffusionUNet.forward is (x, timesteps, cond=None).
    assert len(call_args) >= 2, (
        f"{class_name} must call the inner denoising model with "
        f"(x, timesteps) — got positional args of len {len(call_args)}."
    )
    assert torch.equal(call_args[1], t), (
        f"{class_name} must pass the timesteps it received to the inner "
        f"denoising model unchanged. Got {call_args[1]}, expected {t}."
    )


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("mriforge.models.generators.laplace_diffusion_generator", "LaplaceDiffusionGenerator"),
        ("mriforge.models.generators.rician_diffusion_generator", "RicianDiffusionGenerator"),
    ],
)
def test_generator_forward_default_timesteps_is_zeros(
    module_path: str, class_name: str
) -> None:
    """When the wrapper's forward is called WITHOUT ``timesteps`` (the
    behavior the diffusion-strategy fallback path produces when the
    introspection can't find the kwarg), the inner UNet must still be
    called — with a zeros tensor as the default. This is the safety
    net that lets non-diffusion code paths use the generator."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    inst = cls.__new__(cls)
    torch.nn.Module.__init__(inst)
    inst.denoising_model = MagicMock(name="inner_unet", return_value=torch.zeros(2, 1, 8, 8))
    inst.diffusion = MagicMock(device=torch.device("cpu"))

    x = torch.randn(2, 1, 8, 8)
    inst.forward(x)  # no timesteps

    assert inst.denoising_model.call_count == 1
    args, _ = inst.denoising_model.call_args
    assert len(args) >= 2, "Default-timesteps path must still call inner UNet with (x, t)"
    # The default must be all-zeros and match batch size.
    assert torch.equal(args[1], torch.zeros(2, dtype=args[1].dtype, device=args[1].device))
