"""FDB (Frequency-Decomposed Bridge) adapter — Karaoglu 2024.

Plan: ``TODO/backlog_baseline_replication_experiment_11.md`` Phase D.

Wraps the upstream FDB ``UNetModel`` and ``DiffusionBridge`` from
``external/baselines/fdb/utils/`` so they consume this repo's canonical
data layout (``[B, 2, H, W]`` complex, ``fft2c``-centred).

Upstream repo: https://github.com/icon-lab/FDB (vendored under
``external/baselines/fdb/``).

The bridge schedule has known numerical-stability issues at very low
acceleration (R ≤ 2) and very high (R ≥ 16); the Phase D plan calls
for a stability shim (D.2) which is not yet wired here — single-step
forward through the UNet is what this adapter currently exposes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch

from mriforge.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
)
from mriforge.models.registry import register_model

# parents[4], not [3]: this file is `src/mriforge/models/baselines/<x>.py`,
# so [3] is `src/` and [4] is the repo root. It was `src/models/baselines/`
# before the 2026-05 `src -> src/mriforge` refactor, when [3] WAS the root.
# The off-by-one made the vendored upstream unreachable and produced an
# error telling the user to run a vendoring command they had already run.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FDB_DIR = _REPO_ROOT / "external" / "baselines" / "fdb"
# Sentinel FILE, not the directory. `git submodule add` records a gitlink, so a
# checkout that never ran `git submodule update --init` leaves an EMPTY directory
# behind — `_FDB_DIR.exists()` is True there and the guard waves it through, so
# the failure surfaced as a bare `ModuleNotFoundError: utils.script_util_duo`
# with no mention of vendoring. cdiffmr.py already checks its own sentinel file
# (`_UPSTREAM_NETWORK_FILE`); this restores parity.
_FDB_SCRIPT_UTIL = _FDB_DIR / "utils" / "script_util_duo.py"


def _ensure_upstream_on_sys_path() -> None:
    """Add the FDB root to ``sys.path`` so ``from utils.<x>`` resolves.

    The FDB upstream is a script-style codebase: its ``utils/`` package
    is meant to be imported with the FDB root as the working directory.
    We add the root to ``sys.path`` (idempotent) so the regular Python
    import machinery resolves ``from utils.script_util_duo import ...``.
    """
    if not _FDB_SCRIPT_UTIL.exists():
        raise FileNotFoundError(
            f"FDB upstream not found at {_FDB_SCRIPT_UTIL}. "
            f"The directory {_FDB_DIR} "
            f"{'exists but is empty — the submodule was never initialised' if _FDB_DIR.exists() else 'does not exist'}. "
            "Run `git submodule update --init --recursive`, or the Phase A.1 "
            "vendoring command (git submodule add)."
        )
    dir_str = str(_FDB_DIR)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)


def _load_upstream_factory() -> Any:
    """Import + return the upstream ``create_model_and_diffusion`` factory."""
    _ensure_upstream_on_sys_path()
    mod = importlib.import_module("utils.script_util_duo")
    return mod.create_model_and_diffusion


def _default_model_kwargs(image_size: int) -> dict[str, Any]:
    """Defaults matching the upstream's ``model_and_diffusion_defaults()``.

    Kept here so the adapter has a stable construction surface even if
    upstream tweaks its defaults — Phase A.4 provenance records these
    so the regulatory bundle can prove what was used.
    """
    return {
        "image_size": image_size,
        "class_cond": False,
        "learn_sigma": False,
        "num_channels": 128,
        "num_res_blocks": 2,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "attention_resolutions": "16,8",
        "dropout": 0.0,
        "diffusion_steps": 1000,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "undersampling_rate": 4,
        "data_type": "singlecoil",
    }


def _complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B, C, H, W]`` to real ``[B, 2*C, H, W]`` (real|imag stacked)."""
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


def _real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real ``[B, 2*C, H, W]`` back to complex ``[B, C, H, W]``."""
    if torch.is_complex(x):
        return x
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)


@register_model(name="fdb_baseline", training_mode="cold_diffusion")
class FDBBaseline(BaselineAdapter):
    """FDB (Karaoglu 2024) — frequency-decomposed bridge baseline."""

    REPO_NAME = "fdb"
    PAPER_REF = "Karaoglu2024:FDB"
    PREFERRED_MASK_TYPE = "cartesian_peripheral"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.MULTI_COIL_KSPACE

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        image_size: int = 256,
        bridge_steps: int = 1000,
        undersampling_rate: int = 4,
        bridge_drift: float = 1.0,
        model_kwarg_overrides: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.bridge_steps = bridge_steps
        self.bridge_drift = bridge_drift

        model_kwargs = _default_model_kwargs(image_size)
        model_kwargs["diffusion_steps"] = bridge_steps
        model_kwargs["undersampling_rate"] = undersampling_rate
        if model_kwarg_overrides:
            model_kwargs.update(model_kwarg_overrides)
        self._model_kwargs = model_kwargs

        factory = _load_upstream_factory()
        model, diffusion = factory(**model_kwargs)
        # ``model`` is the UNet (nn.Module); ``diffusion`` is the
        # ``DiffusionBridge`` object that implements the sampling loop.
        self.upstream = model
        self._diffusion = diffusion

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Single denoise-step through the upstream UNet.

        Args:
            x: Input tensor. Accepts complex ``[B, C, H, W]`` (canonical)
                or real ``[B, 2*C, H, W]`` (upstream native).
            t: Diffusion timesteps as a 1-D long tensor ``[B]``. Defaults
                to zeros if omitted.

        Returns:
            Tensor in the same dtype family as the input.

        Note:
            The full FDB bridge sampling loop lives on
            ``self._diffusion.p_sample_loop_condition(...)`` and requires
            a trained checkpoint + a mask + an optional coil-map. Single
            denoise-step here is the smoke-test path; campaign runs
            invoke the diffusion loop directly.
        """
        input_was_complex = torch.is_complex(x)
        x_real = _complex_to_real(x)
        if t is None:
            t = torch.zeros(x_real.shape[0], dtype=torch.long, device=x_real.device)
        out_real = self.upstream(x_real, t)
        if input_was_complex:
            return _real_to_complex(out_real)
        return out_real

    def sample(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        coil_map: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Run the FDB bridge sampling loop end-to-end.

        Delegates to ``self._diffusion.p_sample_loop_condition`` (the
        upstream ``DiffusionBridge`` method) so a campaign run can
        invoke FDB just like P-CD — :meth:`forward` is the per-step
        denoise primitive; this method is the full sampler.

        Args:
            kspace: Undersampled k-space, shape
                ``[batch_size, 2, H, W]`` (real/imag stacked).
            mask: Undersampling mask, shape
                ``[batch_size, 2, H, W]`` (paper convention: replicated
                across real/imag).
            coil_map: Coil sensitivity maps (multi-coil only). ``None``
                for single-coil.
            batch_size: Batch size (used to construct the sample shape).

        Returns:
            The final sampled tensor (last element of the progressive
            iterator). Without trained weights this is noise — the
            scientific use case requires
            ``self.upstream.load_state_dict(...)`` first.

        Note:
            The full sampling loop is slow (``bridge_steps`` UNet
            forwards). Test code should pass a small ``image_size`` and
            a low ``bridge_steps`` (e.g. 5) to keep wall-clock manageable.
        """
        shape = (batch_size, 2, self.image_size, self.image_size)
        return self._diffusion.p_sample_loop_condition(
            self.upstream,
            shape,
            kspace,
            mask,
            coil_map,
        )[-1]


__all__ = ["FDBBaseline"]
