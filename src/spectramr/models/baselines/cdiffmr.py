"""CDiffMR adapter — Huang et al. 2023 baseline.

Plan: ``TODO/backlog_baseline_replication_experiment_11.md`` Phase B.

Wraps the upstream CDiffMR UNet from
``external/baselines/cdiffmr/models/network/cdiffmr/network_cdiff_unet2.py``
so it consumes this repo's canonical data layout (``[B, 2, H, W]`` complex,
``fft2c``-centred) without re-implementing the inner architecture.

Upstream repo: https://github.com/ayanglab/CDiffMR (vendored under
``external/baselines/cdiffmr/`` as a git submodule).

The vendored upstream is a script-style codebase (no ``__init__.py``),
so the UNet is loaded via :mod:`importlib.util` rather than a plain
``import`` — this avoids polluting ``sys.path`` with the upstream's
top-level modules (``models``, ``utils``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
)
from spectramr.models.registry import register_model

# parents[4], not [3]: this file is `src/spectramr/models/baselines/<x>.py`,
# so [3] is `src/` and [4] is the repo root. It was `src/models/baselines/`
# before the 2026-05 `src -> src/spectramr` refactor, when [3] WAS the root.
# The off-by-one made the vendored upstream unreachable and produced an
# error telling the user to run a vendoring command they had already run.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CDIFFMR_DIR = _REPO_ROOT / "external" / "baselines" / "cdiffmr"
_UPSTREAM_NETWORK_FILE = _CDIFFMR_DIR / "models" / "network" / "cdiffmr" / "network_cdiff_unet2.py"


def _load_upstream_model_class() -> type[torch.nn.Module]:
    """Load the upstream ``Model`` UNet class without polluting ``sys.path``.

    Uses :func:`importlib.util.spec_from_file_location` to import the
    single file by path. The class is cached on the module after the
    first call so the import cost is paid once.
    """
    if "_cdiffmr_upstream_network" in sys.modules:
        return sys.modules["_cdiffmr_upstream_network"].Model  # type: ignore[attr-defined]
    if not _UPSTREAM_NETWORK_FILE.exists():
        raise FileNotFoundError(
            f"CDiffMR upstream not found at {_UPSTREAM_NETWORK_FILE}. "
            "Run the Phase A.1 vendoring command (git submodule add)."
        )
    spec = importlib.util.spec_from_file_location(
        "_cdiffmr_upstream_network", _UPSTREAM_NETWORK_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build a spec for {_UPSTREAM_NETWORK_FILE}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cdiffmr_upstream_network"] = module
    spec.loader.exec_module(module)
    return module.Model


def _default_opt(resolution: int, in_channels: int, out_channels: int) -> dict[str, Any]:
    """Sensible defaults matching the paper's small-net experiments.

    The upstream ``Model`` is opinionated about its ``opt`` dict; defaults
    here mirror the upstream's ``options/CDiffMR/*.json`` for ``ksu_m04``.
    """
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "resolution": resolution,
        "emb_channels": 64,
        "emb_channels_multi": [1, 2, 2, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [resolution // 4],
        "dropout": 0.0,
        "resamp_with_conv": True,
    }


def _complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B, C, H, W]`` to real ``[B, 2*C, H, W]`` (real|imag stacked)."""
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


def _real_to_complex(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Convert real ``[B, 2*C, H, W]`` back to complex ``[B, C, H, W]``."""
    if torch.is_complex(x):
        return x
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)


@register_model(name="cdiffmr_baseline", training_mode="cold_diffusion")
class CDiffMRBaseline(BaselineAdapter):
    """CDiffMR (Huang 2023) — cascaded diffusion MRI baseline."""

    REPO_NAME = "cdiffmr"
    PAPER_REF = "Huang2023:CDiffMR"
    PREFERRED_MASK_TYPE = "gaussian_density"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        resolution: int = 256,
        num_steps: int = 50,
        opt_overrides: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.num_steps = num_steps

        # The upstream ``Model`` UNet expects an ``opt`` dict with a small
        # set of keys. We coerce our typed kwargs into that shape and let
        # the caller override individual fields via ``opt_overrides``.
        opt = _default_opt(resolution, in_channels, out_channels)
        if opt_overrides:
            opt.update(opt_overrides)
        self._opt = opt
        ModelCls = _load_upstream_model_class()
        self.upstream = ModelCls(opt)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Single denoise-step through the upstream UNet.

        Args:
            x: Image tensor. Accepts either complex ``[B, C, H, W]`` (canonical
                MRI layout) or real ``[B, 2*C, H, W]`` (upstream native layout).
            t: Diffusion timesteps as a 1-D long tensor ``[B]``. Defaults to
                zeros if omitted.

        Returns:
            Reconstructed tensor in the same dtype family as the input
            (complex in → complex out, real in → real out).

        Note:
            This implements one denoise step, not the full reverse-diffusion
            sampling loop. The full loop requires a trained checkpoint and
            the upstream ``CDiffMR`` trainer-class wrapper; see
            ``external/baselines/cdiffmr/main_test_cdfiimr_ksu_FastMRI_complex.py``
            for that path.
        """
        input_was_complex = torch.is_complex(x)
        x_real = _complex_to_real(x)
        if t is None:
            t = torch.zeros(x_real.shape[0], dtype=torch.long, device=x_real.device)
        out_real = self.upstream(x_real, t)
        if input_was_complex:
            return _real_to_complex(out_real, self.out_channels)
        return out_real

    def sample(
        self,
        x_start: torch.Tensor,
        x_obs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        timesteps: int | None = None,
    ) -> torch.Tensor:
        """Full CDiffMR reverse-diffusion sampling loop.

        The upstream implementation of the full loop is wrapped inside
        the ``CDiffMR`` ``ModelBase`` subclass in
        ``external/baselines/cdiffmr/models/model/cdiffmr/cdm_ksu_m05.py``,
        which requires a ``CharbonnierLoss``, an ``SSIMLoss``, a
        ``denoise_fn`` factory, ``ksu_masks`` precomputed at each
        timestep, wandb wiring, and an ``opt`` dict with many keys.
        Reproducing the full setup without a paper checkpoint is brittle
        and not load-bearing for this adapter's role (which is to plug
        into the repo's Builder + Strategy chain).

        Raises:
            NotImplementedError: with pointers to the upstream files
                that implement the full loop. To enable this method,
                load a trained CDiffMR checkpoint into ``self.upstream``
                and instantiate the upstream ``cdm_ksu_m05.CDM_KSU``
                wrapper class with the published config dict.
        """
        del x_start, x_obs, mask, batch_size, timesteps  # noqa: not yet wired
        raise NotImplementedError(
            "CDiffMR full sampling-loop wiring is deferred. The upstream "
            "wrapper at external/baselines/cdiffmr/models/model/cdiffmr/cdm_ksu_m05.py "
            "requires the full opt-dict setup + a trained checkpoint. "
            "This is deferred until paper-published weights become available."
        )


__all__ = ["CDiffMRBaseline"]
