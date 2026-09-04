"""Uniform forward interface for MRIxFields2026 baseline generators (Task 4).

Loads a pretrained ResNet (CUT / CycleGAN) or StarGAN v2 checkpoint and returns
a ``LoadedBaseline`` whose ``.forward`` callable hides all generator-specific
details (style code sampling for StarGAN, model instantiation for ResNet).

Usage::

    loaded = load_baseline_generator("cut", ckpt_path, "cpu")
    out = loaded.forward(torch.randn(1, 1, 256, 256))   # [B,1,H,W] → [B,1,H,W]

    loaded = load_baseline_generator(
        "stargan_v2", ckpt_path, "cuda", seed=0, target_domain=4
    )
    out = loaded.forward(img_slice)   # style code pre-sampled and bound

Fail-loud obligations:
    - ``target_domain=None`` for ``stargan_v2`` → raises ``ValueError``.
    - Unknown ``method`` → raises ``ValueError``.
    - Checkpoint layout not recognised by ``extract_generator_state`` → propagated
      ``ValueError`` from that function.

StarGAN style determinism:
    A local ``torch.Generator`` seeded with ``seed`` is used for ``z`` sampling;
    the global RNG state is never touched.

Dimension defaults (real checkpoints):
    - ResNet: ``input_nc=1, output_nc=1, ngf=64, n_blocks=9``
    - StarGAN: ``img_size=512, style_dim=64, max_conv_dim=512, input_nc=1``
      + ``latent_dim=16, num_domains=15`` (fixed).

Optional ``resnet_kwargs`` / ``stargan_kwargs`` args override these defaults for
unit tests that need smaller networks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

import spectramr.models.generators.cycle_gan  # noqa: F401 — side-effect: registers cyclegan_generator
from spectramr.models.registry import get_model_class

from .original_arch import (
    ClovaaiMappingNetwork,
    ClovaaiStarGANv2Generator,
    extract_generator_state,
)

# ---------------------------------------------------------------------------
# Real checkpoint dimensions (authoritative — do not change without updating the
# cluster-run command in docs/mrixfields2026_baseline_evaluation.rst).
# ---------------------------------------------------------------------------

_RESNET: dict[str, object] = {"input_nc": 1, "output_nc": 1, "ngf": 64, "n_blocks": 9}
_STARGAN: dict[str, object] = {"img_size": 512, "style_dim": 64, "max_conv_dim": 512, "input_nc": 1}
_LATENT_DIM: int = 16
_NUM_DOMAINS: int = 15


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoadedBaseline:
    """A loaded baseline generator with a uniform forward interface.

    Attributes:
        forward: Callable ``[B,1,H,W] → [B,1,H,W]`` in model output space
            (values in ``[-1, 1]`` before the ``clip * 0.5 + 0.5`` normalisation
            that ``predict_volume`` applies).  For StarGAN the style code is
            pre-sampled and bound — callers need not handle it.
        model_type: ``"resnet"`` or ``"stargan_v2"``.
        crop_size: ``(H, W)`` that the model expects (StarGAN), or ``None``
            (ResNet accepts any size).
        meta: Provenance dict stamped with dims, seed, checkpoint path, and
            (for StarGAN) ``target_domain``.
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    model_type: str
    crop_size: tuple[int, int] | None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_baseline_generator(
    method: str,
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    seed: int = 0,
    target_domain: int | None = None,
    resnet_kwargs: dict | None = None,
    stargan_kwargs: dict | None = None,
) -> LoadedBaseline:
    """Load a pretrained MRIxFields2026 baseline generator.

    Parameters:
        method: ``"cut"``, ``"cyclegan"``, or ``"stargan_v2"``.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: PyTorch device string or object.
        seed: RNG seed for the StarGAN style-code latent ``z`` (local generator,
            does not affect global RNG).
        target_domain: Required for ``stargan_v2``; the joint domain index
            ``[0, 15)`` for the target ``(contrast, field)`` pair.
        resnet_kwargs: Optional overrides for ``_RESNET`` dims (test use only).
        stargan_kwargs: Optional overrides for ``_STARGAN`` dims (test use only).

    Returns:
        ``LoadedBaseline`` with all model state loaded and (for StarGAN) the
        style code pre-computed.

    Raises:
        ValueError: If ``method`` is unknown, ``target_domain`` is ``None`` for
            ``stargan_v2``, or the checkpoint layout is not recognised.
    """
    device = torch.device(device)
    ckpt = Path(checkpoint_path)
    state = torch.load(ckpt, map_location=device, weights_only=True)

    # -- ResNet path (CUT and CycleGAN share the same generator architecture) --
    if method in ("cut", "cyclegan"):
        kw = {**_RESNET, **(resnet_kwargs or {})}
        gen_cls = get_model_class("cyclegan_generator")
        model = gen_cls(**kw).to(device).eval()
        model.load_state_dict(extract_generator_state(state), strict=True)

        def fwd(x: torch.Tensor, _m: torch.nn.Module = model) -> torch.Tensor:
            return _m(x)

        return LoadedBaseline(
            fwd,
            "resnet",
            None,
            {"method": method, **kw, "ckpt": str(ckpt)},
        )

    # -- StarGAN v2 path --
    if method == "stargan_v2":
        if target_domain is None:
            raise ValueError(
                "stargan_v2 requires target_domain (the joint domain index [0, 15)); "
                "got target_domain=None"
            )
        kw = {**_STARGAN, **(stargan_kwargs or {})}
        gen = ClovaaiStarGANv2Generator(**kw).to(device).eval()
        mnet = ClovaaiMappingNetwork(_LATENT_DIM, kw["style_dim"], _NUM_DOMAINS).to(device).eval()
        gen.load_state_dict(state["nets_ema"]["generator"], strict=True)
        mnet.load_state_dict(state["nets_ema"]["mapping_network"], strict=True)

        # Sample style code with a local generator (never touches global RNG).
        g = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(1, _LATENT_DIM, generator=g, device=device)
        with torch.no_grad():
            s = mnet(z, torch.tensor([int(target_domain)], device=device))

        def fwd(
            x: torch.Tensor,
            _g: torch.nn.Module = gen,
            _s: torch.Tensor = s,
        ) -> torch.Tensor:
            return _g(x, _s.expand(x.size(0), -1))

        img_size = int(kw["img_size"])
        cs: tuple[int, int] = (img_size, img_size)
        return LoadedBaseline(
            fwd,
            "stargan_v2",
            cs,
            {
                "method": method,
                "target_domain": int(target_domain),
                "seed": int(seed),
                **kw,
                "ckpt": str(ckpt),
            },
        )

    raise ValueError(f"unknown method {method!r}; expected one of 'cut', 'cyclegan', 'stargan_v2'")
