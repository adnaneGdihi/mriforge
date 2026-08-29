"""Quickstart: forward-noise one synthetic step with a Lévy diffusion process.

This script exercises the diffusion-primitives layer
(``mriforge.models.diffusion``) without training. It runs a single
forward-noising step from an α-stable Lévy sampler, demonstrating that the
heavy-tailed perturbation kicks the image distribution off the Gaussian
manifold.

Designed to complete in under 30 seconds on CPU.

Run:

    python examples/quickstart_diffusion.py
"""

from __future__ import annotations

import torch

import mriforge  # noqa: F401


def main() -> None:
    try:
        from mriforge.models.diffusion.alpha_stable_levy import sample_alpha_stable
    except ImportError as exc:  # pragma: no cover — gate
        raise SystemExit(
            "alpha_stable_levy diffusion primitive not available; "
            f"is mriforge[diffusion] installed? ({exc})"
        )

    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32)
    # alpha=1.5 → heavy-tailed Lévy stable noise, between Cauchy (1) and Gauss (2).
    noise = sample_alpha_stable(shape=tuple(x.shape), alpha=1.5)
    sigma = 0.3
    x_noised = x + sigma * noise
    excess_kurtosis = (
        ((noise - noise.mean()) / noise.std()).pow(4).mean().item() - 3.0
    )
    print(f"alpha-stable noise: shape={tuple(noise.shape)}, "
          f"min={noise.min().item():.3f}, max={noise.max().item():.3f}, "
          f"excess_kurtosis={excess_kurtosis:.2f}")
    print(f"signal  shape: {tuple(x.shape)}, std={x.std().item():.3f}")
    print(f"noised  shape: {tuple(x_noised.shape)}, std={x_noised.std().item():.3f}")
    # Heavy tails => excess kurtosis well above zero (Gaussian baseline).
    if excess_kurtosis < 1.0:
        print("WARN: excess kurtosis lower than expected for α=1.5; "
              "is the sampler in the Gaussian regime?")
    print("OK")


if __name__ == "__main__":
    main()
