"""Quickstart: reconstruct a tiny synthetic phantom with a registered U-Net.

This script demonstrates the *forward-pass* path through spectraMR without
training. It pulls a model class from the registry, instantiates it, and runs
a single inference. Designed to complete in under 30 seconds on CPU.

Run:

    python examples/quickstart_reconstruction.py

The clinical-use warning fires on first import; silence it in batch jobs with
``SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1``.
"""

from __future__ import annotations

import torch

import spectramr  # noqa: F401  (registers the clinical-use warning)
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY


def _synthetic_phantom(side: int = 64, n_channels: int = 2) -> torch.Tensor:
    """Build a tiny synthetic Shepp-Logan-like image; deterministic seed."""
    rng = torch.Generator().manual_seed(42)
    base = torch.randn(1, n_channels, side, side, generator=rng) * 0.05
    # Add a centred Gaussian bump as the "anatomy".
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, side),
        torch.linspace(-1, 1, side),
        indexing="ij",
    )
    bump = torch.exp(-(x ** 2 + y ** 2) / 0.1)
    base[:, 0] += bump
    if n_channels >= 2:
        base[:, 1] += 0.3 * bump
    return base


def main() -> None:
    # Load-bearing, and deliberately spelled out rather than hidden in a helper:
    # MODEL_REGISTRY is EMPTY on a plain ``import spectramr``. The decorators only
    # fire once this walks the curated package list, so a lookup before it
    # returns None for every name in the framework. It is idempotent.
    populate_model_registry()

    model_name = "toeplitz_attention_unet"
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise SystemExit(
            f"Model {model_name!r} is not registered "
            f"({len(MODEL_REGISTRY)} models were populated). "
            "Pick a name from MODEL_REGISTRY."
        )
    model_cls = entry["class"]
    print(f"Building {model_name} (mode={entry.get('mode', '?')})…")
    model = model_cls(in_channels=2, out_channels=2)
    model.eval()
    x = _synthetic_phantom(side=64, n_channels=2)
    print(f"Input  shape: {tuple(x.shape)}")
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    if y.shape != x.shape:
        raise SystemExit(f"Shape mismatch: {y.shape} != {x.shape}")
    print(f"Output stats: min={y.min().item():.4f}, max={y.max().item():.4f}, "
          f"mean={y.mean().item():.4f}")
    print("OK")


if __name__ == "__main__":
    main()
