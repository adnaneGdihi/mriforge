"""Regression guard: ``spectramr.models.latent_diffusion.__all__`` must be importable.

Audit finding (WS-6): the package ``__all__`` listed eight names
(``LatentDiffusionTrainingStrategy``, ``LatentGANTrainingStrategy``,
``VAETrainingStrategy``, ``VQVAETrainingStrategy`` and four ``create_*``
helpers) whose import block was commented out. ``from
spectramr.models.latent_diffusion import *`` therefore raised ``AttributeError``
because those names were advertised in ``__all__`` but absent from the module
namespace. The real VAE/VQVAE strategies live in
``infrastructure/training/strategies/vae.py``; the commented ``latent_training_strategies``
module was dead/superseded.

The fix removed the eight orphaned names from ``__all__`` (and deleted the
commented import block). This test locks the contract: every name in
``__all__`` must resolve via ``getattr`` on the module so ``import *`` can
never raise ``AttributeError``.
"""

from __future__ import annotations

import importlib


def test_all_names_resolve_on_module() -> None:
    """Every name in ``__all__`` is a real attribute (import-star is safe)."""
    mod = importlib.import_module("spectramr.models.latent_diffusion")
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert not missing, (
        "names in __all__ but absent from the module namespace "
        f"(import * would raise AttributeError): {missing}"
    )


def test_import_star_does_not_raise() -> None:
    """Exercise the actual ``import *`` machinery against ``__all__``."""
    mod = importlib.import_module("spectramr.models.latent_diffusion")
    namespace: dict = {}
    # Emulate ``from spectramr.models.latent_diffusion import *``: this is exactly
    # what raised AttributeError before the fix.
    for name in mod.__all__:
        namespace[name] = getattr(mod, name)
    assert set(namespace) == set(mod.__all__)


def test_removed_strategy_names_are_not_advertised() -> None:
    """The eight orphaned strategy names must no longer appear in ``__all__``."""
    mod = importlib.import_module("spectramr.models.latent_diffusion")
    orphaned = {
        "LatentDiffusionTrainingStrategy",
        "LatentGANTrainingStrategy",
        "VAETrainingStrategy",
        "VQVAETrainingStrategy",
        "create_latent_diffusion_training_strategy",
        "create_latent_gan_training_strategy",
        "create_vae_training_strategy",
        "create_vqvae_training_strategy",
    }
    leaked = orphaned & set(mod.__all__)
    assert not leaked, f"orphaned strategy names still advertised: {leaked}"
