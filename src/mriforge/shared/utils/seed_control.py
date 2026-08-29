"""Seed Control Utility

Central seeding utility for reproducible experiments.
Ensures consistent seeding across PyTorch, NumPy, random,
dataset workers, and diffusion noise schedules.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SeedConfig:
    """Configuration for seeding behavior."""

    base_seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    worker_init_fn: Callable | None = None


class SeedController:
    """Central controller for all random seeding in the project.

    Ensures reproducible results across:
    - PyTorch operations
    - NumPy operations
    - Python random module
    - CUDA operations (if available)
    - Dataset workers
    - Diffusion noise schedules
    """

    def __init__(self, config: SeedConfig | None = None):
        """__init__.

        Args:
            config (Optional[SeedConfig]): Description.
        """
        self.config = config or SeedConfig()
        self._worker_seeds: dict[int, int] = {}
        self._diffusion_seeds: dict[str, int] = {}

    def set_global_seed(self, seed: int | None = None) -> int:
        """Set global seed for all random number generators.

        Args:
            seed: Seed value. If None, uses config base_seed.

        Returns:
            The seed value used.

        """
        seed = seed if seed is not None else self.config.base_seed

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

        # CUDA (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior
        if self.config.deterministic:
            self._set_deterministic_mode()

        return seed

    def _set_deterministic_mode(self):
        """Configure PyTorch for deterministic operations."""
        # Disable cuDNN benchmark for reproducibility
        torch.backends.cudnn.benchmark = self.config.benchmark
        torch.backends.cudnn.deterministic = True

        # Set environment variables for additional determinism
        os.environ["PYTHONHASHSEED"] = str(self.config.base_seed)

    def get_worker_seed(self, worker_id: int) -> int:
        """Get seed for a specific dataset worker.

        Args:
            worker_id: Worker ID from DataLoader

        Returns:
            Seed for this worker

        """
        if worker_id not in self._worker_seeds:
            # Generate worker-specific seed
            self._worker_seeds[worker_id] = self.config.base_seed + worker_id + 1

        return self._worker_seeds[worker_id]

    def init_worker_seed(self, worker_id: int):
        """Initialize random state for a dataset worker.

        Args:
            worker_id: Worker ID from DataLoader

        """
        seed = self.get_worker_seed(worker_id)

        # Set worker-specific seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            # ``manual_seed_all`` covers every visible CUDA device. Without
            # it, multi-GPU runs leave non-default devices on the previous
            # global seed. See findings booklet 2026-05-05 SH-4.
            torch.cuda.manual_seed_all(seed)

    def get_diffusion_seed(self, key: str, step: int | None = None) -> int:
        """Get seed for diffusion noise schedule.

        Args:
            key: Identifier for the diffusion process (e.g., 'forward', 'reverse')
            step: Optional step number for time-dependent seeding

        Returns:
            Seed for diffusion noise generation

        """
        if key not in self._diffusion_seeds:
            # ``hash()`` of strings is salted per Python process by default,
            # so the derived seed is non-reproducible across runs unless
            # PYTHONHASHSEED is fixed. Use a stable hash instead.
            # See findings booklet 2026-05-05 SH-1.
            import hashlib

            digest = hashlib.blake2b(
                f"{self.config.base_seed}_{key}".encode(), digest_size=8
            ).digest()
            self._diffusion_seeds[key] = int.from_bytes(digest, "big") % (2**32)

        base_diffusion_seed = self._diffusion_seeds[key]

        if step is not None:
            # Make seed time-dependent for diffusion steps
            return (base_diffusion_seed + step) % (2**32)

        return base_diffusion_seed

    def set_diffusion_noise(
        self,
        noise: torch.Tensor,
        key: str,
        step: int | None = None,
    ):
        """Set reproducible noise for diffusion processes.

        Args:
            noise: Noise tensor to fill
            key: Identifier for the diffusion process
            step: Optional step number

        """
        seed = self.get_diffusion_seed(key, step)

        # Create generator with specific seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Fill noise tensor with reproducible values
        noise.normal_(generator=generator)

    def create_worker_init_fn(self) -> Callable:
        """Create worker initialization function for DataLoader.

        Returns:
            Function suitable for DataLoader worker_init_fn parameter

        """

        def worker_init(worker_id: int):
            """worker_init.

            Args:
                worker_id (int): Description.
            Returns:
                Any: Description.
            """
            self.init_worker_seed(worker_id)

        return worker_init

    def fork_seed(self, fork_id: str | int) -> int:
        """Generate a derived seed for forked processes or parallel operations.

        Args:
            fork_id: Identifier for the fork

        Returns:
            Derived seed

        """
        # Stable hash so fork seeds reproduce across runs.
        # See findings booklet 2026-05-05 SH-1.
        import hashlib

        digest = hashlib.blake2b(
            f"{self.config.base_seed}_fork_{fork_id}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") % (2**32)

    def reset(self):
        """Reset all stored seeds and state."""
        self._worker_seeds.clear()
        self._diffusion_seeds.clear()

    def get_state(self) -> dict[str, Any]:
        """Get current seeding state for serialization.

        Returns:
            Dictionary containing seeding state

        """
        return {
            "config": self.config,
            "worker_seeds": self._worker_seeds.copy(),
            "diffusion_seeds": self._diffusion_seeds.copy(),
        }

    def set_state(self, state: dict[str, Any]):
        """Restore seeding state from dictionary.

        Args:
            state: State dictionary from get_state()

        """
        self.config = state.get("config", SeedConfig())
        self._worker_seeds = state.get("worker_seeds", {})
        self._diffusion_seeds = state.get("diffusion_seeds", {})


# Global instance for convenience
_global_seed_controller: SeedController | None = None


def get_seed_controller() -> SeedController:
    """Get the global seed controller instance."""
    global _global_seed_controller
    if _global_seed_controller is None:
        _global_seed_controller = SeedController()
    return _global_seed_controller


def set_global_seed(seed: int, deterministic: bool = True, rank: int = 0) -> int:
    """Convenience function to set the global seed (SSOT-backed).

    Delegates the actual RNG + determinism policy to the accelerator SSOT
    ``mriforge.accelerator.seed_everything`` so the training pipeline's seeding is
    IDENTICAL to ``initialize_accelerator``'s. The two previously diverged: this
    facade omitted ``torch.use_deterministic_algorithms``, so a pipeline seeded
    only through here silently ran with weaker determinism than the CLI entry
    point (2026-07 determinism-SSOT unification). The singleton controller's
    config is kept in sync for its worker/diffusion helper methods.

    ``rank`` was the SECOND parameter to go missing the same way (issue #1124).
    ``seed_everything`` has always computed ``seed + rank``, and
    ``initialize_accelerator`` has always forwarded it -- but this facade, the only
    seeding path ``pipelines/train.py`` uses, did not declare the parameter at all.
    Every rank of a DDP/FSDP/DeepSpeed run therefore seeded identically, so random
    augmentations and patch sampling were duplicated across the N processes.
    (Data *sharding* was unaffected: ``DistributedSampler`` partitions by rank
    without consulting the RNG.) A facade that restates a subset of the SSOT's
    signature keeps re-acquiring this class of gap; forward what you are given.

    ``controller.config.base_seed`` is deliberately set to the UNOFFSET seed. Its
    ``get_diffusion_seed`` derives blake2b-stable seeds from it so reproducible
    validation noise is identical on every rank -- a separate axis from the
    per-rank augmentation stream, and offsetting it would silently make each rank
    denoise from different noise. Do not "fix" this to ``seed + rank``.

    Args:
        seed: Seed value.
        deterministic: Whether to enable deterministic mode.
        rank: Distributed rank whose data stream this process drives; the RNG is
            seeded with ``seed + rank``. Leave at 0 for single-process runs.
            Callers MUST NOT pass a non-zero rank before model construction --
            see the ordering note in ``accelerator.initialize_accelerator``.

    Returns:
        The (rank-offset) seed value actually used.
    """
    # Lazy import: keep this module cheap to import and avoid any import cycle.
    from mriforge.accelerator import seed_everything

    controller = get_seed_controller()
    controller.config.base_seed = seed
    controller.config.deterministic = deterministic
    return seed_everything(seed, rank=rank, deterministic=deterministic)


def init_worker_seed(worker_id: int):
    """Convenience function to initialize worker seed.

    Args:
        worker_id: Worker ID

    """
    controller = get_seed_controller()
    controller.init_worker_seed(worker_id)


def get_diffusion_seed(key: str, step: int | None = None) -> int:
    """Convenience function to get diffusion seed.

    Args:
        key: Diffusion process identifier
        step: Optional step number

    Returns:
        Seed value

    """
    controller = get_seed_controller()
    return controller.get_diffusion_seed(key, step)


def set_diffusion_noise(noise: torch.Tensor, key: str, step: int | None = None):
    """Convenience function to set diffusion noise.

    Args:
        noise: Noise tensor
        key: Diffusion process identifier
        step: Optional step number

    """
    controller = get_seed_controller()
    controller.set_diffusion_noise(noise, key, step)
