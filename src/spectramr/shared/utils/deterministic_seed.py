"""Deterministic seed harness for reproducible experiments."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Default seed constant
DEFAULT_SEED = 42


def set_global_seed(seed: int) -> None:
    """Set global seed for reproducibility across random, numpy, and torch.

    Args:
        seed: The seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeterministicSeedHarness:
    """Manages deterministic seeding based on configuration hashing."""

    def __init__(self, base_seed: int = DEFAULT_SEED):
        """Initialize the deterministic seed harness.

        Args:
            base_seed: Base seed to use when no config is provided

        """
        self.base_seed = base_seed
        self.logger = logging.getLogger(__name__)
        self.current_config_hash: str | None = None
        self.current_seed: int | None = None

    def hash_config(self, config: dict[str, Any]) -> str:
        """Generate a deterministic hash from configuration.

        Args:
            config: Configuration dictionary to hash

        Returns:
            SHA256 hash string of the configuration

        """
        # Sort keys for deterministic hashing
        sorted_config = self._sort_dict(config)

        # Convert to JSON string with sorted keys
        config_str = json.dumps(sorted_config, sort_keys=True, separators=(",", ":"))

        # Generate SHA256 hash
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

        return config_hash

    def _sort_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively sort dictionary keys for deterministic hashing."""
        if not isinstance(d, dict):
            return d

        sorted_dict = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                sorted_dict[key] = self._sort_dict(value)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    sorted_dict[key] = [self._sort_dict(item) for item in value]
                else:
                    sorted_dict[key] = value
            else:
                sorted_dict[key] = value

        return sorted_dict

    def config_to_seed(self, config: dict[str, Any]) -> int:
        """Convert configuration to a deterministic seed.

        Args:
            config: Configuration dictionary

        Returns:
            Deterministic seed based on config hash

        """
        config_hash = self.hash_config(config)

        # Convert first 8 characters of hash to integer
        # This gives us a 32-bit integer from the hash
        seed_hex = config_hash[:8]
        seed = int(seed_hex, 16)

        # Ensure seed is positive and within reasonable range
        seed = abs(seed) % (2**31 - 1)  # Max 32-bit signed int

        return seed

    def apply_deterministic_seed(self, config: dict[str, Any] | None = None) -> int:
        """Apply deterministic seeding based on configuration.

        Args:
            config: Configuration dictionary. If None, uses base_seed.

        Returns:
            The seed that was applied

        """
        if config is None:
            seed = self.base_seed
            self.logger.info(f"Using base seed: {seed}")
        else:
            seed = self.config_to_seed(config)
            config_hash = self.hash_config(config)
            hash_prefix = config_hash[:16]
            self.logger.info(f"Config hash: {hash_prefix}... -> Seed: {seed}")

        # Store current state
        self.current_seed = seed
        if config is not None:
            self.current_config_hash = self.hash_config(config)

        # Apply the seed
        set_global_seed(seed)

        return seed

    def get_current_seed(self) -> int | None:
        """Get the currently applied seed."""
        return self.current_seed

    def get_current_config_hash(self) -> str | None:
        """Get the hash of the currently applied configuration."""
        return self.current_config_hash

    def save_seed_info(self, output_path: str) -> None:
        """Save seed and configuration information to file.

        Args:
            output_path: Path to save the seed information

        """
        info = {
            "seed": self.current_seed,
            "config_hash": self.current_config_hash,
            "base_seed": self.base_seed,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(info, f, indent=2)

        self.logger.info(f"Seed info saved to: {output_path}")

    def load_seed_info(self, input_path: str) -> dict[str, Any]:
        """Load seed information from file.

        Args:
            input_path: Path to load seed information from

        Returns:
            Dictionary containing seed information

        """
        with open(input_path) as f:
            info = json.load(f)

        self.current_seed = info.get("seed")
        self.current_config_hash = info.get("config_hash")
        self.base_seed = info.get("base_seed", DEFAULT_SEED)

        self.logger.info(f"Seed info loaded from: {input_path}")
        return info

    def verify_reproducibility(self, config: dict[str, Any]) -> bool:
        """Verify that the current configuration produces the expected seed.

        Args:
            config: Configuration to verify

        Returns:
            True if configuration produces expected seed, False otherwise

        """
        expected_seed = self.config_to_seed(config)
        current_seed = self.get_current_seed()

        if current_seed is None:
            self.logger.warning("No seed currently applied")
            return False

        if expected_seed == current_seed:
            msg = "Reproducibility verified: config matches current seed"
            self.logger.info(msg)
            return True
        msg = f"Reproducibility check failed: expected {expected_seed}, "
        msg += f"got {current_seed}"
        self.logger.warning(msg)
        return False


def create_deterministic_harness(
    base_seed: int = DEFAULT_SEED,
) -> DeterministicSeedHarness:
    """Factory function to create a DeterministicSeedHarness instance."""
    return DeterministicSeedHarness(base_seed)
