"""Ablation Config Generator — Parameter sweep config generation.

Generates experiment YAML configs from a base config + sweep axes.
Supports grid search and random sampling.
"""

from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path
from typing import Any, Literal

from spectramr.config.io import read_yaml_dict, write_yaml_dict
from spectramr.config.schemas.campaign import AblationAxisSchema

logger = logging.getLogger(__name__)

__all__ = ["AblationConfigGenerator"]


class AblationConfigGenerator:
    """Generate YAML configs from a base config + parameter sweep axes.

    For grid mode with 2 axes of sizes [4, 3] → 12 configs.
    For random mode, samples N configs from the joint space.
    """

    @staticmethod
    def generate(
        base_config_path: str,
        axes: list[AblationAxisSchema],
        output_dir: str,
        mode: Literal["grid", "random"] = "grid",
        n_random_samples: int = 10,
        seed: int = 42,
    ) -> list[str]:
        """Generate ablation config files.

        Args:
            base_config_path: Path to the base experiment YAML.
            axes: List of AblationAxisSchema defining the sweep.
            output_dir: Directory to write generated configs.
            mode: 'grid' for full factorial, 'random' for random sampling.
            n_random_samples: Number of samples for random mode.
            seed: Random seed for random mode.

        Returns:
            List of paths to generated config files.

        Raises:
            FileNotFoundError: If base config does not exist.
            ValueError: If any axis config_path is invalid.
        """
        base_path = Path(base_config_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")

        # Route the raw load through the config-layer helper so the
        # SSOT pre-commit guard (tests/unit/test_no_yaml_reparse.py)
        # stays clean — see TODO/backlog_ssot_and_layering_cleanup.md.
        base_config = read_yaml_dict(base_path)

        # Validate all axis paths exist in base config
        for axis in axes:
            _get_nested(base_config, axis.config_path)

        # Generate parameter combinations
        if mode == "grid":
            combinations = list(itertools.product(*[a.values for a in axes]))
        elif mode == "random":
            import numpy as np

            rng = np.random.default_rng(seed)
            all_combos = list(itertools.product(*[a.values for a in axes]))
            n = min(n_random_samples, len(all_combos))
            indices = rng.choice(len(all_combos), size=n, replace=False)
            combinations = [all_combos[i] for i in sorted(indices)]
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'grid' or 'random'.")

        # Create output directory
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_paths: list[str] = []
        base_name = base_path.stem

        for combo_idx, combo in enumerate(combinations):
            # Deep copy the base config
            config = copy.deepcopy(base_config)

            # Apply each axis value
            suffix_parts = []
            for axis, value in zip(axes, combo, strict=True):
                _set_nested(config, axis.config_path, value)
                # Build a readable suffix
                label = axis.label or axis.config_path.rsplit(".", maxsplit=1)[-1]
                suffix_parts.append(f"{label}={value}")

            # Update metadata name
            ablation_suffix = "_".join(s.replace(".", "_").replace("=", "_") for s in suffix_parts)
            ablation_name = f"{base_name}_ablation_{ablation_suffix}"

            if "metadata" in config:
                config["metadata"]["name"] = ablation_name
                config["metadata"]["description"] = f"Ablation: {', '.join(suffix_parts)}"
                # Tags can be a list of strings or a dict — handle both
                tags = config["metadata"].get("tags")
                if isinstance(tags, list):
                    if "ablation" not in tags:
                        tags.append("ablation")
                elif isinstance(tags, dict):
                    tags["ablation"] = "true"
                    tags["ablation_index"] = str(combo_idx)
                else:
                    config["metadata"]["tags"] = {
                        "ablation": "true",
                        "ablation_index": str(combo_idx),
                    }

            # Update output directory if present
            if "training" in config and "output_dir" in config["training"]:
                config["training"]["output_dir"] = str(out_dir / "results" / ablation_name)

            # Write config (via the config-layer helper for SSOT hygiene).
            config_path = out_dir / f"{ablation_name}.yaml"
            write_yaml_dict(config, config_path)

            generated_paths.append(str(config_path))
            logger.debug(f"Generated ablation config: {config_path}")

        logger.info(
            f"Generated {len(generated_paths)} ablation configs "
            f"({mode} mode, {len(axes)} axes) in {out_dir}"
        )

        return generated_paths

    @staticmethod
    def summarise_sweep(
        axes: list[AblationAxisSchema],
    ) -> str:
        """Return a human-readable summary of the sweep space.

        Args:
            axes: List of ablation axes.

        Returns:
            Formatted string describing the sweep.
        """
        lines = ["Ablation Sweep Summary", "=" * 40]
        total = 1
        for axis in axes:
            label = axis.label or axis.config_path
            n = len(axis.values)
            total *= n
            lines.append(f"  {label}: {axis.values} ({n} levels)")
        lines.append(f"  Total combinations (grid): {total}")
        return "\n".join(lines)


# ── Internal helpers ─────────────────────────────────────────────────


def _get_nested(config: dict, dotted_path: str) -> Any:
    """Traverse a nested dict using a dotted path.

    Args:
        config: Nested dictionary.
        dotted_path: Dot-separated key path (e.g., 'model.model_kwargs.features').

    Returns:
        Value at the path.

    Raises:
        KeyError: If any segment is missing.
    """
    keys = dotted_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(
                f"Config path '{dotted_path}' not found: "
                f"key '{key}' missing at level {keys.index(key)}"
            )
        current = current[key]
    return current


def _set_nested(config: dict, dotted_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted path.

    Args:
        config: Nested dictionary (mutated in place).
        dotted_path: Dot-separated key path.
        value: Value to set.

    Raises:
        KeyError: If parent keys are missing.
    """
    keys = dotted_path.split(".")
    current = config
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(
                f"Config path '{dotted_path}' not found: key '{key}' missing in parent dict"
            )
        current = current[key]
    current[keys[-1]] = value
