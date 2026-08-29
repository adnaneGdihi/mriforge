"""Automatic preset discovery and registration."""

import logging
from pathlib import Path

from .preset import ConfigPreset
from .registry import PresetRegistry

logger = logging.getLogger(__name__)


def discover_and_register_presets(
    experiments_dir: str | Path = "experiments/inprogress",
) -> int:
    """Discover and register presets from experiment configs.

    Scans experiment directory for YAML files and registers them as presets
    based on naming conventions and metadata.

    Categories are inferred from filename:
    - experiment_*_gan.yaml → "gan"
    - experiment_*_diffusion.yaml → "diffusion"
    - experiment_*_reconstruction.yaml → "reconstruction"
    - experiment_*.yaml (no match) → "other"

    Args:
        experiments_dir: Directory containing experiment configs

    Returns:
        Number of presets registered
    """
    experiments_dir = Path(experiments_dir)

    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found: {experiments_dir}")
        return 0

    registered_count = 0

    # Scan for YAML files
    for yaml_file in sorted(experiments_dir.glob("experiment_*.yaml")):
        try:
            preset_name = yaml_file.stem  # e.g., "experiment_01_baseline_gan"

            # Infer category from filename
            filename_lower = yaml_file.name.lower()
            if "gan" in filename_lower:
                category = "gan"
            elif "diffusion" in filename_lower:
                category = "diffusion"
            elif "reconstruction" in filename_lower:
                category = "reconstruction"
            elif "vae" in filename_lower or "ae" in filename_lower:
                category = "vae"
            elif "ssl" in filename_lower or "contrastive" in filename_lower:
                category = "ssl"
            else:
                category = "other"

            # Create and register preset
            preset = ConfigPreset(
                name=preset_name,
                category=category,
                description=f"Configuration from {yaml_file.name}",
                yaml_path=str(yaml_file),
                version="6.0",
                tags=_infer_tags(yaml_file.name, category),
            )

            PresetRegistry.register(preset)
            registered_count += 1
            logger.debug(f"Registered preset: {preset_name} (category={category})")

        except Exception as e:
            logger.warning(f"Failed to register preset from {yaml_file.name}: {e}")

    logger.info(f"Discovered and registered {registered_count} presets from {experiments_dir}")
    return registered_count


def _infer_tags(filename: str, category: str) -> list[str]:
    """Infer tags from filename and category.

    Args:
        filename: YAML filename
        category: Inferred category

    Returns:
        List of tags
    """
    tags = []

    # Add category tag
    tags.append(category)

    # Add experiment type tags
    filename_lower = filename.lower()
    if "baseline" in filename_lower or "01_" in filename_lower:
        tags.append("baseline")
    if "ablation" in filename_lower:
        tags.append("ablation")
    if "robust" in filename_lower:
        tags.append("robustness")
    if "hpo" in filename_lower or "hyperparameter" in filename_lower:
        tags.append("hpo")
    if "ensemble" in filename_lower:
        tags.append("ensemble")
    if "federated" in filename_lower:
        tags.append("federated")
    if "transformer" in filename_lower or "vit" in filename_lower:
        tags.append("transformer")
    if "mamba" in filename_lower:
        tags.append("mamba")
    if "llm" in filename_lower or "foundation" in filename_lower:
        tags.append("foundation")

    return tags


def get_baseline_presets() -> list[ConfigPreset]:
    """Get baseline presets for each training paradigm.

    Returns:
        List of baseline ConfigPreset instances
    """
    registry = PresetRegistry()
    baselines = []

    for category in ["gan", "diffusion", "reconstruction", "vae", "ssl"]:
        presets = registry.filter(category=category, tag="baseline")
        if presets:
            baselines.append(presets[0])  # First baseline for category
        else:
            # Fall back to first preset in category
            all_in_category = registry.filter(category=category)
            if all_in_category:
                baselines.append(all_in_category[0])

    return baselines
