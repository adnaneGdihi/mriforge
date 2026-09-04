"""Infrastructure Configuration Components
====================================

Concrete implementations of configuration components
following SOLID principles.
"""

import json
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from spectramr.config import IConfig, IConfigLoader, IConfigValidator
from spectramr.config.schemas.renames import UNDECLARED, override_knob

# Legacy data.coil_processing_mode → the four physics.coil_processing axes.
# Behavior-preserving: each row is the axis-combination that reproduces that mode
# (enforced by tests/unit/data/builders/test_coil_processing_parity.py).
_MODE_TO_AXES: dict[str, dict[str, str]] = {
    "none": {"compression": "none", "combine": "none", "domain": "kspace", "channels": "complex"},
    "flatten": {
        "compression": "none",
        "combine": "none",
        "domain": "kspace",
        "channels": "real_interleaved",
    },
    "svd": {
        "compression": "svd",
        "combine": "none",
        "domain": "kspace",
        "channels": "real_interleaved",
    },
    "magnitude": {
        "compression": "none",
        "combine": "rss",
        "domain": "source",
        "channels": "magnitude",
    },
    "rss": {
        "compression": "none",
        "combine": "rss",
        "domain": "kspace",
        "channels": "real_interleaved",
    },
    "rss_image": {
        "compression": "none",
        "combine": "rss",
        "domain": "image",
        "channels": "magnitude",
    },
}


def _derive_coil_processing_from_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """Synthesize the full 4-axis ``physics.coil_processing`` block from the legacy
    ``data.coil_processing_mode`` so the data-load can read ONE resolved block.

    Forward derivation (the inverse of :func:`_sync_coil_processing_to_legacy`):
    each legacy mode maps to a compression/combine/output-domain/output-channels
    combination (``_MODE_TO_AXES``) that reproduces it byte-for-byte. The merge is
    per-sub-key — a user-authored sub-block always wins, only *unset* axes are
    filled from the legacy mode. This both completes the already-migrated partial
    blocks (e.g. ``compression: {method: svd}`` gains the svd output axes) and
    lets a legacy-only config resolve to a complete block. No-op when there is no
    legacy mode (a pure new-block config uses the schema defaults = no-op behavior).
    Silent — the legacy knob is not deprecated.
    """
    data = raw.get("data")

    # Read the CANONICAL path first, then the legacy one. This runs before the
    # rename fold, so a drained arm presents only `data.coils.processing_mode`
    # and a legacy-spelling-only lookup silently no-ops -- leaving `physics`
    # unbuilt entirely, which is a resolved-document change from a rewrite that
    # is supposed to be a no-op. Caught by the drain oracle on
    # `physics_driven/experiment_53_multi_echo_b0_fit.yaml`; the same shape as
    # the `image_size` guard in data.py.
    def _coil_knob(canonical_leaf: str, legacy_leaf: str, default=None):
        """Read a `data.coils.*` knob under either spelling, canonical first.

        All three of these fold to `data.coils.*`, and this runs BEFORE the
        fold, so a drained arm presents only the canonical spelling.
        """
        if not isinstance(data, dict):
            return default
        coils = data.get("coils")
        if isinstance(coils, dict) and canonical_leaf in coils:
            return coils[canonical_leaf]
        return data.get(legacy_leaf, default)

    mode = _coil_knob("processing_mode", "coil_processing_mode")
    if mode not in _MODE_TO_AXES:
        return raw
    axes = _MODE_TO_AXES[mode]

    physics = raw.get("physics")
    if not isinstance(physics, dict):
        physics = {}
        raw["physics"] = physics
    cp = physics.get("coil_processing")
    if not isinstance(cp, dict):
        cp = {}
        physics["coil_processing"] = cp

    comp = cp.setdefault("compression", {})
    if isinstance(comp, dict):
        comp.setdefault("method", axes["compression"])
        if axes["compression"] == "svd":
            comp.setdefault(
                "num_virtual_coils",
                _coil_knob("num_virtual_coils", "num_virtual_coils", 4),
            )
            cal = _coil_knob("svd_calibration_lines", "svd_calibration_lines")
            if cal is not None:
                comp.setdefault("calibration_lines", cal)
    combine = cp.setdefault("combine", {})
    if isinstance(combine, dict):
        combine.setdefault("method", axes["combine"])
    output = cp.setdefault("output", {})
    if isinstance(output, dict):
        output.setdefault("domain", axes["domain"])
        output.setdefault("channels", axes["channels"])
    return raw


def _sync_coil_processing_to_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """Drive the legacy ``data.coil_processing_mode`` machinery from the unified
    ``physics.coil_processing`` block (compression axis).

    The battle-tested per-sample compression path
    (``FastMRISubjectBuilder._apply_coil_processing``) reads the string
    ``data.coil_processing_mode``. When a user authors the new unified block,
    this sync derives the legacy mode from it so the new knob actually executes
    compression — without reimplementing it or double-applying (pitfall #15).

    Opt-in and silent: a no-op when the new block is absent, so legacy YAMLs that
    use ``data.coil_processing_mode`` load byte-for-byte unchanged (the knob is
    NOT deprecated — there is deliberately no legacy→new migration that would
    warn on every such config). Only the *compression* axis maps to
    ``coil_processing_mode``; the estimation axis is consumed by the strategy
    smaps path and the combine axis by ``coil_combine`` target generation. An
    explicit new block wins over a conflicting legacy mode. Raises
    ``NotImplementedError`` on the reserved ``gcc`` method (no silent no-op —
    pitfall #9).
    """
    physics = raw.get("physics")
    if not isinstance(physics, dict):
        return raw
    cp = physics.get("coil_processing")
    if not isinstance(cp, dict):
        return raw
    comp = cp.get("compression")
    if not isinstance(comp, dict):
        return raw
    method = comp.get("method", "none")
    data = raw.get("data")
    if not isinstance(data, dict):
        data = {}
        raw["data"] = data
    if method == "svd":
        # Canonical path, and the legacy spelling cleared with it. This bridge
        # used to write `data.coil_processing_mode`, which collided with an arm
        # already migrated to `data.coils.processing_mode` ("two spellings
        # disagree"). `override_knob` rather than `default_knob` because the
        # docstring's contract is that the new block WINS -- the v6.1 template
        # authors `coils.processing_mode: 'none'`, so deferring would make the
        # unified block inert for every arm derived from it (pitfall #16).
        derived = {"coil_processing_mode": "svd"}
        if "num_virtual_coils" in comp:
            derived["num_virtual_coils"] = comp["num_virtual_coils"]
        if comp.get("calibration_lines") is not None:
            derived["svd_calibration_lines"] = comp["calibration_lines"]
        for leaf, value in derived.items():
            displaced = override_knob(data, "data", leaf, value)
            if displaced is not UNDECLARED and displaced != value:
                logging.getLogger(__name__).info(
                    "[COIL SYNC] physics.coil_processing.compression -> "
                    "data.%s=%r (was %r; the unified block wins)",
                    leaf,
                    value,
                    displaced,
                )
    elif method == "gcc":
        raise NotImplementedError(
            "physics.coil_processing.compression.method='gcc' is reserved but "
            "not implemented; use 'svd' or 'none'."
        )
    # method == "none": leave data.coil_processing_mode at its existing value.
    return raw


class YAMLConfigLoader(IConfigLoader):
    """YAML-based configuration loader."""

    def load_from_file(self, path: Path) -> IConfig:
        """Loads configuration from YAML file."""
        if yaml is None:
            msg = "PyYAML not available. Install with: pip install PyYAML"
            raise ImportError(msg)
        with open(path) as f:
            data = yaml.safe_load(f)
        return Config(data)

    def load_from_dict(self, config_dict: dict[str, Any]) -> IConfig:
        """Loads configuration from dictionary."""
        return Config(config_dict)


class JSONConfigLoader(IConfigLoader):
    """JSON-based configuration loader."""

    def load_from_file(self, path: Path) -> IConfig:
        """Loads configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return Config(data)

    def load_from_dict(self, config_dict: dict[str, Any]) -> IConfig:
        """Loads configuration from dictionary."""
        return Config(config_dict)


class ConfigValidator(IConfigValidator):
    """Configuration validator."""

    def __init__(self) -> None:
        """__init__."""
        self._errors: list[str] = []

    def validate(self, config: IConfig) -> bool:
        """Validates configuration."""
        self._errors = []
        # Add validation logic here
        return len(self._errors) == 0

    def get_validation_errors(self) -> list[str]:
        """Returns validation errors."""
        return self._errors.copy()


class Config(IConfig):
    """Concrete configuration implementation.

    This class follows Interface Segregation by implementing
    multiple focused interfaces rather than one large interface.
    """

    def __init__(self, data: dict[str, Any]):
        """__init__.

        Args:
            data (dict[str, Any]): Description.
        """
        self._data = data
        self._validator = ConfigValidator()

    def save_to_file(self, path: Path) -> None:
        """Saves configuration to file."""
        if yaml is None:
            msg = "PyYAML not available. Install with: pip install PyYAML"
            raise ImportError(msg)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def update_from_dict(self, updates: dict[str, Any]) -> None:
        """Updates configuration from dictionary."""
        self._data.update(updates)

    def get_section(self, section_name: str) -> dict[str, Any]:
        """Gets a configuration section."""
        return dict(self._data.get(section_name, {}))

    # IConfigLoader implementation
    def load_from_file(self, path: Path) -> IConfig:
        """Loads configuration from file."""
        loader = YAMLConfigLoader()
        return loader.load_from_file(path)

    def load_from_dict(self, config_dict: dict[str, Any]) -> IConfig:
        """Loads configuration from dictionary."""
        loader = YAMLConfigLoader()
        return loader.load_from_dict(config_dict)

    # IConfigValidator implementation
    def validate(self, config: IConfig) -> bool:
        """Validates configuration."""
        return self._validator.validate(config)

    def get_validation_errors(self) -> list[str]:
        """Returns validation errors."""
        return self._validator.get_validation_errors()

    # Training config properties
    @property
    def epochs(self) -> int:
        """epochs.

        Returns:
            int: Description.
        """
        val = self._data.get("epochs")
        return int(val) if val is not None else 200

    @property
    def batch_size(self) -> int:
        """batch_size.

        Returns:
            int: Description.
        """
        val = self._data.get("batch_size")
        return int(val) if val is not None else 4

    @property
    def learning_rate(self) -> float:
        """learning_rate.

        Returns:
            float: Description.
        """
        val = self._data.get("learning_rate")
        return float(val) if val is not None else 1e-5

    @property
    def model_type(self) -> str:
        """model_type.

        Returns:
            str: Description.
        """
        return str(self._data.get("model_type", "unet"))

    # Model config properties
    @property
    def in_channels(self) -> int:
        """in_channels.

        Returns:
            int: Description.
        """
        val = self._data.get("in_channels")
        return int(val) if val is not None else 1

    @property
    def out_channels(self) -> int:
        """out_channels.

        Returns:
            int: Description.
        """
        val = self._data.get("out_channels")
        return int(val) if val is not None else 1

    @property
    def model_kwargs(self) -> dict[str, Any]:
        """model_kwargs.

        Returns:
            dict[str, Any]: Description.
        """
        return dict(self._data.get("model_kwargs", {}))

    # Data config properties
    @property
    def input_lr_dir(self) -> str:
        """input_lr_dir.

        Returns:
            str: Description.
        """
        return str(self._data.get("input_lr_dir", "./data_synthetic/A_LRSI"))

    @property
    def input_hr_dir(self) -> str:
        """input_hr_dir.

        Returns:
            str: Description.
        """
        return str(self._data.get("input_hr_dir", "./data_synthetic/A_HRSI"))


class ConfigFactory:
    """Factory for creating configuration objects.

    Follows the Factory pattern and allows easy extension
    for different configuration formats.
    """

    def __init__(self) -> None:
        """__init__."""
        self._loaders: dict[str, IConfigLoader] = {
            ".yaml": YAMLConfigLoader(),
            ".yml": YAMLConfigLoader(),
            ".json": JSONConfigLoader(),
        }

    def create_from_file(self, path: Path) -> IConfig:
        """Creates configuration from file."""
        suffix = path.suffix.lower()
        if suffix not in self._loaders:
            raise ValueError(f"Unsupported config format: {suffix}")

        return self._loaders[suffix].load_from_file(path)

    def create_from_dict(self, config_dict: dict[str, Any]) -> IConfig:
        """Creates configuration from dictionary."""
        return YAMLConfigLoader().load_from_dict(config_dict)

    def register_loader(self, extension: str, loader: IConfigLoader) -> None:
        """Registers a new configuration loader."""
        self._loaders[extension] = loader


# Global config factory instance
_config_factory = None


def get_config_factory() -> ConfigFactory:
    """Gets the global configuration factory."""
    global _config_factory
    if _config_factory is None:
        _config_factory = ConfigFactory()
    return _config_factory
