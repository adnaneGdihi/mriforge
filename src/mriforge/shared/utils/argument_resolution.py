"""Utility functions for resolving arguments with fallback chains.

This module provides utilities for cleanly handling argument resolution
with multiple fallback options (CLI args > config > defaults).
"""

from typing import Any, TypeVar

T = TypeVar("T")


def resolve_argument(
    *sources: Any | None,
    default: T | None = None,
) -> T | None:
    """Resolve an argument from multiple fallback sources.

    Returns the first non-None, non-empty value from sources, or default.

    Args:
        *sources: Ordered sources to check (typically: CLI arg, config value, etc.)
        default: Default value if all sources are None/empty

    Returns:
        Resolved value or default

    Example:
        >>> args.input_dir = "/path/to/input"
        >>> config = {"input": "/config/input"}
        >>> resolve_argument(args.input_dir, config.get("input"), default="/default")
        '/path/to/input'
    """
    for source in sources:
        if source is not None and source != "":
            return source
    return default


def resolve_argument_pair(
    cli_attr: Any | None,
    cli_fallback_attr: Any | None,
    config_key: str,
    config_dict: dict[str, Any],
    config_fallback_key: str | None = None,
    default: T | None = None,
) -> T | None:
    """Resolve argument from CLI with fallback names and config file.

    Precedence: cli_attr > cli_fallback_attr > config_key > config_fallback_key
    > default

    Args:
        cli_attr: Primary CLI argument value (e.g., args.input_dir)
        cli_fallback_attr: Fallback CLI argument value (e.g., args.input)
        config_key: Primary config dictionary key (e.g., "input_dir")
        config_dict: Configuration dictionary
        config_fallback_key: Fallback config key (e.g., "input")
        default: Default value if all sources are None/empty

    Returns:
        Resolved value or default

    Example:
        >>> resolve_argument_pair(
        ...     args.input_dir, args.input,
        ...     "input_dir", config,
        ...     config_fallback_key="input",
        ...     default="/default"
        ... )
    """
    # Check CLI args first (with fallback)
    cli_value = cli_attr or cli_fallback_attr
    if cli_value is not None and cli_value != "":
        return cli_value

    # Fall back to config file
    config_value = config_dict.get(config_key)
    if config_value is not None and config_value != "":
        return config_value

    # Try config fallback key
    if config_fallback_key:
        config_fallback_value = config_dict.get(config_fallback_key)
        if config_fallback_value is not None and config_fallback_value != "":
            return config_fallback_value

    return default


def resolve_boolean_argument(
    cli_value: bool | None,
    config_key: str,
    config_dict: dict[str, Any],
    default: bool = False,
) -> bool:
    """Resolve a boolean argument from CLI or config.

    Args:
        cli_value: CLI argument value (may be None or False)
        config_key: Config dictionary key
        config_dict: Configuration dictionary
        default: Default value (typically False)

    Returns:
        Resolved boolean value

    Example:
        >>> resolve_boolean_argument(
        ...     args.use_ema,
        ...     "use_ema",
        ...     config,
        ...     default=True
        ... )
    """
    if cli_value is not None and cli_value is not False:
        return cli_value
    if config_key in config_dict:
        return bool(config_dict[config_key])
    return default


def resolve_with_getattr(
    obj: Any,
    *attr_names: str,
    config_dict: dict[str, Any] | None = None,
    config_keys: list[str] | None = None,
    default: T | None = None,
) -> T | None:
    """Resolve an attribute using getattr with multiple names + config fallback.

    Args:
        obj: Object to check attributes on
        *attr_names: Attribute names to check (in order)
        config_dict: Optional config dictionary for fallback
        config_keys: Optional config keys to check (in order)
        default: Default value if all sources are None/empty

    Returns:
        Resolved value or default

    Example:
        >>> resolve_with_getattr(
        ...     args,
        ...     "input_dir", "input",
        ...     config_dict=config,
        ...     config_keys=["input_dir", "input"],
        ...     default="/default"
        ... )
    """
    # Check object attributes
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if value is not None and value != "":
            return value

    # Check config dictionary
    if config_dict and config_keys:
        for key in config_keys:
            value = config_dict.get(key)
            if value is not None and value != "":
                return value

    return default


class ArgumentResolver:
    """Context manager for resolving arguments with consistent precedence."""

    def __init__(self, args: Any, config_dict: dict[str, Any] | None = None):
        """Initialize resolver.

        Args:
            args: argparse Namespace or similar object with attributes
            config_dict: Optional configuration dictionary
        """
        self.args = args
        self.config_dict = config_dict or {}

    def resolve_path(
        self,
        cli_attr: str,
        cli_fallback_attr: str | None = None,
        config_key: str | None = None,
        config_fallback_key: str | None = None,
        required: bool = False,
        error_message: str | None = None,
    ) -> str | None:
        """Resolve a file/directory path with full fallback chain.

        Args:
            cli_attr: Primary CLI attribute name
            cli_fallback_attr: Fallback CLI attribute name
            config_key: Primary config key
            config_fallback_key: Fallback config key
            required: If True, raise ValueError if not found
            error_message: Custom error message for required argument

        Returns:
            Resolved path or None

        Raises:
            ValueError: If required=True and no value found
        """
        cli_value = getattr(self.args, cli_attr, None)
        cli_fb_value = getattr(self.args, cli_fallback_attr, None) if cli_fallback_attr else None

        config_value = self.config_dict.get(config_key) if config_key else None
        config_fb_value = self.config_dict.get(config_fallback_key) if config_fallback_key else None

        result = cli_value or cli_fb_value or config_value or config_fb_value

        if required and not result:
            msg = error_message or f"Required path argument '{cli_attr}' not provided"
            raise ValueError(msg)

        return result

    def resolve_value(
        self,
        cli_attr: str,
        config_key: str | None = None,
        default: T | None = None,
    ) -> T | None:
        """Resolve a generic value with CLI > config > default precedence.

        Args:
            cli_attr: CLI attribute name
            config_key: Config key (defaults to cli_attr if not specified)
            default: Default value

        Returns:
            Resolved value
        """
        cli_value = getattr(self.args, cli_attr, None)
        config_key = config_key or cli_attr
        config_value = self.config_dict.get(config_key)

        return cli_value or config_value or default

    def resolve_device(self) -> str:
        """Resolve device with standard fallback chain.

        Returns:
            Device string ("cpu", "cuda", "auto", etc.)
        """
        return (
            self.resolve_value(
                "device",
                config_key="device",
                default="auto",
            )
            or "auto"
        )
