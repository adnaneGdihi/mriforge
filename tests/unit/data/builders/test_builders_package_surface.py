"""What `spectramr.data.builders` exports after the unrunnable lineage was cut.

Unit 6a. `DatasetBuilder(ABC)` / `DatasetBuilderRegistry` / `DatasetBuildConfig`
/ `validation.py` were an abstract-base + registry dataset-creation design that
predates `DatasetInstantiator`. They were not merely uncalled — they were
UNRUNNABLE: ten sites read `config.source.root` on a `DatasetBuildConfig` whose
field is `data_root`, so every entry point raised AttributeError on its own
config object. Wreckage of the `data_root` -> `source.root` block decomposition
that nothing noticed because nothing called it.
"""

from __future__ import annotations

import importlib

import pytest


def test_no_export_is_dangling() -> None:
    """`__all__` entries with no definition are how a pruned re-export rots.

    `validate_dataset_config` was left in `__all__` after its import went; it
    would have raised only on `from ... import *`, which nothing does.
    """
    import spectramr.data.builders as builders

    missing = [name for name in builders.__all__ if not hasattr(builders, name)]
    assert missing == [], f"__all__ names nothing defines: {missing}"


@pytest.mark.parametrize(
    "module",
    [
        "spectramr.data.builders.dataset_builder",
        "spectramr.data.builders.validation",
        "spectramr.data.builders.config",
        "spectramr.data.augmentation_interface",
    ],
)
def test_the_unrunnable_lineage_is_gone(module: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_the_live_substitutes_are_reachable() -> None:
    """Deleting is only safe because each dead thing has a live replacement.

    `DatasetInstantiator` supersedes `DatasetBuilderRegistry` (19 creators vs 3),
    and `bootstrap.py` carries its own `PathResolver`-based data-availability
    check — written deliberately because the deleted one used a raw
    `Path().exists()` that disagreed with how the loader resolves roots.
    """
    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

    assert hasattr(DatasetInstantiator, "create_datasets")

    import inspect

    from spectramr import bootstrap

    assert "_validate_data_availability_at_startup" in dir(bootstrap)
    assert "PathResolver.resolve" in inspect.getsource(
        bootstrap._validate_data_availability_at_startup
    )


def test_the_surviving_validate_name_is_a_different_function() -> None:
    """`validate_dataset_config` also exists in `config/schemas/validator_registry`
    — dict-based, live, and unrelated. The deleted one took a
    `DatasetBuildConfig` and shared only the name."""
    from spectramr.config.schemas.validator_registry import _validate_dataset_config

    assert _validate_dataset_config({"data": {"patch_size": 16}}) == []
