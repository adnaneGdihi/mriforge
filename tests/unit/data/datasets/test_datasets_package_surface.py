"""What ``spectramr.data.datasets`` actually exports, and what it must not claim.

A12 (data-layer audit). The package docstring asserted that the legacy
``create_dataset`` / ``register_dataset`` shims and the parallel
``DatasetRegistry`` bridge "were removed in 2026-05-15 (audit-17 / D18)", and
named ``tests/unit/data/test_d18_legacy_dataset_factory_deleted.py`` as the
regression guard.

**Both claims were false.** The shims are still exported and still monkeypatch
``DatasetRegistry.create_dataset`` at import time, and the named guard file has
never existed — so the claim had nothing behind it and had been read as settled
for months.

These tests pin the REAL state. When the parallel registry is genuinely retired
(with D26's registry conversion), the assertions here flip and the docstring
warning comes out — deliberately, in the same change.
"""

from __future__ import annotations

import importlib

import pytest


def test_the_removal_claim_is_not_restated_without_a_guard() -> None:
    """The docstring must not re-assert a deletion that has not happened.

    Checked as an implication rather than a keyword ban: it is fine to name
    ``create_dataset`` (the warning does), but not to call it removed while it
    is still importable.
    """
    import spectramr.data.datasets as pkg

    doc = (pkg.__doc__ or "").lower()
    claims_removed = "were removed" in doc or "was removed" in doc
    if claims_removed:
        with pytest.raises(ImportError):
            importlib.import_module("spectramr.data.datasets.api")


def test_the_cited_guard_file_is_not_cited_unless_it_exists() -> None:
    """A docstring that names a test file makes readers stop looking."""
    from pathlib import Path

    import spectramr.data.datasets as pkg

    doc = pkg.__doc__ or ""
    repo = Path(__file__).resolve().parents[4]
    for token in doc.split():
        cleaned = token.strip("`,;:()").rstrip(".")
        if cleaned.startswith("tests/") and cleaned.endswith(".py"):
            assert (repo / cleaned).exists(), (
                f"the package docstring cites {cleaned!r} as a guard, but that file does not exist"
            )


class TestTheParallelRegistryIsStillLive:
    """Reality, asserted — so the next reader is not misled again."""

    def test_the_legacy_api_module_still_exists(self) -> None:
        """``register_dataset`` is deliberately NOT in this list any more.

        It called ``registry.register(name, dataset_class)`` against a
        classmethod whose signature is ``(entity: DatasetEntity)`` -- one
        argument, and an entity rather than a (name, class) pair -- so every
        call raised ``TypeError``. Registration has one owner:
        ``datasets.registry.register_dataset`` (non-negotiable 17).
        """
        api = importlib.import_module("spectramr.data.datasets.api")
        for name in ("create_dataset", "get_registry"):
            assert hasattr(api, name), f"{name} vanished — update the docstring"

    def test_importing_it_monkeypatches_the_domain_registry(self) -> None:
        """``api``'s import side effect rebinds a domain-layer method.

        This is the part that makes the surface hard to reason about: the
        binding changes as a side effect of an import, so what
        ``DatasetRegistry.create_dataset`` does depends on whether some other
        module happened to import ``datasets.api`` first.
        """
        importlib.import_module("spectramr.data.datasets.api")
        from spectramr.domain.entities.data.dataset_registry import DatasetRegistry

        assert DatasetRegistry.create_dataset.__name__ == "create_dataset_wrapper", (
            "the import-time monkeypatch is gone — update the docstring warning"
        )

    def test_it_has_a_live_production_consumer(self) -> None:
        """Not dormant: the leaf DatasetBuilder calls it.

        This asserted that an import STRING appeared in a source file. A source
        grep cannot tell a working entry point from a broken one, which is why
        the sibling ``register_dataset`` -- ``registry.register(name, cls)``
        against a classmethod taking ``(entity)``, i.e. ``TypeError`` on every
        call -- lived here undetected. Call the thing instead.
        """
        from spectramr.data.datasets.api import create_dataset, get_registry

        # Reachable and callable, not merely mentioned.
        assert callable(create_dataset)
        registry = get_registry()
        assert hasattr(registry, "create_dataset"), (
            "the import-time bridge in datasets/factory.py did not attach "
            "create_dataset to DatasetRegistry"
        )

    def test_the_broken_register_dataset_is_gone(self) -> None:
        """``datasets.api.register_dataset`` raised TypeError on every call.

        Registration has one owner: ``datasets.registry.register_dataset``, the
        registry ``DataPipelineDirector`` dispatches ``dataset_type`` on
        (non-negotiable 17).
        """
        import spectramr.data.datasets.api as ds_api

        assert not hasattr(ds_api, "register_dataset"), (
            "a second dataset-registration entry point is back; the live one is "
            "spectramr.data.datasets.registry.register_dataset"
        )
        from spectramr.data.datasets.registry import register_dataset

        assert callable(register_dataset)


def test_synthetic_paired_dataset_is_gone() -> None:
    """D20: deleted — no instantiator branch, no factory branch, no strategy.

    The two arms mentioning ``synthetic_paired`` set
    ``training.synthetic_paired_manifest``, a different key that
    ``DataEfficiencyHarnessStrategy`` only existence-checks and never loads, so
    nothing ever constructed this class.
    """
    import spectramr.data.datasets as pkg

    assert not hasattr(pkg, "SyntheticPairedDataset")
    with pytest.raises(ImportError):
        importlib.import_module("spectramr.data.datasets.synthetic_paired_dataset")
