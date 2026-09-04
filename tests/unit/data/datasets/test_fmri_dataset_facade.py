"""Pins the fmri_dataset facade created by the Wave 0 exit-criterion split (#1400).

``fmri_dataset.py`` was 427 LOC against the 300 ceiling (NN20). Its three
datasets now live in sibling modules and are re-exported from the original path.
These tests pin the two things a facade can silently get wrong: a re-export that
is a *copy* rather than the definition, and an import graph that only happens to
work in one import order.
"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestFacadeIdentity:
    @pytest.mark.parametrize(
        ("name", "module"),
        [
            ("FMRIVolumeDataset", "spectramr.data.datasets.fmri_volume_dataset"),
            ("build_fmri_index", "spectramr.data.datasets.fmri_volume_dataset"),
            ("FMRIBoldSeriesDataset", "spectramr.data.datasets.fmri_bold_series_dataset"),
            ("CorticalSurfaceDataset", "spectramr.data.datasets.cortical_surface_dataset"),
        ],
    )
    def test_facade_exposes_the_definition_not_a_copy(self, name: str, module: str) -> None:
        """One owner per symbol (NN17)."""
        from spectramr.data.datasets import fmri_dataset

        assert getattr(fmri_dataset, name) is getattr(importlib.import_module(module), name)

    def test_all_lists_exactly_the_four_public_names(self) -> None:
        from spectramr.data.datasets import fmri_dataset

        assert set(fmri_dataset.__all__) == {
            "CorticalSurfaceDataset",
            "FMRIBoldSeriesDataset",
            "FMRIVolumeDataset",
            "build_fmri_index",
        }

    def test_package_init_still_re_exports_through_the_facade(self) -> None:
        from spectramr.data import datasets
        from spectramr.data.datasets import fmri_dataset

        assert datasets.FMRIVolumeDataset is fmri_dataset.FMRIVolumeDataset
        assert datasets.CorticalSurfaceDataset is fmri_dataset.CorticalSurfaceDataset


class TestImportGraphIsAcyclic:
    """The surface dataset subclasses the volume one, so a facade that both
    re-exported the surface dataset and defined its base would be circular. The
    volume half moved out for exactly that reason -- pin it, because the cycle
    would only show up under one import order."""

    @pytest.mark.parametrize(
        "first",
        [
            "spectramr.data.datasets.cortical_surface_dataset",
            "spectramr.data.datasets.fmri_bold_series_dataset",
            "spectramr.data.datasets.fmri_volume_dataset",
            "spectramr.data.datasets.fmri_dataset",
        ],
    )
    def test_any_module_imports_first_without_a_cycle(self, first: str) -> None:
        for mod in list(sys.modules):
            if (mod.startswith("spectramr.data.datasets.") and "fmri" in mod) or mod.endswith(
                "cortical_surface_dataset"
            ):
                sys.modules.pop(mod, None)
        importlib.import_module(first)  # a cycle raises ImportError here

    def test_the_facade_holds_no_dataset_definitions(self) -> None:
        """A definition left behind is a second owner waiting to drift."""
        import inspect

        from spectramr.data.datasets import fmri_dataset

        own = [
            n
            for n, v in vars(fmri_dataset).items()
            if inspect.isclass(v) and getattr(v, "__module__", "") == fmri_dataset.__name__
        ]
        assert not own, f"facade defines its own classes: {own}"
