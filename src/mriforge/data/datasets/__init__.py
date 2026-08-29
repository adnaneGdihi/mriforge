"""Data datasets module — concrete Dataset classes.

Provides dataset adapters and specialized datasets for various medical
imaging formats. Dataset *construction* on the live path is the responsibility
of :class:`mriforge.data.builders.dataset_instantiator.DatasetInstantiator`, via
:class:`mriforge.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`.

.. warning::
   **A parallel dataset-construction path is still live here.** An earlier
   version of this docstring described the legacy ``create_dataset`` /
   ``register_dataset`` shims and the ``DatasetRegistry`` bridge as deleted in
   2026-05 (audit-17 / D18) and pointed at a regression guard by filename.
   Neither the deletion nor the guard file was real, and the claim went
   unchallenged for months precisely because it named a test.

   What is actually here: :mod:`mriforge.data.datasets.api` still exports
   ``create_dataset`` / ``register_dataset`` / ``get_registry``, and importing
   it runs ``initialize_dataset_registry()``, which **monkeypatches**
   ``DatasetRegistry.create_dataset`` at import time
   (``datasets/factory.py``). Because that rebinding is an import side effect,
   what the method does depends on whether something else imported this module
   first. The path is not dormant either — the leaf ``DatasetBuilder``
   (``infrastructure/builders/leaf/data_builders.py``) calls into it.

   The two vocabularies **disagree**: the parallel factory recognises roughly
   ``{auto, fastmri, fastmri_h5, image, nifti, synthetic}`` while the canonical
   instantiator serves the full ``dataset_type`` set — so a type that resolves
   on the live path can fail on this one. Retiring the parallel registry
   belongs with the registry conversion (D26), not here; until then this note
   is the honest description. Pinned by
   ``tests/unit/data/datasets/test_datasets_package_surface.py``.
"""

from .fmri_dataset import CorticalSurfaceDataset, FMRIVolumeDataset
from .inference_dataset import InferenceDataset
from .meta_learning_dataset import MetaLearningDataset
from .preprocessed_dataset import PreprocessedMRIDataset
from .slice_dataset import SliceDataset
from .synthetic_mri_dataset import SyntheticMRIDataset
from .universal_dataset import UniversalMRIDataset

__all__ = [
    "CorticalSurfaceDataset",
    "FMRIVolumeDataset",
    "InferenceDataset",
    "MetaLearningDataset",
    "PreprocessedMRIDataset",
    "SliceDataset",
    "SyntheticMRIDataset",
    "UniversalMRIDataset",
]
