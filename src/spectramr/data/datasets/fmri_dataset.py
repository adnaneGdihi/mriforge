"""fMRI volume + cortical-surface datasets (fMRI plan §§1 and 3).

Two minimal-viable Dataset classes for the fMRI 2026 plan:

* :class:`FMRIVolumeDataset` — reads NIfTI 4-D BOLD volumes
  (`.nii` / `.nii.gz`). Each sample is a single volume with optional
  TR / phase-encode metadata routed through the batch dict so the
  downstream EPI strategy can read them.
* :class:`CorticalSurfaceDataset` — pairs each volume with a
  precomputed conformal-flattening grid stored alongside as a NumPy
  array (``<subject>_cortex_flatten.npy``) or, if absent, populates
  the grid lazily from the
  :func:`attach_cortex_flatten_grid` transform.

Both datasets ingest a flat directory of files; for real cohorts
callers can subclass or wire through :class:`DataPipelineDirector`.
NIfTI / NumPy reads are deferred to ``nibabel`` / ``numpy``. A volume
whose I/O fails raises from ``__getitem__`` — it is NOT replaced by a
zeros placeholder (that would train the model on fabricated data;
pitfall #9/#16).
"""

# Facade. The three datasets live in sibling modules (300-LOC ceiling, NN20) and
# are re-exported here under their original names, so every existing importer --
# datasets/__init__.py, dataset_instantiator.py and five test modules -- resolves
# them through this path unchanged, against one definition each (NN17).

from __future__ import annotations

from spectramr.data.datasets.cortical_surface_dataset import CorticalSurfaceDataset
from spectramr.data.datasets.fmri_bold_series_dataset import FMRIBoldSeriesDataset
from spectramr.data.datasets.fmri_volume_dataset import FMRIVolumeDataset, build_fmri_index

__all__ = [
    "CorticalSurfaceDataset",
    "FMRIBoldSeriesDataset",
    "FMRIVolumeDataset",
    "build_fmri_index",
]
