"""Manifest builders for dataset splits.

Small, dependency-light scripts that read raw-dataset metadata and emit the
v3.0 JSON manifests consumed by
:class:`spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`.

These builders live under :mod:`spectramr.data` (the data-loading SSOT, see
CLAUDE.md pitfall #11) so that anything turning files into a manifest stays in
one place. They are *offline* tools: run once to produce a manifest, then point
an experiment YAML's ``data.index_path`` / ``data.validation_index_path`` at it.
"""

from spectramr.data.manifests.build_m4raw_motion_subset import (
    build_motion_subset_manifest,
    filter_motion_subjects,
)

__all__ = [
    "build_motion_subset_manifest",
    "filter_motion_subjects",
]
