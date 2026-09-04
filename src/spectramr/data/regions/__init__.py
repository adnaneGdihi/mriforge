"""Region-axis data: the SynthSeg mask cache and the region registry.

The sweep consumes cached masks and **never calls a segmenter**: SynthSeg is a 3-D
network that cannot run on a single slice, and re-segmenting a degraded frame
would let the region boundary move with the artifact. See
:mod:`spectramr.data.regions.synthseg_labels`.
"""

from __future__ import annotations

from spectramr.data.regions.mask_cache import (
    CACHE_FORMAT_VERSION,
    DEFAULT_MAX_DROP_RATE,
    DEFAULT_QC_FLOOR,
    LABELS_KEY,
    RegionCacheFormatError,
    RegionCacheReport,
    RegionMaskCache,
    TissueAxisUnsoundError,
    VolumeQC,
    load_region_cache,
    write_volume_cache,
)
from spectramr.data.regions.region_registry import (
    REGION_SPECS,
    RegionSpec,
    RegionTier,
    control_id_for,
    is_control_id,
    lesion_region_id,
    region_spec,
    tissue_region_ids,
)
from spectramr.data.regions.synthseg_labels import (
    DK_PARCELLATION_LABELS,
    LABEL_TO_TISSUE,
    TissueClass,
    UnmappedSynthSegLabelError,
    brain_mask,
    label_map_to_tissue,
    structure_mask,
    tissue_mask,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_MAX_DROP_RATE",
    "DEFAULT_QC_FLOOR",
    "DK_PARCELLATION_LABELS",
    "LABELS_KEY",
    "LABEL_TO_TISSUE",
    "REGION_SPECS",
    "RegionCacheFormatError",
    "RegionCacheReport",
    "RegionMaskCache",
    "RegionSpec",
    "RegionTier",
    "TissueAxisUnsoundError",
    "TissueClass",
    "UnmappedSynthSegLabelError",
    "VolumeQC",
    "brain_mask",
    "control_id_for",
    "is_control_id",
    "label_map_to_tissue",
    "lesion_region_id",
    "load_region_cache",
    "region_spec",
    "structure_mask",
    "tissue_mask",
    "tissue_region_ids",
    "write_volume_cache",
]
