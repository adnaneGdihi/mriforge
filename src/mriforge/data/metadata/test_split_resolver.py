"""Resolve the held-out test split's input paths from the paired v4 manifest.

The test split is the unpaired-ULF cohort (``split_hint == "test"``): those
records carry no HF target, so they are inference-only and never reach a
training loader. This module is the single place that turns that manifest
cohort into a concrete roster of input files.

Lives under ``data/`` rather than beside a builder because it is a
file->paths concern (CLAUDE.md non-negotiable #7): the inference pipeline
consumes it directly instead of routing a manifest read through a director.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["resolve_manifest_test_paths"]


def resolve_manifest_test_paths(data_config) -> list[Path]:
    """Input paths for the held-out test split from the paired v4 manifest.

    ``allow_unpaired`` is forced on (the cohort is unpaired by construction)
    and the direction is pinned to ``ulf_to_hf`` so the bidirectional filter
    never drops them: the 64mT scan is always the network input.

    Args:
        data_config: The resolved ``data:`` config block.

    Returns:
        Resolved input paths, or ``[]`` when no paired manifest is configured.
    """
    manifest_path = data_config.source.paired_manifest_path
    if not manifest_path:
        return []

    from types import SimpleNamespace

    from mriforge.config.schemas.data import (
        DataPairingConfigSchema,
        DataSplitConfigSchema,
    )
    from mriforge.data.metadata.index_builder import IndexBuilder
    from mriforge.data.metadata.path_resolver import PathResolver

    resolved_manifest = PathResolver.resolve(manifest_path)
    # `load_paired_bids_manifest` walks `config.split.type` and the whole
    # `config.pairing` block, so this synthetic config must carry both
    # sub-blocks a real DataConfigSchema has. Built from the schemas rather
    # than hand-written: a stand-in that restates defaults is how the same
    # read broke here in the first place.
    test_cfg = SimpleNamespace(
        pairing=DataPairingConfigSchema(
            # the test cohort is unpaired by construction
            allow_unpaired=True,
            # the 64mT scan is the input here; never swap or drop an arm
            bidirectional_mode="ulf_to_hf",
            contrasts=data_config.pairing.contrasts,
        ),
        split=DataSplitConfigSchema(),
    )
    records = IndexBuilder.load_paired_bids_manifest(
        manifest_path=Path(resolved_manifest), split="test", config=test_cfg
    )
    paths: list[Path] = []
    for rec in records:
        primary = rec.get("primary_path")
        if primary:
            paths.append(Path(PathResolver.resolve(primary)))
    return paths
