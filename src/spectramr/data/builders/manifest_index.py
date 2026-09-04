"""Build a dataset loader's native index from an external manifest (Mechanism A).

When an experiment selects a dataset via ``data.index_path:
data/manifests/external/<id>.json``, the non-Cartesian / paired loaders
(``bart_kspace``, ``ismrmrd_kspace``, ``oracle_bssfp``) build their index from the
manifest's ``records[]`` + embedded ``data_root`` instead of re-globbing the disk.
This makes ``index_path`` a genuinely *read* knob for every dataset family (no
unread-knob facade — CLAUDE.md pitfall #15) and gives one SSOT for which files a
run consumes (the populated manifest), removing the ``data_root``+glob duplication.

The manifest is produced by ``scripts/data/gen_external_dataset_manifests.py
--populate`` (records are ``{'relative_path', 'file_id'}`` relative to
``data_root``). This module is the pure path-construction half; per-file existence
is enforced downstream by the dataset/reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spectramr.data.metadata.path_resolver import PathResolver

__all__ = ["build_index_from_manifest", "read_external_manifest"]


def read_external_manifest(index_path: str | Path) -> dict[str, Any]:
    """Load + validate a populated external manifest.

    Raises ``FileNotFoundError`` if the manifest file is missing or its
    ``records[]`` is empty (an empty record list means the extracted data was
    never populated on the cluster — fail loud, never run on nothing), and
    ``ValueError`` if it carries no ``data_root``.
    """
    p = Path(index_path)
    if not p.is_file():
        raise FileNotFoundError(f"external manifest not found: {p}")
    manifest = json.loads(p.read_text())
    data_root = manifest.get("data_root")
    if not data_root:
        raise ValueError(
            f"external manifest {p} has no 'data_root' (cannot resolve records). "
            "In-house manifests use 'local_manifest' and a different load path."
        )
    records = manifest.get("records") or []
    if not records:
        raise FileNotFoundError(
            f"external manifest {p} has an empty 'records[]' — the extracted data "
            "was never populated. Run `gen_external_dataset_manifests.py --populate` "
            "on the cluster where databases/external/ lives (no silent fallback)."
        )
    return manifest


def build_index_from_manifest(
    index_path: str | Path,
    file_type: str,
    *,
    oracle_b0_map: str | None = None,
    file_pattern: str | None = None,
) -> list[Any]:
    """Return the loader's native index from a populated external manifest.

    - ``bart`` / ``ismrmrd`` / ``twix`` → ``[str(data_root / relative_path), ...]``
      (the same flat-path shape ``build_bart_index`` / ``build_ismrmrd_index`` return;
      the ``twix`` reader is ``spectramr.data.twix`` via ``TwixStrategy``).
    - ``oracle_bssfp`` → ``[{'stack': ..., 'b0_map': ...}, ...]`` pairing each
      stack record with the shared B0 map from the format sub-block
      (``oracle_b0_map``, required for this format).

    ``file_pattern`` (e.g. ``'b1map'``) restricts the bart/ismrmrd/twix records to
    those whose ``relative_path`` stem contains it — the manifest analogue of the
    ``build_bart_index`` glob filter. Without it a mixed-acquisition manifest
    loads radial siblings into a Cartesian arm and crashes the canonicalizer
    (F3 — ``data.bart.file_pattern`` was a read knob on the glob branch only,
    pitfall #15).

    Raises ``ValueError`` for an unsupported ``file_type`` (no silent fallback).
    """
    manifest = read_external_manifest(index_path)
    data_root = Path(PathResolver.resolve(manifest["data_root"]))
    records = manifest["records"]

    if file_type in ("bart", "ismrmrd", "twix"):
        if file_pattern:
            records = [r for r in records if file_pattern in Path(r["relative_path"]).stem]
        return [str(data_root / r["relative_path"]) for r in records]

    if file_type == "oracle_bssfp":
        if not oracle_b0_map:
            raise ValueError(
                "oracle_bssfp manifest index requires the shared b0_map from "
                "data.oracle_bssfp.b0_map (the phase-cycled stack records pair "
                "with one absolute B0 reference)."
            )
        b0 = str(data_root / oracle_b0_map)
        return [{"stack": str(data_root / r["relative_path"]), "b0_map": b0} for r in records]

    raise ValueError(
        f"build_index_from_manifest: unsupported file_type {file_type!r} "
        "(supported: bart, ismrmrd, twix, oracle_bssfp)."
    )
