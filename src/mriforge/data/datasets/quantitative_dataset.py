"""Quantitative MRI dataset (Phase 4b).

A standalone dataset for parameter-map regression tasks where both the
input contrasts (T1w / T2w / PDw / ...) and the target quantitative maps
(T1, T2, T2*, PD, ADC) live on disk.

Glob convention
===============

Per-subject record (one row in the index)::

    {
        "input_paths": ["sub-001_T1w.nii.gz", "sub-001_T2w.nii.gz", ...],
        "map_paths":   {"t1": "sub-001_T1map.nii.gz",
                        "t2": "sub-001_T2map.nii.gz",
                        "pd": "sub-001_PDmap.nii.gz"},
        "subject_id":  "sub-001",
    }

The dataset yields a :class:`tio.Subject` per record with the input
contrasts stacked as a single multi-channel ScalarImage under the key
``input``, plus one ScalarImage per declared target map. A downstream
``LoadAcquisitionMetadata`` transform (Phase 4b) attaches the per-sample
acquisition scalars if enabled.

This dataset is intentionally small and focused. Use UniversalMRIDataset
for k-space-based reconstruction with optional parameter-map auxiliary
outputs; this one is for the strictly quantitative regression task.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torchio as tio
from torch.utils.data import Dataset

from mriforge.config.schemas.data import QuantitativeConfigSchema

__all__ = [
    "QuantitativeMapDataset",
    "build_quantitative_index",
    "load_quantitative_manifest",
]


def _declared_map_paths(record: dict[str, Any], target_maps: list[str]) -> list[str]:
    """The record's map paths for ``target_maps``, normalised for compare."""
    map_paths = record.get("map_paths") or {}
    return [str(Path(map_paths[m])) for m in target_maps if m in map_paths]


def _input_is_the_maps(record: dict[str, Any], target_maps: list[str]) -> bool:
    """True when ``input_paths`` is the same *set of files* as the declared maps.

    ``scripts/data/build_nist_mrf_manifest.py`` writes records this way on
    purpose (the stacked parameter maps ARE the generative model's data), so
    the condition is a legitimate mode -- but it is also what makes ``input``
    and ``target`` the same tensor, which is the identity for any arm that
    means to *predict* the maps. It is declared, never inferred; see
    ``QuantitativeConfigSchema.input_source``.

    Compared as a multiset, deliberately. The question is whether the inputs
    are the same files as the maps, not whether two channel axes happen to
    agree: ``input`` is ordered by ``input_paths`` (the order the generator was
    invoked with) while ``target`` is ordered by ``target_maps`` (which an arm
    sets to match its consumer -- ``[pd, t1, t2]`` for the Bloch manifold's
    (M0, T1, T2) coordinates). An ordered compare would make a correct arm
    unconstructible because of how someone invoked a script on the cluster.
    """
    declared = _declared_map_paths(record, target_maps)
    if len(declared) != len(target_maps):
        return False
    return sorted(str(Path(p)) for p in record.get("input_paths", [])) == sorted(declared)


def _load_volume(path: Path) -> torch.Tensor:
    """Load a 3D volume from NIfTI / NumPy / H5 to a [C, X, Y, Z] tensor.

    Kept small and inline — the canonical IO strategies handle k-space and
    paired NIfTI; quantitative maps are simple intensity volumes so we
    don't need the full IOStrategyFactory here.
    """
    # DICOM-stack directory (e.g. NIST-MRF parameter maps stored as a 7-slice
    # .IMA series per MAP_MASKED dir). Delegate to the canonical DicomStrategy,
    # which finds slices by the DICM magic byte (so .IMA works) and stacks by
    # InstanceNumber. Done before the suffix switch because a directory has none.
    if path.is_dir():
        from mriforge.data.io_strategies import DicomStrategy

        data = DicomStrategy().load(str(path)).get("data")
        if data is None:
            raise ValueError(f"DICOM directory {path} produced no 'data' volume.")
        tensor = (
            data.float()
            if isinstance(data, torch.Tensor)
            else torch.as_tensor(data, dtype=torch.float32)
        )
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        return tensor

    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "QuantitativeMapDataset requires nibabel for NIfTI loading. "
                "Install with `pip install nibabel`."
            ) from e
        vol = nib.load(str(path)).get_fdata()  # type: ignore[no-untyped-call]
        tensor = torch.as_tensor(vol, dtype=torch.float32)
    elif suffix.endswith(".npy"):
        tensor = torch.from_numpy(  # type: ignore[no-untyped-call]
            __import__("numpy").load(str(path))
        ).to(torch.float32)
    elif suffix.endswith(".pt"):
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        tensor = (
            loaded.float()
            if isinstance(loaded, torch.Tensor)
            else torch.as_tensor(loaded, dtype=torch.float32)
        )
    else:
        raise ValueError(
            f"Unsupported quantitative volume format: {path}. Use .nii / .nii.gz / .npy / .pt."
        )
    # Normalize to (1, X, Y, Z) — TorchIO ScalarImage convention.
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    return tensor


class QuantitativeMapDataset(Dataset):
    """Yields TorchIO Subjects with input contrasts + named quantitative maps.

    Args:
        index: List of records (see module docstring).
        quantitative_config: Populated :class:`QuantitativeConfigSchema`.
            Determines which maps to emit and in what units.
        transform: Optional TorchIO transform pipeline applied after the
            Subject is built. The transform pipeline typically contains
            :class:`LoadAcquisitionMetadata` to attach the per-sample
            scan parameters.

    Yields per call to ``__getitem__``::

        tio.Subject {
            input:           (n_contrasts, X, Y, Z),   ScalarImage
            target:          (n_maps, X, Y, Z),        ScalarImage (maps stacked)
            t1, t2, pd, ...: (1, X, Y, Z),             ScalarImage (per declared map)
            subject_id:      str,
            file_id:         str,
        }

    ``target`` is the declared ``target_maps`` concatenated on the channel
    axis, in declaration order. Whether it is a *different* tensor from
    ``input`` is the arm's declared ``input_source`` (see
    :class:`QuantitativeConfigSchema`), checked against the index at
    construction rather than inferred per sample.

    Maps in Bloch-physical units (ms for T1/T2/T2*, unitless 0-1 for PD,
    mm²/s × 1e-3 for ADC) unless ``quantitative_config.units='normalized'``.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        quantitative_config: QuantitativeConfigSchema,
        transform: Callable | None = None,
    ):
        if not quantitative_config.enabled:
            raise ValueError("QuantitativeMapDataset requires quantitative_config.enabled=True.")
        if not quantitative_config.target_maps:
            raise ValueError(
                "QuantitativeMapDataset requires at least one entry in "
                "quantitative_config.target_maps (e.g. ['t1', 't2', 'pd'])."
            )
        self.index = index
        self.config = quantitative_config
        self.transform = transform
        self._check_input_source(index, list(quantitative_config.target_maps))

    def _check_input_source(self, index: list[dict[str, Any]], target_maps: list[str]) -> None:
        """Enforce the declared ``input_source`` against the actual index.

        The declaration is the arm's claim about what ``input_paths`` are; this
        is where the claim meets the manifest. Both directions are errors: a
        ``contrasts`` arm whose inputs ARE the maps would train the identity,
        and a ``maps`` arm whose inputs are something else is not modelling the
        map distribution it says it is.

        O(N) over the whole index, once per construction (so twice per run,
        train and val). Measured at ~180 ms per 10k records, ~1.7 s per 100k —
        one-off startup cost, but worth knowing because the BIDS globber path
        of ``build_quantitative_index`` puts no bound on N.
        """
        declared = self.config.input_source
        if declared is None:
            raise ValueError(
                "QuantitativeMapDataset requires quantitative.input_source to "
                "be declared ('contrasts' or 'maps'); got None. The schema "
                "validator normally catches this at load time."
            )
        want_self_paired = declared == "maps"
        offenders = [
            record.get("subject_id", f"record-{i}")
            for i, record in enumerate(index)
            if _input_is_the_maps(record, target_maps) is not want_self_paired
        ]
        if not offenders:
            return
        shown = offenders[:10]
        more = f" (+{len(offenders) - len(shown)} more)" if len(offenders) > len(shown) else ""
        if want_self_paired:
            detail = (
                "input_source='maps' claims every record's input_paths ARE the "
                f"target maps {target_maps}, but these records' input_paths are "
                "something else. Either the manifest is not a maps-style "
                "manifest, or the arm means input_source='contrasts'."
            )
        else:
            detail = (
                "input_source='contrasts' claims input_paths are weighted "
                "images the maps are predicted from, but these records' "
                f"input_paths are exactly the target maps {target_maps} -- so "
                "input and target would be the same tensor and the arm would "
                "train the identity. Declare input_source='maps' if that is a "
                "generative arm, or point input_paths at real contrasts."
            )
        raise ValueError(
            f"{len(offenders)} of {len(index)} quantitative records disagree "
            f"with the declared input_source. {detail}\n  "
            + "\n  ".join(str(o) for o in shown)
            + more
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tio.Subject:
        record = self.index[idx]

        # Load each input contrast as a separate volume, then stack to a
        # single multi-channel ScalarImage.
        input_paths = [Path(p) for p in record.get("input_paths", [])]
        if not input_paths:
            raise ValueError(f"Record {idx} has no 'input_paths'. Expected a non-empty list.")
        contrast_tensors = [_load_volume(p) for p in input_paths]
        # Each is (1, X, Y, Z); stack along channel dim → (C, X, Y, Z).
        input_tensor = torch.cat(contrast_tensors, dim=0)

        # Build the subject. ScalarImage is the right type for both inputs
        # and continuous-valued parameter maps; downstream Resample defaults
        # to bspline which is fine for intensity & parameter maps alike.
        kwargs: dict[str, Any] = {
            "input": tio.ScalarImage(tensor=input_tensor),
            "subject_id": record.get("subject_id", f"subj-{idx:06d}"),
            "file_id": record.get(
                "file_id", str(input_paths[0].stem) if input_paths else f"row-{idx}"
            ),
        }

        # Load each declared quantitative map.
        map_paths: dict[str, str] = record.get("map_paths", {})
        map_tensors: list[torch.Tensor] = []
        for map_name in self.config.target_maps:
            path_str = map_paths.get(map_name)
            if path_str is None:
                raise KeyError(
                    f"Quantitative map '{map_name}' is declared in "
                    f"quantitative.target_maps but record {idx} "
                    f"(subject={record.get('subject_id', '?')}) has no "
                    f"'map_paths.{map_name}'."
                )
            map_tensor = _load_volume(Path(path_str))
            if self.config.units == "normalized":
                map_tensor = _normalize_to_unit(map_tensor)
            kwargs[map_name] = tio.ScalarImage(tensor=map_tensor)
            map_tensors.append(map_tensor)

        # The canonical target: the declared maps stacked on the channel axis,
        # in ``target_maps`` order. Without this key every batch was rejected by
        # ``BatchAdapter.from_dict``, which requires both 'input' and 'target',
        # so ``dataset_type: quantitative`` could not serve a single step. The
        # per-map named keys stay: they are the documented contract and are read
        # by name (e.g. core/metrics/nr_cross.py).
        kwargs["target"] = tio.ScalarImage(tensor=torch.cat(map_tensors, dim=0))

        subject = tio.Subject(**kwargs)
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def dry_iter(self) -> list[tio.Subject]:
        """Lightweight Subject shells for TorchIO Queue init.

        Mirrors :meth:`UniversalMRIDataset.dry_iter` — TorchIO Queue
        introspects metadata without loading voxels.
        """
        stub = torch.zeros(1, 1, 1, 1)
        out: list[tio.Subject] = []
        for record in self.index:
            kwargs: dict[str, Any] = {
                "input": tio.ScalarImage(tensor=stub),
                "subject_id": record.get("subject_id", "unknown"),
                "file_id": record.get("file_id", "unknown"),
            }
            for map_name in self.config.target_maps:
                kwargs[map_name] = tio.ScalarImage(tensor=stub)
            # Key parity with __getitem__: the Queue sizes itself against these
            # shells, so a key missing here is a key the patch sampler never
            # learns to carry.
            kwargs["target"] = tio.ScalarImage(tensor=stub)
            out.append(tio.Subject(**kwargs))
        return out


def _normalize_to_unit(tensor: torch.Tensor) -> torch.Tensor:
    """Min-max scale a parameter map to [0, 1] for visualization mode.

    Robust to all-zero maps (returns the input unchanged).
    """
    flat = tensor.flatten()
    lo = flat.min()
    hi = flat.max()
    if (hi - lo).abs() < 1e-9:
        return tensor
    return (tensor - lo) / (hi - lo)


def load_quantitative_manifest(
    manifest_path: str | Path, target_maps: list[str]
) -> list[dict[str, Any]]:
    """Load a pre-built quantitative index from a committed-generator manifest.

    Non-BIDS quantitative corpora (e.g. the cluster NIST-MRF in-vivo parameter
    maps under ``Subj_*/MRF_7SLICE_5MM_<TYPE>_MAP_MASKED_*/MRF_<TYPE>.nii.gz``)
    are indexed by a deterministic generator
    (``scripts/data/build_nist_mrf_manifest.py``) rather than the BIDS globber,
    so the layout-specific scanning stays a committed SSOT producer and this
    loader only consumes its records. The manifest is a JSON list of records in
    the :class:`QuantitativeMapDataset` contract; we validate that every record
    carries non-empty ``input_paths`` and all requested ``target_maps`` so a
    silently-incomplete manifest fails loud (CLAUDE.md #9) rather than yielding
    half subjects.
    """
    import json

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Quantitative manifest not found: {manifest_path}. Generate it on "
            f"the cluster with scripts/data/build_nist_mrf_manifest.py."
        )
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(
            f"Quantitative manifest {manifest_path} must be a JSON list of "
            f"records, got {type(records).__name__}."
        )
    want = [m.lower() for m in target_maps]
    for i, rec in enumerate(records):
        if not rec.get("input_paths"):
            raise ValueError(
                f"Quantitative manifest record {i} ({rec.get('subject_id')!r}) "
                f"has empty 'input_paths'."
            )
        have = {k.lower() for k in (rec.get("map_paths") or {})}
        missing = [m for m in want if m not in have]
        if missing:
            raise ValueError(
                f"Quantitative manifest record {i} ({rec.get('subject_id')!r}) "
                f"is missing requested map(s) {missing}; has {sorted(have)}."
            )
    return records


def build_quantitative_index(
    data_root: str | Path,
    target_maps: list[str],
    input_contrast_suffixes: list[str] | None = None,
    map_suffixes: dict[str, str] | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Glob a BIDS-ish directory tree into a quantitative index.

    Looks for ``sub-*/anat/sub-*_{contrast}.nii.gz`` under ``data_root``.
    The default suffix mapping follows BIDS conventions:

    - Inputs: ``T1w``, ``T2w``, ``PDw`` (override via ``input_contrast_suffixes``)
    - Maps: ``T1map``, ``T2map``, ``T2starmap``, ``PDmap``, ``ADCmap``

    Args:
        data_root: Top-level directory containing ``sub-*`` subdirectories.
        target_maps: Names of maps to look for (e.g., ``['t1', 't2', 'pd']``).
        input_contrast_suffixes: BIDS contrast suffixes to glob as inputs.
        map_suffixes: Map name → BIDS-suffix override (e.g.,
            ``{'t1': 'T1map', 't2star': 'T2starmap'}``).
        manifest_path: If given, bypass the BIDS globber and load records from a
            committed-generator manifest (non-BIDS corpora — e.g. NIST-MRF).

    Returns:
        A list of records compatible with :class:`QuantitativeMapDataset`.
    """
    if manifest_path:
        return load_quantitative_manifest(manifest_path, target_maps)

    root = Path(data_root)
    input_contrast_suffixes = input_contrast_suffixes or ["T1w", "T2w"]
    default_map_suffixes = {
        "t1": "T1map",
        "t2": "T2map",
        "t2star": "T2starmap",
        "pd": "PDmap",
        "adc": "ADCmap",
        "fa_dti": "FAmap",
    }
    if map_suffixes:
        default_map_suffixes.update(map_suffixes)

    records: list[dict[str, Any]] = []
    for subj_dir in sorted(root.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        anat_dir = subj_dir / "anat"
        if not anat_dir.exists():
            anat_dir = subj_dir  # Flat layout fallback.
        subject_id = subj_dir.name

        # Find input contrasts.
        input_paths: list[str] = []
        for suffix in input_contrast_suffixes:
            matches = sorted(anat_dir.glob(f"{subject_id}*_{suffix}.nii*"))
            if matches:
                input_paths.append(str(matches[0]))

        # Find requested maps.
        map_paths: dict[str, str] = {}
        for map_name in target_maps:
            suffix = default_map_suffixes.get(map_name, map_name)
            matches = sorted(anat_dir.glob(f"{subject_id}*_{suffix}.nii*"))
            if matches:
                map_paths[map_name] = str(matches[0])

        # Only include records with at least one input AND all requested maps.
        if input_paths and len(map_paths) == len(target_maps):
            records.append(
                {
                    "subject_id": subject_id,
                    "file_id": subject_id,
                    "input_paths": input_paths,
                    "map_paths": map_paths,
                }
            )

    return records
