"""Multi-echo bundling transform (Phase 4b-extra).

For multi-echo MRI (T2 mapping, T2* mapping, ME-MRA), each scan
contributes one volume per echo time. There are two on-disk layouts:

- **Channel-stacked**: a single 4D NIfTI ``(H, W, slices, echoes)`` —
  echoes already collapsed into the last axis at acquisition time.
- **Multi-file**: one NIfTI per echo, named ``..._echo-1_..._mag.nii.gz``,
  ``..._echo-2_..._mag.nii.gz``, etc. (BIDS-conformant).

Phase 4b-extra's transform handles both cases: when the Subject's input
is multi-channel, it parses per-echo metadata (TE values as a list);
when the input is single-channel and there are sibling files, it loads
and stacks them.

Pair with :py:class:`LoadAcquisitionMetadata` — that transform reads
the per-echo TE list from BIDS sidecars (one ``EchoTime`` per echo) when
the source files declare them; this transform synchronizes the channel
axis with the metadata list.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torchio as tio

__all__ = ["BundleMultiEcho", "parse_echo_index_from_filename"]


_ECHO_RE = re.compile(r"echo[-_]?(\d+)", re.IGNORECASE)


def parse_echo_index_from_filename(path: str | Path) -> int | None:
    """Extract the echo index (1-based) from a BIDS-style filename.

    Examples:

    - ``sub-001_echo-1_T2w.nii.gz`` → 1
    - ``sub-001_echo-12_PD.nii.gz`` → 12
    - ``sub-001_T2w.nii.gz`` → None
    """
    name = Path(path).name
    match = _ECHO_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


class BundleMultiEcho(tio.Transform):
    """Synchronize the channel axis with per-echo metadata.

    When applied to a Subject whose ``input`` is already multi-channel
    (e.g., from a 4D NIfTI with echoes stacked at load time), this
    transform validates that ``subject["acquisition_metadata"]["TE"]``
    is a list of the same length as the channel count, and raises if
    they disagree (silent mismatch corrupts T2 fitting).

    When the input is single-channel AND ``subject["sibling_echo_paths"]``
    is a list of N paths, this transform loads each sibling, stacks them,
    and replaces the ``input`` ScalarImage with the multi-channel volume.

    Args:
        echo_axis_in: Axis on the on-disk file that holds the echo index.
            For 4D NIfTI with echoes stacked, typically 3 (last). For
            BIDS multi-file layout, ignored (each file is a 3D volume).
        validate_metadata: When True (default), raise if the metadata
            TE-list length doesn't match the channel count.
    """

    def __init__(
        self,
        echo_axis_in: int = 3,
        validate_metadata: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.echo_axis_in = echo_axis_in
        self.validate_metadata = validate_metadata

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Case 1: load + stack multiple echo files when paths are provided.
        sibling_paths = subject.get("sibling_echo_paths")
        if isinstance(sibling_paths, (list, tuple)) and len(sibling_paths) > 0:
            try:
                import nibabel as nib  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "BundleMultiEcho needs nibabel to stack echo files. "
                    "Install with `pip install nibabel`."
                ) from e
            volumes: list[torch.Tensor] = []
            for path in sibling_paths:
                vol = nib.load(str(path)).get_fdata()  # type: ignore[no-untyped-call]
                tensor = torch.as_tensor(vol, dtype=torch.float32)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                volumes.append(tensor)
            stacked = torch.cat(volumes, dim=0)  # (n_echoes, X, Y, Z)
            subject["input"] = tio.ScalarImage(tensor=stacked)
            subject["n_echoes"] = len(volumes)

        # Always record the channel count as n_echoes so downstream code
        # can rely on it without re-inspecting the tensor shape.
        input_img = subject.get("input")
        if input_img is None:
            return subject
        n_channels = input_img.data.shape[0]
        subject["n_echoes"] = n_channels

        if not self.validate_metadata:
            return subject

        # Validate per-echo metadata alignment.
        metadata = subject.get("acquisition_metadata")
        if isinstance(metadata, dict):
            te = metadata.get("TE")
            if isinstance(te, list) and len(te) != n_channels:
                raise ValueError(
                    f"BundleMultiEcho: input has {n_channels} channels but "
                    f"acquisition_metadata['TE'] has {len(te)} entries. "
                    "Silent mismatch corrupts T2 fitting — fix the source "
                    "or update the BIDS sidecar."
                )
        return subject
