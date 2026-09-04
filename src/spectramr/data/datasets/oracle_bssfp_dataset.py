"""oracle_bssfp dataset — phase-cycled bSSFP stack + real Hz B0 (vf_29 Path A).

The real-data counterpart to the synthesised ``bssfp_banding`` Path B: instead of
synthesising the phase-cycled stack from a clean image + a real B0, this loads a
*measured* phase-cycled bSSFP stack and an analytically-known off-resonance map
(Plaehn et al., MRM 2025 — ``oracle_bssfp``). The model recovers ΔB0 from the
real banding and is graded against the real B0 (external validity).

**Extraction prerequisite.** The raw ``oracle_bssfp`` is cluster-only Siemens
TWIX ``.dat``. A TWIX reader now exists in-repo (``spectramr.data.twix`` /
``TwixStrategy``) and can decode the raw VD/VE k-space, but *this* loader still
consumes the EXTRACTED image form: the phase-cycled stack as a real-interleaved
NIfTI ``[H, W, 2N]`` (N real + N imag channels, the model's 2C-real = C-complex
convention) and the B0 map as a Hz NIfTI ``[H, W]`` — both read through the SSOT
``NiftiStrategy`` (the analytical Hz B0 has no TWIX source; grading is in image
space). The cluster TWIX→NIfTI conversion (which can use the new reader) remains a
prerequisite. It RAISES if the data / extraction is absent or the stack channel
axis is not ``2N`` (no speculative no-op — pitfall #15).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.data.io_strategies import NiftiStrategy

__all__ = ["OracleBssfpDataset", "build_oracle_bssfp_index"]


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else root / pp


def build_oracle_bssfp_index(
    data_root: str | Path,
    stack_glob: str,
    b0_map: str,
) -> list[dict[str, str]]:
    """Pair each phase-cycled-stack NIfTI (``stack_glob``) with the shared B0 map.

    Returns records ``{'stack', 'b0_map'}``. Raises if ``data_root`` is missing,
    the B0 map is absent, or the glob matches no stack (no silent fallback — the
    field reference must exist for the run to be falsifiable).
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"oracle_bssfp data_root does not exist: {root}")
    b0 = _resolve(root, b0_map)
    if not b0.is_file():
        raise FileNotFoundError(f"oracle_bssfp b0_map not found: {b0}")
    stacks = sorted(root.glob(stack_glob))
    if not stacks:
        raise FileNotFoundError(
            f"oracle_bssfp found no phase-cycled stacks under {root}/{stack_glob} "
            "(extract the TWIX bSSFP to a real-interleaved [H,W,2N] NIfTI first)."
        )
    return [{"stack": str(s), "b0_map": str(b0)} for s in stacks]


class OracleBssfpDataset(Dataset):
    """Real phase-cycled bSSFP stack → ``tio.Subject`` (input = stack, + real b0_map)."""

    def __init__(
        self,
        index: list[dict[str, str]],
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        self.index = list(index)
        self.transform = transform
        self._reader = NiftiStrategy()

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[tio.Subject]:
        """Lightweight ``Subject`` shells for TorchIO ``Queue`` compatibility.

        ``tio.Queue.iterations_per_epoch`` calls ``self.subjects_dataset.dry_iter()``
        and only needs an *indexable* ``Sequence[Subject]`` of the right length — it
        counts subjects and reads each subject's optional ``num_samples`` attribute,
        never the voxel data. So we return one 1x1x1x1-stub ``Subject`` per index
        record (no NIfTI read, no decode). Mirrors :meth:`BartKspaceDataset.dry_iter`
        and the rest of the dataset family.

        Without this method an oracle_bssfp arm wrapped in a ``tio.Queue`` crashes at
        the first training step with ``'OracleBssfpDataset' object has no attribute
        'dry_iter'`` (cluster VF diagnostics 2026-06-26: exp_vf_29 Path-A oracle arms).
        """
        stub = torch.zeros(1, 1, 1, 1)
        subjects: list[tio.Subject] = []
        for rec in self.index:
            stack = str(rec.get("stack", "unknown"))
            subjects.append(
                tio.Subject(
                    input=tio.ScalarImage(tensor=stub),
                    stack=stack,
                    file_id=Path(stack).stem,
                )
            )
        return subjects

    def __getitem__(self, idx: int) -> tio.Subject:
        rec = self.index[idx]
        stack = self._load_stack(str(rec["stack"]))  # [2N, H, W] real-interleaved
        b0 = self._load_map(str(rec["b0_map"]))  # [1, H, W] Hz
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=stack.unsqueeze(-1)),
            target=tio.ScalarImage(tensor=stack.clone().unsqueeze(-1)),
            b0_map=tio.ScalarImage(tensor=b0.unsqueeze(-1)),
        )
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _raw(self, path: str) -> torch.Tensor:
        data: Any = self._reader.load(path)["data"]
        t = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        return t.float().squeeze()

    def _load_stack(self, path: str) -> torch.Tensor:
        """NIfTI ``[H, W, 2N]`` (channels last) → ``[2N, H, W]`` real-interleaved."""
        t = self._raw(path)
        if t.dim() != 3 or t.shape[-1] % 2 != 0:
            raise ValueError(
                f"oracle_bssfp stack must be a real-interleaved [H, W, 2N] NIfTI "
                f"(even channel axis = N real + N imag), got shape {tuple(t.shape)}."
            )
        return t.permute(2, 0, 1).contiguous()  # [2N, H, W]

    def _load_map(self, path: str) -> torch.Tensor:
        """NIfTI ``[H, W]`` Hz B0 → ``[1, H, W]``."""
        t = self._raw(path)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t
