r"""Defensible metadata stamping for every figure / table.

Per `TODO/report_step` Phase 0: each artifact records the git commit
hash, seed, dataset version, and timestamp so that any reviewer can
reproduce the exact number behind a chart.

Stamps are added in two places:

* As a one-line footer on every Matplotlib figure (small, low-key text in
  the bottom corner so it is not a visual element of the plot itself).
* As a sidecar JSON file written next to every output artifact
  (``<artifact>.meta.json``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.figure as mfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunMetadata:
    """Defensible per-artifact metadata stamp."""

    git_commit: str = "unknown"
    git_dirty: bool = False
    seed: int | None = None
    dataset_version: str | None = None
    config_path: str | None = None
    config_hash: str | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    framework_version: str = "mriforge"

    def to_dict(self) -> dict:
        return asdict(self)

    def short_label(self) -> str:
        """Compact one-line label for figure footers."""
        commit = self.git_commit[:7] if self.git_commit != "unknown" else "?"
        dirty = "*" if self.git_dirty else ""
        seed = f"seed={self.seed}" if self.seed is not None else ""
        ds = f"data={self.dataset_version}" if self.dataset_version else ""
        bits = [b for b in (f"git={commit}{dirty}", seed, ds) if b]
        return "  ·  ".join(bits)


def _git(args: list[str]) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git"] + args,
                cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def detect_metadata(
    *,
    seed: int | None = None,
    dataset_version: str | None = None,
    config_path: str | os.PathLike | None = None,
    config_hash: str | None = None,
) -> RunMetadata:
    """Detect git status and assemble a `RunMetadata` stamp.

    All inputs are optional — callers pass what they know.
    """
    commit = _git(["rev-parse", "HEAD"]) or "unknown"
    status = _git(["status", "--porcelain"])
    dirty = bool(status)
    return RunMetadata(
        git_commit=commit,
        git_dirty=dirty,
        seed=seed,
        dataset_version=dataset_version,
        config_path=str(config_path) if config_path is not None else None,
        config_hash=config_hash,
    )


def stamp_figure(fig: mfig.Figure, meta: RunMetadata) -> None:
    """Annotate a figure with the metadata footer."""
    fig.text(
        0.99,
        0.005,
        meta.short_label(),
        ha="right",
        va="bottom",
        fontsize=5,
        color="#888888",
    )


def write_sidecar(artifact_path: str | os.PathLike, meta: RunMetadata) -> Path:
    """Write ``<artifact>.meta.json`` alongside the artifact."""
    artifact_path = Path(artifact_path)
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as f:
        json.dump(meta.to_dict(), f, indent=2, sort_keys=True)
    return sidecar


#: Repo root, from ``src/mriforge/infrastructure/reporting/metadata.py``:
#: reporting -> infrastructure -> mriforge -> src -> ROOT. That is FIVE hops, i.e.
#: ``parents[4]``. It was ``parents[3]`` and resolved to ``src/`` -- correct
#: before the 2026-05 ``src -> src/mriforge`` refactor added a level, and silently
#: wrong after it, because the only consumer swallows the miss (#721): every
#: vendored baseline recorded ``upstream_commit: None``, so the provenance dict
#: could not identify which upstream revision produced the published numbers --
#: the one question it exists to answer.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _vendor_commit_sha(repo_name: str) -> str | None:
    """Return the submodule commit SHA for ``external/baselines/<repo_name>/``.

    Returns ``None`` if the vendored directory does not exist or is not a
    git repository — common during development before the submodule is
    added.
    """
    vendor = _REPO_ROOT / "external" / "baselines" / repo_name
    if not vendor.exists():
        # Legitimately absent (submodule not initialised). DEBUG names the path
        # probed, which is what would have made the `parents[3]` bug obvious.
        logger.debug("[baseline-provenance] no vendored dir at %s", vendor)
        return None
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=vendor,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as exc:
        # WARN, not silence. The directory EXISTS, so this means the vendored
        # baseline is present but unidentifiable. The bare `except Exception:
        # return None` this replaces is why the path bug above produced
        # `upstream_commit: None` on every baseline for months without anyone
        # noticing: "not vendored" and "vendored but unreadable" shared one
        # answer, and only one of them is benign.
        logger.warning(
            "[baseline-provenance] %s exists but `git rev-parse HEAD` failed "
            "(%s); upstream_commit will be null, so this run cannot name the "
            "upstream revision it replicates.",
            vendor,
            exc,
        )
        return None
    return sha or None


def baseline_provenance(adapter_cls: type) -> dict[str, str | None]:
    """Return reproducibility metadata for a baseline-replication run.

    Implements Phase A.4 of
    ``TODO/backlog_baseline_replication_experiment_11.md``. Each baseline
    training run must log enough metadata that a reviewer can identify
    the exact code path that produced the numbers:

    - This-repo commit hash (the adapter source-of-truth).
    - Vendored upstream commit hash (from
      ``external/baselines/<repo_name>/`` if present).
    - Adapter class declarations: paper reference, mask convention,
      FFT-norm convention, coil-handling strategy.

    Mask seeds are tracked separately by ``KSpaceMaskGenerator`` (each
    ``(subject_id, slice_idx, R)`` triple yields a deterministic seed)
    and are not part of this dict.

    Args:
        adapter_cls: Subclass of ``BaselineAdapter`` (or any class that
            exposes the declarative ``REPO_NAME``/``PAPER_REF``/etc. class
            attributes).

    Returns:
        Mapping with the keys ``repo_name``, ``paper_ref``,
        ``preferred_mask_type``, ``preferred_fft_norm``, ``coil_handling``,
        ``adapter_module``, ``adapter_qualname``, ``adapter_commit``
        (this repo), and ``upstream_commit`` (vendored sub-repo).
    """
    # Declarative attributes — the adapter class is required to set these
    # (BaselineAdapter enforces this via ``__init_subclass__``).
    repo_name = getattr(adapter_cls, "REPO_NAME", "<unknown>")
    paper_ref = getattr(adapter_cls, "PAPER_REF", "<unknown>")
    preferred_mask_type = getattr(adapter_cls, "PREFERRED_MASK_TYPE", "<unknown>")
    fft_norm_attr = getattr(adapter_cls, "PREFERRED_FFT_NORM", None)
    coil_handling_attr = getattr(adapter_cls, "COIL_HANDLING", None)

    return {
        "repo_name": repo_name,
        "paper_ref": paper_ref,
        "preferred_mask_type": preferred_mask_type,
        "preferred_fft_norm": (
            getattr(fft_norm_attr, "value", fft_norm_attr) if fft_norm_attr else None
        ),
        "coil_handling": (
            getattr(coil_handling_attr, "value", coil_handling_attr) if coil_handling_attr else None
        ),
        "adapter_module": adapter_cls.__module__,
        "adapter_qualname": adapter_cls.__qualname__,
        "adapter_commit": _git(["rev-parse", "HEAD"]),
        "upstream_commit": _vendor_commit_sha(repo_name),
    }
