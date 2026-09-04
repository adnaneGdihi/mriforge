"""Matplotlib renderers for the dataset EDA.

Each function writes exactly one PNG and returns its ``Path``. Pure presentation: takes
already-computed arrays/tensors, never reads data files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _save(fig, out_dir: Path, fname: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _mag(t: torch.Tensor) -> np.ndarray:
    return (t.abs() if t.is_complex() else t).detach().cpu().numpy()


# ── universal ──────────────────────────────────────────────────────────────────


def render_card(entry, profile: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    s = entry.source or {}
    banner = "" if entry.tier == "present" else f"   [{str(entry.status).upper()}]"
    lines = [
        f"{entry.dataset_id}{banner}",
        f"modality: {entry.modality}    tier: {entry.tier}    file_type: {entry.file_type}",
        f"role: {s.get('role', '?')}    field_T: {s.get('field_T', '?')}    anatomy: {s.get('anatomy', '?')}",
        f"records: {len(entry.records)}    status: {entry.status}",
        f"provides: {str(s.get('provides', ''))[:88]}",
    ]
    inv = (profile or {}).get("inventory", {})
    if inv:
        lines.append(
            f"profiled: {inv.get('n_records_profiled', '?')}/{inv.get('n_records_total', '?')}"
            f"   vendors: {inv.get('vendors', '?')}"
        )
    if s.get("note"):
        lines.append(f"note: {str(s['note'])[:88]}")
    for link in (s.get("links") or [])[:3]:
        lines.append(f"  • {link}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    color = {"present": "#1b7837", "absent": "#b2182b"}.get(entry.tier, "#777777")
    fig.patch.set_edgecolor(color)
    fig.patch.set_linewidth(4)
    return _save(fig, out_dir, "00_card.png")


def render_quality(quality: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(quality.keys())
    ax.bar(keys, [quality[k] for k in keys], color="#d6604d")
    ax.set_title("Data-quality flags")
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_dir, "01_quality.png")


def render_inventory(counts: dict[str, dict], out_dir: Path) -> Path:
    items = [(k, v) for k, v in counts.items() if v]
    n = max(len(items), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (title, mapping) in zip(axes, items, strict=False):
        ax.bar(list(map(str, mapping.keys())), list(mapping.values()), color="#4393c3")
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if not items:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "no inventory metadata", ha="center")
    fig.suptitle("Dataset composition")
    return _save(fig, out_dir, "02_inventory.png")


# ── image / recon ──────────────────────────────────────────────────────────────


def render_montage(images: list[torch.Tensor], out_dir: Path, fname: str, title: str) -> Path:
    n = min(len(images), 8) or 1
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    axes = np.atleast_1d(axes)
    for ax, im in zip(axes, images[:n], strict=False):
        ax.imshow(_mag(im), cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    return _save(fig, out_dir, fname)


def render_norm_voxel_dist(dist: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, vals) in zip(axes.ravel(), dist.items(), strict=False):
        ax.hist(np.asarray(vals).ravel(), bins=80, color="steelblue", log=True)
        ax.set_title(f"normalization: {name}")
        ax.set_xlabel("intensity")
        ax.set_ylabel("count (log)")
    fig.suptitle("Normalized voxel-intensity distribution (foreground)")
    fig.tight_layout()
    return _save(fig, out_dir, "20_norm_voxel_dist.png")


def render_percentile_table(perc: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    rows = [[k, f"{v:.4g}"] for k, v in perc.items()]
    ax.table(cellText=rows, colLabels=["percentile", "value"], loc="center")
    ax.set_title("Intensity percentiles")
    return _save(fig, out_dir, "23_percentile_table.png")


# ── raw k-space ─────────────────────────────────────────────────────────────────


def render_kspace_logmag(ksp: torch.Tensor, out_dir: Path) -> Path:
    k = ksp if ksp.ndim == 3 else ksp.unsqueeze(0)
    n = min(k.shape[0], 4)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, range(n), strict=False):
        ax.imshow(np.log1p(k[c].abs().detach().cpu().numpy()), cmap="viridis")
        ax.axis("off")
    fig.suptitle("k-space log-magnitude (per coil)")
    return _save(fig, out_dir, "41_kspace_logmag.png")


def render_radial_energy(radii, energy, cumulative, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(radii, np.asarray(energy) + 1e-12, color="navy", label="mean |k|²")
    ax2 = ax.twinx()
    ax2.plot(radii, cumulative, color="darkorange", label="cumulative")
    ax.set_xlabel("radial spatial frequency")
    ax.set_ylabel("energy")
    ax2.set_ylabel("cumulative fraction")
    ax.set_title("Radial k-space energy")
    return _save(fig, out_dir, "42_radial_energy.png")


def render_sampling_pattern(mask: np.ndarray, info: dict, out_dir: Path) -> Path:
    mask = np.asarray(mask)
    fig, ax = plt.subplots(figsize=(5, 4))
    # vmin/vmax pinned to [0, 1]: a constant mask (e.g. fully-sampled = all ones) would
    # otherwise auto-normalize to vmin==vmax and render solid black — indistinguishable
    # from an unsampled scan.
    ax.imshow(
        np.tile(mask[:, None], (1, max(len(mask) // 2, 8))),
        cmap="gray",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"Sampling: R≈{info['acceleration']:.2f}, ACS={info['acs_size']}, full={info['fully_sampled']}"
    )
    return _save(fig, out_dir, "43_sampling_pattern.png")


def render_coil_maps(csm: torch.Tensor, out_dir: Path) -> Path:
    n = min(csm.shape[0], 8)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, range(n), strict=False):
        ax.imshow(csm[c].abs().detach().cpu().numpy(), cmap="magma")
        ax.axis("off")
    fig.suptitle("ESPIRiT coil sensitivities")
    return _save(fig, out_dir, "44_coil_maps.png")


def render_phase(phase: np.ndarray, grad_mag: np.ndarray, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].imshow(phase, cmap="twilight")
    axes[0].set_title("phase")
    axes[0].axis("off")
    axes[1].hist(np.asarray(phase).ravel(), bins=80, color="purple")
    axes[1].set_title("phase histogram")
    axes[2].imshow(grad_mag, cmap="inferno")
    axes[2].set_title("|∇phase|")
    axes[2].axis("off")
    return _save(fig, out_dir, "47_phase.png")


# ── paired ──────────────────────────────────────────────────────────────────────


def render_paired_montage(pairs, out_dir: Path) -> Path:
    n = min(len(pairs), 4) or 1
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    axes = np.asarray(axes).reshape(2, -1)
    for j, (src, tgt) in enumerate(pairs[:n]):
        axes[0, j].imshow(_mag(src), cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(_mag(tgt), cmap="gray")
        axes[1, j].axis("off")
    axes[0, 0].set_title("source")
    axes[1, 0].set_title("target")
    fig.suptitle("Paired source / target")
    return _save(fig, out_dir, "30_paired_montage.png")


def render_paired_bland_altman(src: torch.Tensor, tgt: torch.Tensor, out_dir: Path) -> Path:
    a, b = _mag(src).ravel(), _mag(tgt).ravel()
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    mean, diff = (a + b) / 2, b - a
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, diff, s=2, alpha=0.2)
    ax.axhline(diff.mean(), color="red")
    ax.axhline(diff.mean() + 1.96 * diff.std(), color="red", ls="--")
    ax.axhline(diff.mean() - 1.96 * diff.std(), color="red", ls="--")
    ax.set_xlabel("mean")
    ax.set_ylabel("target - source")
    ax.set_title("Bland-Altman (source vs target)")
    return _save(fig, out_dir, "32_paired_bland_altman.png")


# ── quantitative map / trajectory ────────────────────────────────────────────────


def render_map_montage(maps: list[torch.Tensor], out_dir: Path, unit: str) -> Path:
    n = min(len(maps), 6) or 1
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    axes = np.atleast_1d(axes)
    for ax, mp in zip(axes, maps[:n], strict=False):
        im = ax.imshow(_mag(mp), cmap="viridis")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label=unit)
    fig.suptitle(f"Quantitative map ({unit})")
    return _save(fig, out_dir, "50_map_montage.png")


def render_temporal_strip(frames: list[torch.Tensor], out_dir: Path) -> Path:
    """The temporal suite: a row of up to 6 evenly-sampled frames + a temporal-std panel (where the
    frames share a shape) — the dedicated time-axis view for cine / 4D-flow / 5D datasets."""
    n = min(len(frames), 6) or 1
    shapes = {tuple(f.shape) for f in frames}
    have_std = len(shapes) == 1 and len(frames) > 1
    cols = n + (1 if have_std else 0)
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.6))
    axes = np.atleast_1d(axes)
    for ax, fr in zip(axes[:n], frames[:n], strict=False):
        ax.imshow(_mag(fr), cmap="gray")
        ax.axis("off")
    if have_std:
        sd = torch.stack([f.float() for f in frames]).std(dim=0)
        im = axes[n].imshow(_mag(sd), cmap="magma")
        axes[n].set_title("temporal std", fontsize=8)
        axes[n].axis("off")
        fig.colorbar(im, ax=axes[n], fraction=0.046)
    fig.suptitle("Temporal frames (evenly sampled across time) + temporal std")
    return _save(fig, out_dir, "61_temporal_strip.png")


def render_trajectory(kx: np.ndarray, ky: np.ndarray, t: np.ndarray, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(kx, ky, c=t, s=2, cmap="plasma")
    fig.colorbar(sc, ax=ax, label="time")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title("k-space trajectory")
    return _save(fig, out_dir, "60_trajectory.png")
