#!/usr/bin/env python3
"""
Cluster Data Inspection Tool
============================

Scans common cluster mount points to identify MRI datasets (FastMRI, M4Raw, etc.).
inspects file headers to determine validity and internal structure (keys, shapes).

Usage:
    python src/tools/inspect_cluster_data.py [--deep-scan]
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import h5py
import nibabel as nib

# Common cluster mount points
SEARCH_ROOTS = [
    str(Path.cwd() / "databases"),
]


def inspect_h5(path):
    """Inspect H5 file structure."""
    info = {"type": "h5", "keys": [], "shapes": {}}
    try:
        with h5py.File(path, "r") as f:
            info["keys"] = list(f.keys())
            for k in f.keys():
                if hasattr(f[k], "shape"):
                    info["shapes"][k] = f[k].shape
            info["attrs"] = dict(f.attrs)
    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_nifti(path):
    """Inspect NIfTI file headers."""
    info = {"type": "nifti"}
    try:
        img = nib.load(path)
        info["shape"] = img.shape
        info["affine"] = img.affine.tolist()
    except Exception as e:
        info["error"] = str(e)
    return info


def scan_directory(root, max_depth=4, sample_size=3):
    """Recursively scan directory for data clusters."""
    root = Path(root)
    if not root.exists():
        return {}

    dataset_candidates = defaultdict(list)

    print(f"Scanning {root}...")

    try:
        for root_dir, dirs, files in os.walk(root):
            # Depth check
            depth = Path(root_dir).relative_to(root).parts
            if len(depth) > max_depth:
                continue

            h5_files = [f for f in files if f.endswith(".h5")]
            nii_files = [f for f in files if f.endswith((".nii", ".nii.gz"))]
            pt_files = [f for f in files if f.endswith(".pt")]

            if h5_files:
                dataset_candidates[root_dir].append(("h5", h5_files))
            if nii_files:
                dataset_candidates[root_dir].append(("nifti", nii_files))
            if pt_files:
                dataset_candidates[root_dir].append(("torch", pt_files))

    except PermissionError:
        print(f"Permission denied: {root}")

    return dataset_candidates


def main():
    """main.

    Returns:
        Any: Description.
    """
    parser = argparse.ArgumentParser(description="Scan cluster for MRI datasets")
    parser.add_argument("--roots", nargs="+", help="Explicit roots to scan")
    parser.add_argument("--deep-scan", action="store_true", help="Scan deeper")
    args = parser.parse_args()

    roots = args.roots if args.roots else SEARCH_ROOTS

    print("=" * 60)
    print("CLUSTER DATA INSPECTION REPORT")
    print("=" * 60)
    print(f"Scanning roots: {roots}")

    found_any = False

    for root in roots:
        candidates = scan_directory(root, max_depth=5 if args.deep_scan else 3)

        for dir_path, types in candidates.items():
            found_any = True
            print(f"\n[FOUND] Candidate Dataset at: {dir_path}")

            for dtype, files in types:
                count = len(files)
                print(f"  - {dtype.upper()}: {count} files")

                # Inspect a sample
                if count > 0:
                    sample_file = Path(dir_path) / files[0]
                    print(f"    Inspecting sample: {files[0]}")

                    if dtype == "h5":
                        info = inspect_h5(sample_file)
                        if "error" in info:
                            print(f"      ERROR: {info['error']}")
                        else:
                            print(f"      Keys: {info['keys']}")
                            for k, s in info["shapes"].items():
                                print(f"      Shape[{k}]: {s}")

                            # Heuristic ID
                            if "kspace" in info["keys"]:
                                print("      -> Likely FastMRI or compatible raw data")
                            if "reconstruction_rss" in info["keys"]:
                                print("      -> Contains RSS target")

                    elif dtype == "nifti":
                        info = inspect_nifti(sample_file)
                        if "error" in info:
                            print(f"      ERROR: {info['error']}")
                        else:
                            print(f"      Shape: {info['shape']}")
                            print("      -> Volumetric/Slice data")

                    elif dtype == "torch":
                        print("      -> PyTorch tensor (likely sensitivity maps or checkpoints)")

    if not found_any:
        print("\n[WARNING] No MRI-like data found in scanned locations.")
        print("Suggest: Check mount points or ask admin for paths.")

    print("\nDone.")


if __name__ == "__main__":
    main()
