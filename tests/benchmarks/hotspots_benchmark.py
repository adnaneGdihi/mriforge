import os
import sys
import tempfile
import time

import h5py
import numpy as np
import torch
import torchio as tio

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from spectramr.data.io_strategies import FastMRIH5Strategy
from spectramr.data.transforms.geometric import SmartGeometricStandardization


def benchmark_fastmri_h5_strategy():
    print("Benchmarking FastMRIH5Strategy...")

    # Create a dummy H5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp_path = f.name

    try:
        # Create a large-ish volume (e.g. 30 slices, 15 coils, 640x320)
        # Slices, Coils, Height, Width
        shape = (30, 15, 640, 320)
        with h5py.File(tmp_path, "w") as hf:
            hf.create_dataset(
                "kspace",
                data=np.random.randn(*shape).astype(np.float32)
                + 1j * np.random.randn(*shape).astype(np.float32),
            )
            hf.create_dataset(
                "reconstruction_rss",
                data=np.random.randn(30, 320, 320).astype(np.float32),
            )
            hf.attrs["ismrmrd_header"] = "dummy_header"

        strategy = FastMRIH5Strategy()

        # Benchmark loading full volume (current behavior)
        start_time = time.time()
        for _ in range(10):
            _ = strategy.load(tmp_path)
        end_time = time.time()
        print(f"  Load Full Volume (avg of 10): {(end_time - start_time) / 10:.4f} s")

        # Benchmark slice-specific loading (NEW FEATURE)
        start_time = time.time()
        for _ in range(100):
            _ = strategy.load(tmp_path, metadata={"slice_index": 15})
        end_time = time.time()
        print(f"  Strategy Slice Load (NEW): {(end_time - start_time) / 100:.4f} s")

        # Benchmark intended slice load (simulated by manually slicing for now to see potential)
        with h5py.File(tmp_path, "r") as hf:
            start_time = time.time()
            for _ in range(100):
                # Simulate what we WANT to do: read 1 slice
                _ = hf["kspace"][15]
            end_time = time.time()
        print(f"  Raw H5 Slice Read (Target): {(end_time - start_time) / 100:.4f} s")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def benchmark_smart_geometric_standardization():
    print("\nBenchmarking SmartGeometricStandardization...")

    # Create a dummy subject with complex-like data (stored as real with extra channel or just handle real for simplicity first,
    # but the transform handles key 'sensitivity_complex' specially)
    # Shape: (Coils, H, W, D) -> (15, 640, 400, 1) to simulate a 2D slice with coils, or (1, 640, 400, 30) for 3D?
    # The transform iterates images. Let's simulate a standard case.
    # Input: (1, 640, 372, 1) - typical FastMRI knee non-square

    dataset = []
    for _ in range(10):
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=torch.randn(1, 640, 372, 1)),
            target=tio.ScalarImage(tensor=torch.randn(1, 640, 372, 1)),
            sensitivity=tio.ScalarImage(
                tensor=torch.randn(15, 640, 372, 1)
            ),  # Multi-channel
        )
        dataset.append(subject)

    transform = SmartGeometricStandardization(target_shape=(320, 320))

    start_time = time.time()
    for subject in dataset:
        _ = transform(subject)
    end_time = time.time()

    print(f"  Transform 10 subjects: {(end_time - start_time):.4f} s")
    print(f"  Avg per subject: {(end_time - start_time) / 10:.4f} s")


if __name__ == "__main__":
    benchmark_fastmri_h5_strategy()
    benchmark_smart_geometric_standardization()
