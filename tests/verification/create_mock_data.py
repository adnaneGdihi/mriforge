from pathlib import Path

import h5py
import numpy as np


def create_mock_data(root="data/processed/dummy_fastmri"):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating mock data in {root_path}...")

    # Create 5 mock files
    for i in range(5):
        fname = root_path / f"file_brain_AXT2_{i}.h5"

        # Dimensions: Slices=10, Coils=4, H=320, W=320
        # Wrapper expects complex as last dim if using torch view_as_real,
        # OR complex numpy array. Wrapper code handles numpy complex.

        num_slices = 10
        num_coils = 4
        h, w = 320, 320

        # Complex K-Space
        kspace = np.random.randn(num_slices, num_coils, h, w) + 1j * np.random.randn(
            num_slices, num_coils, h, w
        )
        kspace = kspace.astype(np.complex64)

        # Target (RSS)
        target = np.abs(kspace).sum(axis=1)  # Simple mock RSS

        with h5py.File(fname, "w") as f:
            f.create_dataset("kspace", data=kspace)
            f.create_dataset("reconstruction_rss", data=target)
            f.attrs["acquisition"] = "AXT2"

    print("Mock data created.")


if __name__ == "__main__":
    create_mock_data()
