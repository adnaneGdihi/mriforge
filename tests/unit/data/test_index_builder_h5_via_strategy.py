"""Regression test for D17 — audit 16 F3.

``src/data/metadata/index_builder.py:767`` used to open ``h5py.File``
directly, violating CLAUDE.md pitfall #11 (data-loading SSOT). The
read now goes through
``spectramr.data.io_strategies.FastMRIH5Strategy.load_torchio_tensor``,
which preserves the manifest reader's key-precedence behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def test_index_builder_does_not_import_h5py_directly() -> None:
    """The inline ``import h5py`` next to the manifest subject builder
    is the symptom we removed — the data-layer strategy owns h5py now."""
    src = (REPO / "src/spectramr/data/metadata/index_builder.py").read_text()
    builder_region = src.split("if fmt == \"h5\":")[1].split("elif fmt")[0]
    assert "import h5py" not in builder_region, (
        "src/spectramr/data/metadata/index_builder.py reintroduced a direct h5py "
        "import inside the manifest subject-builder. Route the read "
        "through spectramr.data.io_strategies.FastMRIH5Strategy.load_torchio_tensor "
        "instead — audit 16 F3 / D17 / CLAUDE.md pitfall #11."
    )


def test_index_builder_uses_canonical_strategy() -> None:
    src = (REPO / "src/spectramr/data/metadata/index_builder.py").read_text()
    assert "FastMRIH5Strategy.load_torchio_tensor" in src, (
        "index_builder no longer calls the canonical strategy helper. "
        "The manifest reader's H5 path must go through "
        "FastMRIH5Strategy.load_torchio_tensor — audit 16 F3 / D17."
    )


@pytest.mark.parametrize(
    "dataset_key, payload, expected_shape",
    [
        ("kspace", np.ones((4, 8), dtype=np.complex64), (2, 4, 8)),  # complex → stacked
        ("reconstruction_rss", np.ones((4, 8), dtype=np.float32), (1, 4, 8)),  # 2D → C1
        ("data", np.ones((3, 4, 8), dtype=np.float32), (3, 4, 8)),  # 3D → as-is
    ],
)
def test_load_torchio_tensor_key_precedence(
    tmp_path: Path,
    dataset_key: str,
    payload: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    """Each precedence key produces the right ``tio.ScalarImage``-shaped tensor."""
    import h5py

    from spectramr.data.io_strategies import FastMRIH5Strategy

    h5_path = tmp_path / f"{dataset_key}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(dataset_key, data=payload)

    tensor = FastMRIH5Strategy.load_torchio_tensor(h5_path)
    assert tuple(tensor.shape) == expected_shape


def test_load_torchio_tensor_falls_back_to_first_key(tmp_path: Path) -> None:
    """If none of the precedence keys match, the helper picks the first
    key in the file — matching the pre-migration behaviour."""
    import h5py

    from spectramr.data.io_strategies import FastMRIH5Strategy

    h5_path = tmp_path / "unknown.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("my_custom_array", data=np.zeros((5, 6), dtype=np.float32))

    tensor = FastMRIH5Strategy.load_torchio_tensor(h5_path)
    assert tuple(tensor.shape) == (1, 5, 6)


def test_load_torchio_tensor_kspace_wins_over_reconstruction_rss(
    tmp_path: Path,
) -> None:
    """When both ``kspace`` and ``reconstruction_rss`` exist, ``kspace``
    is chosen — matching the pre-migration key precedence."""
    import h5py

    from spectramr.data.io_strategies import FastMRIH5Strategy

    h5_path = tmp_path / "mixed.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("kspace", data=np.ones((2, 3), dtype=np.complex64))
        f.create_dataset(
            "reconstruction_rss", data=np.zeros((2, 3), dtype=np.float32)
        )

    tensor = FastMRIH5Strategy.load_torchio_tensor(h5_path)
    # complex stacked → 2 channels of real/imag
    assert tuple(tensor.shape) == (2, 2, 3)
