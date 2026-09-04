"""K-space loading is decided by the IO strategy, never by a dataset flag (#918).

``load_kspace`` was derived from ``data.dataset_type`` in
``DatasetInstantiator``, threaded through two datasets, stored on both, and read
by neither. Deleting it is only safe if k-space loading never depended on it,
and that is the property these tests pin -- separately from the signature check,
because a signature assertion alone would still pass on a build that had quietly
stopped loading k-space at all.

The two halves are deliberately independent:

* ``TestKSpaceStillLoads`` exercises the real reader against a real HDF5 file.
* ``TestTheInertKnobIsGone`` pins the signatures, and pins that
  ``load_sensitivity`` -- derived from the *same* expression, and the reason
  this could not be a blanket delete -- is still honoured.
"""

import inspect

import numpy as np
import pytest

from spectramr.data.datasets.contrast_aware import ContrastAwarePairedDataset
from spectramr.data.datasets.universal_dataset import UniversalMRIDataset
from spectramr.data.io_strategies import IOStrategyFactory


class TestKSpaceStillLoads:
    """The capability, against a real file -- not a mock and not a signature."""

    @staticmethod
    def _fastmri_h5(tmp_path, coils=4, slices=2, h=8, w=8):
        """A multi-coil fastMRI-shaped file: kspace (S, C, H, W) complex."""
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "sample.h5"
        rng = np.random.default_rng(0)
        ks = (rng.standard_normal((slices, coils, h, w))
              + 1j * rng.standard_normal((slices, coils, h, w))).astype(np.complex64)
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=ks)
            f.create_dataset(
                "reconstruction_rss",
                data=np.abs(ks).sum(axis=1).astype(np.float32),
            )
        return path

    def test_multicoil_kspace_is_returned(self, tmp_path):
        """No flag is passed anywhere here -- the reader decides, and it loads."""
        path = self._fastmri_h5(tmp_path)
        result = IOStrategyFactory.get("fastmri_h5").load(str(path))

        assert "kspace" in result, (
            "the fastMRI reader stopped returning k-space -- removing the inert "
            "load_kspace flag must not touch the reader that actually loads it"
        )
        ks = result["kspace"]
        assert ks is not None
        assert ks.is_complex(), f"k-space must stay complex, got {ks.dtype}"
        assert ks.shape[-2:] == (8, 8), ks.shape

    def test_a_file_without_kspace_is_still_readable(self, tmp_path):
        """The reader keys on the FILE's contents. That is the real gate."""
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "image_only.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "reconstruction_rss", data=np.zeros((2, 8, 8), dtype=np.float32)
            )

        result = IOStrategyFactory.get("fastmri_h5").load(str(path))
        assert result.get("kspace") is None
        assert "data" in result


class TestTheInertKnobIsGone:
    """`load_kspace` is not advertised; `load_sensitivity` still is, and works."""

    @pytest.mark.parametrize(
        "cls", [UniversalMRIDataset, ContrastAwarePairedDataset], ids=lambda c: c.__name__
    )
    def test_load_kspace_is_not_a_parameter(self, cls):
        params = inspect.signature(cls.__init__).parameters
        assert "load_kspace" not in params, (
            f"{cls.__name__} still advertises load_kspace. It was stored and "
            "never read -- a config-derived value delivered to a consumer that "
            "ignores it (pitfall #15)"
        )

    def test_load_sensitivity_survived_and_still_gates_the_io(self):
        """It came off the SAME expression, so it is the one that had to stay.

        `UniversalMRIDataset.__init__` reads the bare local at
        ``sensitivity_io=self.io if load_sensitivity else None`` -- a real
        decision, unlike its deleted sibling.
        """
        assert "load_sensitivity" in inspect.signature(UniversalMRIDataset.__init__).parameters

        on = UniversalMRIDataset(index=[], io_strategy="fastmri_h5", load_sensitivity=True)
        off = UniversalMRIDataset(index=[], io_strategy="fastmri_h5", load_sensitivity=False)

        assert on.subject_builder.sensitivity_io is not None
        assert off.subject_builder.sensitivity_io is None, (
            "load_sensitivity stopped gating sensitivity_io -- it must not have "
            "been swept up with load_kspace"
        )
