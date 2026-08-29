"""UniversalMRIDataset must not silently drop k-space normalization.

``data.normalize_kspace`` is honoured by ``KSpaceNormalizationTransform`` (built
by ``TorchIOTransformBuilder``), not by the dataset or its subject builder --
both used to normalize as well, giving a double percentile divide and a double
log1p. See ``docs/kspace_normalization_ssot.rst``.
"""

from __future__ import annotations

import pytest

from mriforge.data.datasets.universal_dataset import UniversalMRIDataset


def test_normalize_kspace_without_a_transform_raises() -> None:
    """No silent skip: nothing would apply the requested normalization."""
    with pytest.raises(ValueError, match="normalize_kspace=True but no transform"):
        UniversalMRIDataset(index=[], normalize_kspace=True, transform=None)


def test_normalize_kspace_with_a_transform_is_accepted() -> None:
    """A transform is present to apply it, so construction proceeds."""
    ds = UniversalMRIDataset(
        index=[], normalize_kspace=True, transform=lambda subject: subject
    )
    assert ds.normalize_kspace is True


def test_no_normalization_requested_needs_no_transform() -> None:
    ds = UniversalMRIDataset(index=[], normalize_kspace=False, transform=None)
    assert ds.normalize_kspace is False


def test_subject_builder_no_longer_takes_normalization_params() -> None:
    """The builder matches and serves — it must not advertise a morph.

    Guards the regression path: re-adding these would resurrect the second
    normalizer that ran alongside the transform.
    """
    import inspect

    from mriforge.data.builders.torchio_subject_builder import FastMRISubjectBuilder

    params = inspect.signature(FastMRISubjectBuilder.__init__).parameters
    for gone in ("normalize_kspace", "kspace_percentile", "log_scaling"):
        assert gone not in params, f"{gone} must not be a subject-builder knob"
