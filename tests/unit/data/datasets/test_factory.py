"""``data.datasets.factory.create_dataset`` split handling (cohort review 2026-09-02, T0.2).

The factory is dormant on the training path (``DataPipelineDirector`` builds
through ``DatasetInstantiator``), but it is reachable through
``datasets/api.py`` and it carried two silent-fallback shapes: a private
split boundary that disagreed with ``split_utils.split_index``, and a manifest
failure that degraded to an EMPTY index. Both are now one owner / loud.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.data.datasets import factory as factory_mod
from spectramr.data.split_utils import split_index
from tests.utils.data_config_stub import DataConfigStub


def _cfg(tmp_path, *, validation_split: float = 0.1) -> SimpleNamespace:
    return SimpleNamespace(
        data=DataConfigStub(
            index_path=str(tmp_path / "manifest.json"),
            validation_index_path=None,
            data_root=str(tmp_path),
            holdout_site=None,
            validation_split=validation_split,
            contrasts=None,
            target_contrasts=None,
        )
    )


class _CaptureDS:
    last_index: list | None = None

    def __init__(self, *, index, **_kw) -> None:
        _CaptureDS.last_index = list(index)


@pytest.fixture
def _patched(monkeypatch):
    import spectramr.data.datasets.universal_dataset as ud

    records = [{"primary_path": f"{i}.h5", "file_id": f"{i}"} for i in range(25)]
    monkeypatch.setattr(ud, "parse_fastmri_index", lambda **_kw: list(records))
    monkeypatch.setattr(ud, "UniversalMRIDataset", _CaptureDS)
    return records


def test_manifest_failure_raises_instead_of_an_empty_index(tmp_path, monkeypatch) -> None:
    """The planted violation: a manifest that cannot be read used to yield
    ``index = []`` under a warning -- a dataset that trains on nothing."""
    import spectramr.data.datasets.universal_dataset as ud

    def _boom(**_kw):
        raise OSError("manifest unreadable")

    monkeypatch.setattr(ud, "parse_fastmri_index", _boom)
    with pytest.raises(RuntimeError, match="Failed to load manifest"):
        factory_mod.create_dataset("kspace", _cfg(tmp_path), split="train")


def test_random_split_uses_the_shared_split_owner(tmp_path, _patched) -> None:
    """25 records at 0.1: the private ``int((1-f)*n)`` boundary gave 3
    validation items, ``split_index`` gives ``round(2.5) == 2``. One owner."""
    factory_mod.create_dataset("kspace", _cfg(tmp_path), split="val")
    val = _CaptureDS.last_index
    factory_mod.create_dataset("kspace", _cfg(tmp_path), split="train")
    train = _CaptureDS.last_index
    exp_train, exp_val = split_index(_patched, 0.1)
    assert val == exp_val and train == exp_train
    assert len(val) == 2


def test_unknown_split_raises(tmp_path, _patched) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        factory_mod.create_dataset("kspace", _cfg(tmp_path), split="test")
