"""Regression test: the startup data-root pre-flight resolves paths like the loader.

``_validate_data_availability_at_startup`` used to raw-check
``Path(config.data.data_root).exists()`` (relative to CWD) plus an ad-hoc
``SPECTRAMR_DATA_ROOT/data_root`` join with a ``./databases`` default. That diverged
from the dataset builder, which resolves a manifest's embedded ``data_root`` via
``PathResolver`` (``src/spectramr/data/builders/manifest_index.py``): a config whose
data the loader would locate (under ``PROJECT_ROOT`` / ``SPECTRAMR_DATA_ROOT``) could
still fail the pre-flight with "Data root not found". The pre-flight now routes
through ``PathResolver.resolve`` so the two agree. These tests exercise the pure
path logic with a fake config (no DI, no data load).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.bootstrap import _validate_data_availability_at_startup
from tests.utils.data_config_stub import DataConfigStub


def _config(data_root: str) -> SimpleNamespace:
    # strategy_class must be set (the pre-flight raises otherwise, before data check);
    # a non-synthetic dataset_type so requires_data is True.
    return SimpleNamespace(
        training=SimpleNamespace(strategy_class="ReconstructionStrategy", device=None),
        # bootstrap.py:117 reads `data_config.source.root`; a flat `data_root`
        # here predates the decomposition and the reader never saw it. Routed
        # through the shared stub so the sub-block shape comes from the REAL
        # schema rather than a second, hand-kept copy of it.
        data=DataConfigStub(data_root=data_root, dataset_type="kspace"),
    )


def test_resolves_relative_data_root_under_project_root(tmp_path, monkeypatch):
    """A relative data_root that exists under PROJECT_ROOT (not CWD) now passes —
    the old raw Path.exists() (CWD-relative) would have raised."""
    (tmp_path / "site_data").mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    # must not raise
    _validate_data_availability_at_startup(_config("site_data"))


def test_honors_spectramr_data_root_env(tmp_path, monkeypatch):
    (tmp_path / "site_data").mkdir()
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.setenv("SPECTRAMR_DATA_ROOT", str(tmp_path))
    _validate_data_availability_at_startup(_config("site_data"))


def test_still_fails_loud_when_genuinely_absent(tmp_path, monkeypatch):
    """A data_root that resolves nowhere still raises (fail-loud preserved), and the
    message reports the resolved candidate for debuggability."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))  # empty root
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match=r"Data root not found.*resolved:"):
        _validate_data_availability_at_startup(_config("definitely_absent_xyz"))
