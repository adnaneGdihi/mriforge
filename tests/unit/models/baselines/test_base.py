"""Tests for the baseline adapter base contract.

Locks in ``TODO/backlog_baseline_replication_experiment_11.md`` Phase A.2.
The base class refuses to instantiate any concrete subclass that
leaves required provenance attributes at the sentinel value — this
is the CLAUDE.md pitfall #9 guard.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.baselines import BaselineAdapter, CoilHandling, FFTNorm


class _CompleteAdapter(BaselineAdapter):
    """Minimal complete adapter — all required overrides set."""

    REPO_NAME = "fake_baseline"
    PAPER_REF = "arxiv:0000.00000"
    PREFERRED_MASK_TYPE = "equispaced"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return x


def test_complete_subclass_instantiates() -> None:
    """A subclass that overrides every required attribute instantiates cleanly."""
    adapter = _CompleteAdapter()
    assert adapter.REPO_NAME == "fake_baseline"
    assert adapter.PREFERRED_FFT_NORM is FFTNorm.ORTHO


def test_subclass_missing_repo_name_raises_at_class_creation() -> None:
    """Subclass that forgets ``REPO_NAME`` fails loudly at class-creation time."""
    with pytest.raises(TypeError, match="REPO_NAME"):

        class _Broken(BaselineAdapter):
            PAPER_REF = "arxiv:0000.0"
            PREFERRED_MASK_TYPE = "equispaced"

            def forward(self, x, *args, **kwargs):
                return x


def test_subclass_missing_paper_ref_raises() -> None:
    with pytest.raises(TypeError, match="PAPER_REF"):

        class _Broken(BaselineAdapter):
            REPO_NAME = "fake"
            PREFERRED_MASK_TYPE = "equispaced"

            def forward(self, x, *args, **kwargs):
                return x


def test_subclass_missing_mask_type_raises() -> None:
    with pytest.raises(TypeError, match="PREFERRED_MASK_TYPE"):

        class _Broken(BaselineAdapter):
            REPO_NAME = "fake"
            PAPER_REF = "arxiv:0000.0"

            def forward(self, x, *args, **kwargs):
                return x


def test_provenance_returns_declared_attrs() -> None:
    """``provenance()`` exposes the declared class attributes for the run summary."""
    adapter = _CompleteAdapter()
    prov = adapter.provenance()
    assert prov["repo_name"] == "fake_baseline"
    assert prov["paper_ref"] == "arxiv:0000.00000"
    assert prov["preferred_mask_type"] == "equispaced"
    assert prov["preferred_fft_norm"] == "ortho"
    assert prov["coil_handling"] == "rss"


def test_forward_is_abstract() -> None:
    """An adapter that only sets attributes (no ``forward``) is still abstract."""
    with pytest.raises(TypeError):

        class _NoForward(BaselineAdapter):
            REPO_NAME = "fake"
            PAPER_REF = "arxiv:0000.0"
            PREFERRED_MASK_TYPE = "equispaced"
            # no forward override

        _NoForward()


def test_base_class_itself_is_not_directly_instantiable() -> None:
    """``BaselineAdapter`` is abstract (it has @abstractmethod ``forward``)."""
    with pytest.raises(TypeError):
        BaselineAdapter()
