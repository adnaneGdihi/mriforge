"""The optional-backend probes must agree with what production actually does.

``requires_cuda_for_mamba`` exists to keep 68 Mamba tests from failing on a CPU
node with an opaque ``torch/_library`` dispatch traceback (job 8000966, 78
failures across 19 files). A skip marker is only as honest as its predicate, and
this one has a specific way to go wrong: it must fire on ``mamba_ssm`` present +
CUDA absent, and must NOT fire when ``mamba_ssm`` is absent -- that case takes
``MambaBlock``'s fail-loud ImportError path, which needs no GPU and is itself
under test in ``tests/unit/models/blocks/test_mamba_block.py``.

The tests below therefore pin the predicate against ``MambaBlock`` itself rather
than restating its import logic, so a change to the block's dispatch cannot
leave the marker quietly answering for the old one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.blocks.mamba_block import MambaBlock  # noqa: E402
from tests.utils.optional_backends import (  # noqa: E402
    HAS_MAMBA_SSM,
    HAS_TORCH_FIDELITY,
    requires_cuda_for_mamba,
    requires_torch_fidelity,
)


@pytest.mark.unit
def test_mamba_probe_agrees_with_what_the_block_dispatches_to(monkeypatch) -> None:
    """``HAS_MAMBA_SSM`` must predict ``MambaBlock.use_official``.

    This is the only assertion that can catch drift: the probe duplicates the
    block's two-step import, and a duplicated decision is one that can disagree.
    Constructing the block is the arbiter.
    """
    monkeypatch.delenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", raising=False)

    if not HAS_MAMBA_SSM:
        # No kernel -> fail loud, no CUDA involved. The marker must stay quiet.
        with pytest.raises(ImportError, match="mamba_ssm"):
            MambaBlock(d_model=8, d_state=4)
        return

    block = MambaBlock(d_model=8, d_state=4)
    assert block.use_official is True


@pytest.mark.unit
def test_marker_fires_on_exactly_the_kernel_present_no_cuda_pairing() -> None:
    """Neither half alone is a reason to skip.

    Kernel absent -> ImportError path, runs anywhere. CUDA present -> the real
    kernel runs. Only the pairing is unrunnable, and that pairing is the CPU
    test cluster.
    """
    expected = HAS_MAMBA_SSM and not torch.cuda.is_available()

    assert requires_cuda_for_mamba.args[0] is expected
    assert not (expected and torch.cuda.is_available())


@pytest.mark.unit
def test_marker_reason_names_the_condition() -> None:
    """A bare "skipped" in a 45k-test report is unactionable."""
    reason = requires_cuda_for_mamba.kwargs["reason"]

    assert "mamba_ssm" in reason
    assert "CUDA" in reason


@pytest.mark.unit
def test_torch_fidelity_probe_matches_the_import() -> None:
    """The sibling probe, pinned the same way."""
    try:
        import torch_fidelity  # noqa: F401

        installed = True
    except ImportError:
        installed = False

    assert HAS_TORCH_FIDELITY is installed
    assert requires_torch_fidelity.args[0] is not installed
