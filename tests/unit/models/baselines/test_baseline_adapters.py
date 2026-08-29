"""Tests for the three baseline adapter skeletons (CDiffMR, Shen 2024, FDB).

Locks Phases B / C / D of
``TODO/backlog_baseline_replication_experiment_11.md``: each adapter
declares the contract attributes the registry-dispatcher and the
provenance manifest consume, and ``forward`` raises a clear
NotImplementedError until the upstream is vendored.
"""

from __future__ import annotations

import sys

import pytest
import torch

from mriforge.infrastructure.reporting.metadata import baseline_provenance
from mriforge.models.baselines import BaselineAdapter, CoilHandling, FFTNorm
from mriforge.models.baselines.cdiffmr import (
    _UPSTREAM_NETWORK_FILE,
    CDiffMRBaseline,
)
from mriforge.models.baselines.fdb import _FDB_SCRIPT_UTIL, FDBBaseline
from mriforge.models.baselines.shen2024 import Shen2024Baseline

# ---------------------------------------------------------------------------
# Vendored-upstream gates
# ---------------------------------------------------------------------------
#
# CDiffMR and FDB are git SUBMODULES (.gitmodules). A checkout that never ran
# `git submodule update --init` has the gitlink but no files, so every test
# below that touches upstream code fails on a missing import rather than
# skipping. That is what happened on cluster job 8004252: 9 red tests that said
# nothing about this repo.
#
# The gate reads the same SENTINEL FILE each adapter's own guard reads, so the
# skip cannot drift from the thing being guarded. Checking the DIRECTORY would
# not work: an uninitialised submodule leaves an empty directory that exists.
_VENDOR_HINT = (
    "vendored upstream absent — run `git submodule update --init --recursive`"
)
needs_cdiffmr = pytest.mark.skipif(
    not _UPSTREAM_NETWORK_FILE.exists(),
    reason=f"CDiffMR {_VENDOR_HINT}",
)
needs_fdb = pytest.mark.skipif(
    not _FDB_SCRIPT_UTIL.exists(),
    reason=f"FDB {_VENDOR_HINT}",
)


@pytest.mark.parametrize(
    "cls,expected_repo,expected_paper,expected_mask",
    [
        (CDiffMRBaseline, "cdiffmr", "Huang2023:CDiffMR", "gaussian_density"),
        (Shen2024Baseline, "shen2024", "Shen2024:CondDiffMRI", "gaussian_density"),
        (FDBBaseline, "fdb", "Karaoglu2024:FDB", "cartesian_peripheral"),
    ],
)
def test_adapter_declares_required_attributes(
    cls: type,
    expected_repo: str,
    expected_paper: str,
    expected_mask: str,
) -> None:
    """Each adapter sets the three required overrides."""
    assert cls.REPO_NAME == expected_repo
    assert cls.PAPER_REF == expected_paper
    assert cls.PREFERRED_MASK_TYPE == expected_mask


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_adapter_is_baseline_adapter_subclass(cls: type) -> None:
    """Each adapter subclasses the canonical base."""
    assert issubclass(cls, BaselineAdapter)


@pytest.mark.parametrize(
    "cls,expected_coil",
    [
        (CDiffMRBaseline, CoilHandling.RSS),
        (Shen2024Baseline, CoilHandling.RSS),
        (FDBBaseline, CoilHandling.MULTI_COIL_KSPACE),
    ],
)
def test_coil_handling_matches_paper(cls: type, expected_coil: CoilHandling) -> None:
    """Coil handling matches the paper's evaluation protocol."""
    assert cls.COIL_HANDLING is expected_coil


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_fft_norm_is_ortho(cls: type) -> None:
    """All three baselines use the repo's canonical ortho FFT."""
    assert cls.PREFERRED_FFT_NORM is FFTNorm.ORTHO


def test_shen2024_forward_still_raises_pending_upstream() -> None:
    """Shen 2024's upstream is still not vendored (DAS link TBD).

    The adapter loudly raises ``NotImplementedError`` rather than
    silently returning zeros — CLAUDE.md pitfall #9.
    """
    adapter = Shen2024Baseline()
    x = torch.zeros(1, 2, 16, 16, dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match="not yet wired"):
        adapter.forward(x)


@needs_cdiffmr
@needs_fdb
@pytest.mark.parametrize(
    "cls,resolution,in_ch",
    [
        (CDiffMRBaseline, 32, 1),
        (FDBBaseline, 64, 1),
    ],
)
def test_wired_adapter_complex_round_trip(
    cls: type, resolution: int, in_ch: int
) -> None:
    """CDiffMR and FDB run end-to-end on complex MRI input.

    The adapter coerces complex ``[B, C, H, W]`` to real ``[B, 2C, H, W]``,
    forwards through the upstream UNet, and coerces back. Shape and dtype
    must round-trip exactly — anything else means the channel-coercion shim
    drifted.
    """
    kwargs = {"in_channels": 2 * in_ch, "out_channels": 2 * in_ch}
    if cls is CDiffMRBaseline:
        kwargs["resolution"] = resolution
    else:
        kwargs["image_size"] = resolution
        kwargs["bridge_steps"] = 10
    adapter = cls(**kwargs)
    x = torch.randn(1, in_ch, resolution, resolution, dtype=torch.complex64)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.is_complex(y)


@needs_cdiffmr
@needs_fdb
@pytest.mark.parametrize(
    "cls,resolution",
    [(CDiffMRBaseline, 32), (FDBBaseline, 64)],
)
def test_wired_adapter_accepts_real_input(cls: type, resolution: int) -> None:
    """Real-valued input (upstream native layout) passes through unmolested."""
    if cls is CDiffMRBaseline:
        adapter = cls(resolution=resolution)
    else:
        adapter = cls(image_size=resolution, bridge_steps=10)
    x = torch.randn(1, 2, resolution, resolution)
    y = adapter(x)
    assert y.shape == x.shape
    assert not torch.is_complex(y)


@needs_cdiffmr
def test_cdiffmr_upstream_module_loaded_via_importlib() -> None:
    """CDiffMR upstream is loaded via ``importlib.util`` — not via sys.path pollution.

    Regression: previously we considered adding the CDiffMR root to
    ``sys.path``, which would also expose ``models`` and ``utils`` as
    top-level packages and shadow this repo's modules of the same name.
    Importlib loads only the single ``network_cdiff_unet2.py`` file.
    """
    import importlib
    importlib.invalidate_caches()
    # The sentinel module name must be present once the adapter loaded.
    from mriforge.models.baselines.cdiffmr import _load_upstream_model_class
    _load_upstream_model_class()
    assert "_cdiffmr_upstream_network" in sys.modules
    # The upstream's "models" and "utils" packages must NOT have leaked.
    if "models" in sys.modules:
        mod_file = getattr(sys.modules["models"], "__file__", "")
        assert "external/baselines/cdiffmr" not in str(mod_file), (
            "CDiffMR's top-level `models` package has polluted sys.modules"
        )


@needs_fdb
def test_fdb_upstream_path_injection_is_idempotent() -> None:
    """Repeated calls to ``_ensure_upstream_on_sys_path`` don't multiply entries."""
    from mriforge.models.baselines.fdb import _FDB_DIR, _ensure_upstream_on_sys_path

    _ensure_upstream_on_sys_path()
    n_before = sys.path.count(str(_FDB_DIR))
    _ensure_upstream_on_sys_path()
    _ensure_upstream_on_sys_path()
    assert sys.path.count(str(_FDB_DIR)) == n_before


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_provenance_dict_complete_for_each_adapter(cls: type) -> None:
    """``baseline_provenance`` emits a complete record for every adapter."""
    prov = baseline_provenance(cls)
    assert prov["repo_name"] == cls.REPO_NAME
    assert prov["paper_ref"] == cls.PAPER_REF
    assert prov["preferred_mask_type"] == cls.PREFERRED_MASK_TYPE
    assert prov["preferred_fft_norm"] == "ortho"


@needs_cdiffmr
def test_cdiffmr_constructor_accepts_paradigm_kwargs() -> None:
    """The constructor takes a sampler-step-count + channel knobs."""
    adapter = CDiffMRBaseline(
        in_channels=2, out_channels=2, resolution=32, num_steps=25,
    )
    assert adapter.num_steps == 25


@needs_fdb
def test_fdb_constructor_accepts_bridge_kwargs() -> None:
    """The FDB constructor accepts bridge-schedule hyperparameters."""
    adapter = FDBBaseline(image_size=64, bridge_steps=200, bridge_drift=0.5)
    assert adapter.bridge_steps == 200
    assert adapter.bridge_drift == pytest.approx(0.5)


@needs_fdb
def test_fdb_sample_loop_runs_end_to_end_on_synthetic_input() -> None:
    """The full bridge sampling loop returns a tensor of the requested shape.

    Without trained weights the output is noise — but the plumbing must
    work so a campaign can later drop in checkpoints. The test uses tiny
    image_size + 3 bridge steps to keep wall-clock under a second.
    """
    adapter = FDBBaseline(image_size=32, bridge_steps=3)
    kspace = torch.randn(1, 2, 32, 32)
    mask = torch.ones(1, 2, 32, 32)
    out = adapter.sample(kspace, mask, batch_size=1)
    assert out.shape == (1, 2, 32, 32)


@needs_cdiffmr
def test_cdiffmr_sample_raises_until_full_upstream_wrapper_lands() -> None:
    """CDiffMR full sampling loop is deferred — calling it must raise loudly."""
    adapter = CDiffMRBaseline(resolution=32)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(NotImplementedError, match="full sampling-loop wiring is deferred"):
        adapter.sample(x_start=x)


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("cdiffmr_baseline", CDiffMRBaseline),
        ("shen2024_baseline", Shen2024Baseline),
        ("fdb_baseline", FDBBaseline),
    ],
)
def test_baseline_name_resolves_to_real_adapter_not_stub(name, expected_cls) -> None:
    """Regression: each baseline name resolves to its real BaselineAdapter
    implementation, not the removed ``_BackboneAlias`` placeholder that used to
    live in ``_v6_1_registrations.py`` and double-registered the name (tripping
    the fail-loud collision guard during collection).
    """
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import get_model_class

    populate_model_registry()  # idempotent; must not raise a collision
    resolved = get_model_class(name)
    assert resolved is expected_cls
    assert issubclass(resolved, BaselineAdapter)


def test_repo_root_resolves_to_the_directory_holding_external_baselines() -> None:
    """The adapters' `_REPO_ROOT` must be the repo root, not `src/`.

    Regression for the `src -> src/mriforge` refactor (2026-05). These modules
    moved from `src/models/baselines/` to `src/mriforge/models/baselines/`, one
    level deeper, but kept `parents[3]` — which stopped being the repo root and
    became `src/`. The vendored upstreams then looked absent, and the adapters
    raised an error instructing the user to run a `git submodule add` they had
    already run, against a path (`src/external/baselines/...`) that has never
    been where the submodules live.

    Asserted against `pyproject.toml` rather than a parent count, so the next
    move of these files fails here with a clear reason instead of re-breaking
    the lookup.
    """
    from mriforge.models.baselines.cdiffmr import _CDIFFMR_DIR, _REPO_ROOT as CD_ROOT
    from mriforge.models.baselines.fdb import _FDB_DIR, _REPO_ROOT as FDB_ROOT

    for root in (CD_ROOT, FDB_ROOT):
        assert (root / "pyproject.toml").is_file(), (
            f"_REPO_ROOT={root} is not the repository root; "
            "the parents[] index is wrong for this file's depth"
        )

    for upstream in (_CDIFFMR_DIR, _FDB_DIR):
        assert upstream.parent.name == "baselines"
        assert upstream.parent.parent.name == "external"
        assert "src" not in upstream.parts, (
            f"{upstream} points inside the package; the vendored submodules live "
            "at <repo>/external/baselines/"
        )
