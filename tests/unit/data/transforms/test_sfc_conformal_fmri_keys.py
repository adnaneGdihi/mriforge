"""Tests for the SFC/conformal/fMRI batch-key populators.

Targets ``spectramr.data.transforms.sfc_conformal_fmri_keys``. Focus: the
sample-invariant caching added by the wasted-compute audit (SFC-1/2/3). The
identity Jacobian, identity flatten grid, and constant GLM regressor depend only
on shape/config, so when a ``cache`` dict is supplied the populators must reuse
one instance per shape (no rebuild every call) while producing values identical
to the uncached path.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.sfc_conformal_fmri_keys import (
    attach_conformal_jacobian,
    attach_cortex_flatten_grid,
    attach_glm_design_matrix,
)


def _ref_batch() -> dict:
    return {"image": torch.randn(2, 1, 8, 8)}


def test_jacobian_values_match_with_and_without_cache() -> None:
    cache: dict = {}
    # These two tests asserted the fabricated identity (`torch.ones_like`) and
    # that it was cached. Both are gone with C13: an all-ones Jacobian weights
    # the conformal projection by 1, making it numerically the plain path — and
    # it SATISFIED the isinstance check that ConformalDiffusionReconStrategy
    # raises on precisely to stop that. Caching a facade only made it cheaper.
    with pytest.raises(ValueError, match="no real Jacobian"):
        attach_conformal_jacobian(_ref_batch(), cache=cache)


def test_a_supplied_jacobian_is_not_overwritten() -> None:
    """The no-op the wrapper's docstring always promised and no code did.

    The old body went straight to fabricating an identity, so an upstream
    dataset that DID compute a real Jacobian had it silently replaced.
    """
    real = torch.rand(2, 1, 8, 8)
    batch = {**_ref_batch(), "conformal_jacobian": real}
    assert attach_conformal_jacobian(batch)["conformal_jacobian"] is real


def test_the_override_is_honoured() -> None:
    """The documented escape. It has no producer anywhere, which IS the honest
    state of the mechanism — but the hook must work when someone writes one."""
    real = torch.rand(2, 1, 8, 8)
    batch = {**_ref_batch(), "_conformal_jacobian_override": real}
    assert attach_conformal_jacobian(batch)["conformal_jacobian"] is real


def test_the_message_names_both_ways_out() -> None:
    """Supply a Jacobian, or stop claiming the paradigm. Leaving only "it
    raised" would push a reader toward re-adding the identity."""
    with pytest.raises(ValueError) as exc:
        attach_conformal_jacobian(_ref_batch())
    message = str(exc.value)
    assert "_conformal_jacobian_override" in message
    assert "expose.conformal_jacobian: false" in message


def test_grid_cache_reuses_instance_and_matches_uncached() -> None:
    cache: dict = {}
    uncached = attach_cortex_flatten_grid(_ref_batch(), grid_height=4, grid_width=4)[
        "cortex_flatten_grid"
    ]
    first = attach_cortex_flatten_grid(
        _ref_batch(), grid_height=4, grid_width=4, cache=cache
    )["cortex_flatten_grid"]
    second = attach_cortex_flatten_grid(
        _ref_batch(), grid_height=4, grid_width=4, cache=cache
    )["cortex_flatten_grid"]
    assert first is second
    assert torch.equal(first, uncached)


def test_design_matrix_cache_reuses_instance_and_matches_uncached() -> None:
    cache: dict = {}
    uncached = attach_glm_design_matrix(_ref_batch(), n_timepoints=8)["design_matrix"]
    first = attach_glm_design_matrix(_ref_batch(), n_timepoints=8, cache=cache)[
        "design_matrix"
    ]
    second = attach_glm_design_matrix(_ref_batch(), n_timepoints=8, cache=cache)[
        "design_matrix"
    ]
    assert first is second
    assert first.shape == (2, 8)
    assert torch.equal(first, uncached)


def test_no_cache_builds_fresh_each_call() -> None:
    """Guards against a global cache. Moved off the Jacobian, which no longer
    BUILDS anything (C13) — `attach_cortex_flatten_grid` still does, so the
    concern is exercised where it still applies."""
    a = attach_cortex_flatten_grid(_ref_batch(), grid_height=4, grid_width=4)[
        "cortex_flatten_grid"
    ]
    b = attach_cortex_flatten_grid(_ref_batch(), grid_height=4, grid_width=4)[
        "cortex_flatten_grid"
    ]
    assert a is not b


# ---------------------------------------------------------------------------
# Commit 2 — populators resolve the reference tensor from input/target, not
# only 'image'. Recon datasets (m4raw/bart/SliceDataset) emit input/target and
# NO 'image' key, so gating on 'image' left the key unattached and the
# conformal-diffusion arm silently degraded to plain MSE (pitfall #16).
# ---------------------------------------------------------------------------


def _recon_batch() -> dict:
    """A reconstruction batch as the SSOT datasets actually emit it: input +
    target, no 'image' key."""
    return {"input": torch.randn(2, 1, 8, 8), "target": torch.randn(2, 1, 8, 8)}


def test_jacobian_is_not_fabricated_from_any_key() -> None:
    """This asserted the Jacobian was BUILT from `input` when `image` was
    absent — a real fix at the time (recon datasets emit input/target and no
    `image`, so gating on `image` left the key unattached).

    C13 removes the build entirely: whichever key it was sized from, the result
    was all-ones, and all-ones weights the conformal projection by 1. Sizing the
    facade correctly did not stop it being a facade. The reference-tensor
    resolution it protected still matters for the populators that genuinely
    build — see `test_cortex_grid_attached_from_input_key` and
    `test_design_matrix_attached_from_input_key`, which are unchanged.
    """
    with pytest.raises(ValueError, match="no real Jacobian"):
        attach_conformal_jacobian(_recon_batch())


def test_cortex_grid_attached_from_input_key() -> None:
    out = attach_cortex_flatten_grid(_recon_batch(), grid_height=4, grid_width=4)
    assert "cortex_flatten_grid" in out
    assert out["cortex_flatten_grid"].shape[0] == 2


def test_design_matrix_attached_from_input_key() -> None:
    out = attach_glm_design_matrix(_recon_batch(), n_timepoints=8)
    assert "design_matrix" in out
    assert out["design_matrix"].shape == (2, 8)


def test_image_key_still_preferred_when_present() -> None:
    """Key preference now only matters for the populators that build; the
    Jacobian raises regardless of which reference tensors are present."""
    batch = {"image": torch.randn(3, 1, 8, 8), "input": torch.randn(2, 1, 8, 8)}
    out = attach_cortex_flatten_grid(batch, grid_height=4, grid_width=4)
    assert out["cortex_flatten_grid"].shape[0] == 3


def test_no_ref_tensor_still_refuses() -> None:
    """Previously a silent no-op ("nothing to size"). It now raises like every
    other no-Jacobian case: an arm that asked to expose the key and gets nothing
    should hear about it, and the size was never the problem — the VALUE was.
    """
    with pytest.raises(ValueError, match="no real Jacobian"):
        attach_conformal_jacobian({"scanner": ["A", "B"]})


# ── WS3a: scanner/site IDs must be stable across processes (forkserver) ────────

import subprocess  # noqa: E402
import sys  # noqa: E402

from spectramr.data.transforms.sfc_conformal_fmri_keys import (  # noqa: E402
    _stable_id,
    attach_scanner_id,
    attach_site_id,
)


def test_stable_id_is_pinned_value() -> None:
    # Pins the exact (blake2b, digest_size=8, mod 1000) algorithm. A regression
    # back to the salted builtin ``hash()`` would not reliably reproduce these.
    assert _stable_id("Siemens") == 299
    assert _stable_id("GE") == 961


def test_stable_id_survives_pythonhashseed_change() -> None:
    """The load-bearing invariant: the same vendor string maps to the same id
    regardless of PYTHONHASHSEED, so forkserver workers (each a fresh
    interpreter with its own salt) agree. ``hash()`` would not."""
    prog = (
        "from spectramr.data.transforms.sfc_conformal_fmri_keys import _stable_id;"
        "print(_stable_id('Siemens'), _stable_id('Philips'))"
    )
    outs = set()
    for seed in ("0", "1", "424242"):
        env = {"PYTHONHASHSEED": seed, "PATH": __import__("os").environ["PATH"]}
        res = subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, env=env
        )
        assert res.returncode == 0, res.stderr
        outs.add(res.stdout.strip())
    assert len(outs) == 1, f"stable_id varied across PYTHONHASHSEED: {outs}"


def test_attach_scanner_id_maps_vendors_stably() -> None:
    batch = {"image": torch.randn(2, 1, 4, 4), "scanner": ["Siemens", "GE"]}
    out = attach_scanner_id(batch)
    assert out["scanner_id"].tolist() == [299, 961]


def test_attach_site_id_defaults_when_absent() -> None:
    batch = {"image": torch.randn(2, 1, 4, 4)}
    out = attach_site_id(batch, default_id=7)
    assert out["site_id"].tolist() == [7, 7]
