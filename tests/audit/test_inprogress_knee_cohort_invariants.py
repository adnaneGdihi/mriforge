"""Dataset-wiring invariants for the inprogress FastMRI knee cohort.

Pins the contract from the 2026-06-11 knee dataset-wiring pass
(docs/superpowers/specs/2026-06-11-inprogress-knee-dataset-wiring-design.md).
Every inprogress arm whose data block targets FastMRI knee must:

* declare ``dataset_type: fastmri_kspace`` -- the explicit, self-documenting form.
  (Functionally ``kspace`` and ``fastmri_kspace`` both normalize to ``"kspace"``
  and route to ``fastmri_h5`` via ``factory.py:56,200`` / the ``data.py:2558``
  validator, so this is a clarity/consistency convention matching the gold arm
  ``spec_tasks/task_2_1_recon_hssc.yaml`` -- NOT a reader fix.)
* reference the **SSOT manifest** via ``index_path`` -- the canonical v2-manifest
  path whose embedded ``data_root`` is authoritative (``manifest_loader.py:145``),
  rather than relying on an on-the-fly ``data_root`` directory scan.
* keep coil-mode <-> ``in_channels`` consistent: ``svd`` + ``num_virtual_coils=N``
  yields ``in_channels = 2N`` (complex), ``rss`` yields a single magnitude channel.
  The pre-fix arms declared ``rss`` with ``in_channels: 2`` -- a latent shape
  mismatch only ``--probe`` / real training would catch.
* crop within the real 640x372 (single) / 640x368 (multi) FOV.

None of these is a schema error, so ``from_yaml`` / the audit ladder accept the
pre-fix state; only this contract test catches it (and pins it against drift).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.corpus_keys import read_key

_REPO = Path(__file__).resolve().parents[2]
_INPROGRESS = _REPO / "experiments" / "inprogress"


def _load(p: Path) -> dict[str, Any]:
    with p.open() as fh:
        doc = yaml.safe_load(fh)
    return doc if isinstance(doc, dict) else {}


def _is_knee(cfg: dict[str, Any]) -> bool:
    root = read_key(cfg, "data.source.root", default="")
    index = read_key(cfg, "data.source.index_path", default="")
    blob = f"{root} {index}".lower()
    return "knee" in blob and "ulf" not in blob


def _discover() -> list[Path]:
    arms: list[Path] = []
    for p in tracked_yamls(_INPROGRESS):
        try:
            cfg = _load(p)
        except Exception:
            continue
        if _is_knee(cfg):
            arms.append(p)
    return arms


_ARMS = _discover()
_IDS = [str(p.relative_to(_INPROGRESS)) for p in _ARMS]


def test_the_knee_cohort_sweep_is_non_empty() -> None:
    """An empty `_ARMS` makes every parametrized test below vanish silently.

    `_is_knee` matches on `data.source.root` / `data.source.index_path`, both of
    which are renamed keys. When the in-progress key drain rewrote the corpus,
    a lookup on the retired spelling returned "" for every arm and this whole
    file would have gone green while testing nothing -- the same vacuous-sweep
    failure the mrixfields cap guard hit. The reader follows RENAMES now, and
    this pins that it still finds the cohort.
    """
    assert _ARMS, (
        "no knee arms discovered under experiments/inprogress -- `_is_knee` "
        "matches on renamed keys, so this is far more likely a stale lookup "
        "than an empty cohort"
    )


def _in_channels(model: dict[str, Any]) -> Any:
    if model.get("in_channels") is not None:
        return model["in_channels"]
    return (model.get("model_kwargs") or {}).get("in_channels")


@pytest.fixture(params=_ARMS, ids=_IDS)
def cfg(request: pytest.FixtureRequest) -> dict[str, Any]:
    return _load(request.param)


def test_knee_arms_discovered() -> None:
    """Guard against an empty parametrization silently passing every invariant."""
    assert len(_ARMS) >= 28, f"expected the inprogress knee cohort, found {len(_ARMS)}"


def test_declares_explicit_fastmri_type(cfg: dict[str, Any]) -> None:
    dt = (cfg.get("data") or {}).get("dataset_type")
    assert dt == "fastmri_kspace", (
        f"knee arms standardize on the explicit, self-documenting dataset_type "
        f"'fastmri_kspace' (matches the gold arm); got dataset_type={dt!r}. "
        "Functionally equivalent to 'kspace' but unambiguous about the source."
    )


def test_references_knee_manifest(cfg: dict[str, Any]) -> None:
    ip = read_key(cfg, "data.source.index_path", default="") or ""
    assert "fastmri_knee" in ip and ip.endswith("coil.json"), (
        f"knee arm must point at the SSOT manifest via index_path; got {ip!r}"
    )


def test_coil_channel_consistency(cfg: dict[str, Any]) -> None:
    mode = read_key(cfg, "data.coils.processing_mode")
    in_ch = _in_channels(cfg.get("model") or {})
    assert in_ch is not None, "model.in_channels must be declared for a knee arm"
    if mode == "svd":
        nvc = read_key(cfg, "data.coils.num_virtual_coils")
        assert nvc is not None and in_ch == 2 * nvc, (
            f"svd coil mode -> in_channels must equal 2*num_virtual_coils "
            f"(got in_channels={in_ch}, num_virtual_coils={nvc})"
        )
    elif mode == "rss":
        assert (
            in_ch == 1
        ), f"rss coil mode -> magnitude single channel; got in_channels={in_ch}"
    else:
        pytest.fail(f"unexpected coil_processing_mode={mode!r} for a knee arm")


def test_patch_within_fov(cfg: dict[str, Any]) -> None:
    patch = read_key(cfg, "data.sampling.patch_size")
    assert patch and len(patch) >= 2, f"patch_size must be set; got {patch!r}"
    h, w = patch[0], patch[1]
    assert (
        h <= 640 and w <= 640
    ), f"knee patch {h}x{w} exceeds the real FOV (640x372 single / 640x368 multi)"
