"""``direct_ulf_to_hf_sr`` must share stage1_vae_hf's data, differing only in direction.

The arm is the single-stage ULF->HF super-resolution counterpart to the
two-stage LDM: same per-slice 2D data pipeline as ``stage1_vae_hf`` (v6 manifest,
full-slice sampling, minmax[-1,1], multi-contrast), but reframed as a resolution
task — ULF input, HF target. This test pins that contract: the ``data`` block is
identical to stage1_vae_hf EXCEPT ``bidirectional_mode`` (hf_to_hf -> ulf_to_hf),
and the arm uses a reconstruction model/strategy rather than the VAE.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

_DIR = pathlib.Path("experiments/inprogress/ldm_two_stage_ulf_to_hf")
_SR = _DIR / "direct_ulf_to_hf_sr.yaml"
_VAE = _DIR / "stage1_vae_hf.yaml"


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_sr_arm_exists() -> None:
    assert _SR.exists()


def test_same_data_except_direction_and_dropped_facade() -> None:
    """The SR arm shares stage1_vae_hf's data pipeline EXCEPT two intentional
    changes: (1) the direction flip (hf_to_hf → ulf_to_hf), and (2) the inert
    ``multi_contrast`` FiLM block is dropped — it is never read on the
    ``nifti_paired`` loader path and ``mamba_unet`` takes no contrast embedding
    (#16 facade). Every other data key must match verbatim."""
    from tests.utils.corpus_keys import read_key

    sr_doc, vae_doc = _load(_SR), _load(_VAE)
    sr, vae = sr_doc["data"], vae_doc["data"]
    differing = [k for k in sorted(set(sr) | set(vae)) if sr.get(k) != vae.get(k)]
    assert differing == ["multi_contrast", "pairing"], (
        f"the SR arm must share stage1_vae_hf's data except the direction flip "
        f"and the dropped inert multi_contrast facade; differing keys = {differing}"
    )
    # `pairing` may only differ by the direction flip. The drain moved
    # data.bidirectional_mode into data.pairing.bidirectional_mode, so the
    # top-level comparison above now names the enclosing BLOCK rather than the
    # leaf -- which would let any other pairing knob diverge unnoticed. Pin the
    # leaf explicitly; this is stricter than the pre-drain assertion, which said
    # nothing about a block that did not yet exist.
    pairing_diff = sorted(
        k
        for k in set(sr.get("pairing") or {}) | set(vae.get("pairing") or {})
        if (sr.get("pairing") or {}).get(k) != (vae.get("pairing") or {}).get(k)
    )
    assert pairing_diff == [
        "bidirectional_mode"
    ], f"only the direction flip may differ inside data.pairing; got {pairing_diff}"
    # autoencode HF (input≡target=HF) vs ULF input -> HF target
    assert read_key(vae_doc, "data.pairing.bidirectional_mode") == "hf_to_hf"
    assert read_key(sr_doc, "data.pairing.bidirectional_mode") == "ulf_to_hf"
    # the dropped block is the (inert) facade, present only on the VAE side
    assert "multi_contrast" in vae and "multi_contrast" not in sr


def test_no_inert_multi_contrast_facade() -> None:
    """mamba_unet has no contrast conditioning and the nifti_paired loader does
    not read multi_contrast, so advertising it would be a #16 inert facade."""
    data = _load(_SR)["data"]
    assert "multi_contrast" not in data


def test_linearization_is_explicit_and_resolution_safe() -> None:
    """The SSM scan order is declared (not an implicit default) and is a
    resolution-agnostic mode — the arm trains on variable full-slice sizes and
    tests on native-geometry ULF, where strict square-pow2 ``hilbert_2d`` crashes."""
    mk = _load(_SR)["model"]["model_kwargs"]
    mode = mk["mamba_config"]["linearization_mode"]
    assert mode in ("raster", "hilbert_2d_rect"), (
        f"linearization_mode must be resolution-safe, got {mode!r} "
        f"(strict hilbert_2d/3d require square/cubic power-of-2 dims)"
    )


def test_same_manifest_and_root_as_vae() -> None:
    # Read through RENAMES: this guard reads the arm's TEXT, so the loader's
    # fold cannot help it once the cohort is drained to the canonical spelling.
    # Every leaf below is a `fold` record, so both spellings must stay
    # acceptable -- a legacy-spelled arm still loads.
    from tests.utils.corpus_keys import read_key

    doc = _load(_SR)
    assert (
        read_key(doc, "data.source.paired_manifest_path")
        == "data/manifests/ulf_paired_v6.json"
    )
    assert read_key(doc, "data.source.root") == "databases/ulf_paired/preprocessed"
    # the per-slice full-FOV pipeline is preserved
    assert read_key(doc, "data.sampling.enable_slice_2d") is True
    assert doc["data"]["modes"]["train"]["sampler"]["type"] == "full"


def test_is_a_reconstruction_resolution_task_not_vae() -> None:
    cfg = _load(_SR)
    assert cfg["model"]["model_type"] == "mamba_unet"  # recon, not autoencoder_kl
    assert cfg["training"]["task"] == "super_resolution"
    assert cfg["training"]["training_mode"] == "reconstruction"
    assert cfg["training"]["strategy_class"].endswith("ReconstructionTrainingStrategy")
    # no VAE/KL latent blocks leaked in from the reference config
    assert "latent" not in cfg["training"]
    assert "vae" not in cfg["training"]
