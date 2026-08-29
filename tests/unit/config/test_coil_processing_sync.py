"""Tests for the new→legacy coil-processing sync.

When a user authors the unified ``physics.coil_processing`` block, the loader
projects its *compression* axis onto the still-first-class legacy
``data.coil_processing_mode`` knob that ``FastMRISubjectBuilder._apply_coil_processing``
actually executes — so the new block drives the battle-tested per-sample path
without reimplementation or double-apply (pitfall #15).

The sync is opt-in (no-op when the new block is absent) and silent: there is NO
legacy→new migration and NO deprecation warning, so the ~200 existing YAMLs that
use ``data.coil_processing_mode`` / ``physics.coil_sensitivity`` load unchanged.
``data.coil_processing_mode`` is the live mechanism, not a deprecated one.
"""

import pathlib

import pytest

from mriforge.config.schemas.loader import (
    _derive_coil_processing_from_legacy,
    _sync_coil_processing_to_legacy,
)
from tests.utils.corpus import tracked_yamls

# ---------------------------------------------------------------------------
# _derive_coil_processing_from_legacy — legacy mode → full 4-axis block.
# ---------------------------------------------------------------------------

_EXPECTED_AXES = {
    "none": ("none", "none", "kspace", "complex"),
    "flatten": ("none", "none", "kspace", "real_interleaved"),
    "svd": ("svd", "none", "kspace", "real_interleaved"),
    "magnitude": ("none", "rss", "source", "magnitude"),
    "rss": ("none", "rss", "kspace", "real_interleaved"),
    "rss_image": ("none", "rss", "image", "magnitude"),
}


@pytest.mark.parametrize("mode,axes", list(_EXPECTED_AXES.items()))
def test_derive_each_legacy_mode(mode, axes):
    raw = {"data": {"coil_processing_mode": mode, "num_virtual_coils": 3}}
    out = _derive_coil_processing_from_legacy(raw)
    cp = out["physics"]["coil_processing"]
    comp, comb, dom, ch = axes
    assert cp["compression"]["method"] == comp
    assert cp["combine"]["method"] == comb
    assert cp["output"]["domain"] == dom
    assert cp["output"]["channels"] == ch
    if comp == "svd":
        assert cp["compression"]["num_virtual_coils"] == 3


def test_derive_noop_without_mode():
    raw = {"data": {"batch_size": 4}}
    out = _derive_coil_processing_from_legacy(dict(raw))
    assert "physics" not in out or "coil_processing" not in out.get("physics", {})


def test_derive_merges_partial_block_user_wins():
    # The already-migrated partial block (compression only) gets completed with
    # the svd-derived combine/output, but the user's compression sub-block wins.
    raw = {
        "data": {"coil_processing_mode": "svd", "num_virtual_coils": 1},
        "physics": {
            "coil_processing": {
                "compression": {"method": "svd", "num_virtual_coils": 8}
            }
        },
    }
    out = _derive_coil_processing_from_legacy(raw)
    cp = out["physics"]["coil_processing"]
    assert cp["compression"]["num_virtual_coils"] == 8  # user wins
    assert cp["combine"]["method"] == "none"  # filled from svd derivation
    assert cp["output"]["channels"] == "real_interleaved"  # filled


def test_derive_user_output_override_wins():
    # A user who wants svd to keep complex coils (a NEW combo) overrides output.
    raw = {
        "data": {"coil_processing_mode": "svd", "num_virtual_coils": 4},
        "physics": {
            "coil_processing": {"output": {"channels": "complex"}}
        },
    }
    out = _derive_coil_processing_from_legacy(raw)
    cp = out["physics"]["coil_processing"]
    assert cp["compression"]["method"] == "svd"  # filled
    assert cp["output"]["channels"] == "complex"  # user override wins



def test_sync_svd_sets_legacy_mode_and_coils():
    raw = {
        "physics": {
            "coil_processing": {
                "compression": {"method": "svd", "num_virtual_coils": 6}
            }
        }
    }
    out = _sync_coil_processing_to_legacy(raw)
    assert out["data"]["coils"]["processing_mode"] == "svd"
    assert out["data"]["coils"]["num_virtual_coils"] == 6


def test_sync_svd_default_coils_when_unspecified():
    raw = {"physics": {"coil_processing": {"compression": {"method": "svd"}}}}
    out = _sync_coil_processing_to_legacy(raw)
    assert out["data"]["coils"]["processing_mode"] == "svd"
    # num_virtual_coils not forced — left to the schema default downstream.
    assert "num_virtual_coils" not in out["data"]
    # calibration_lines not forced when unspecified (full-FoV default).
    assert "svd_calibration_lines" not in out["data"]


def test_sync_svd_calibration_lines():
    raw = {
        "physics": {
            "coil_processing": {
                "compression": {
                    "method": "svd",
                    "num_virtual_coils": 6,
                    "calibration_lines": 24,
                }
            }
        }
    }
    out = _sync_coil_processing_to_legacy(raw)
    assert out["data"]["coils"]["processing_mode"] == "svd"
    assert out["data"]["coils"]["num_virtual_coils"] == 6
    assert out["data"]["coils"]["svd_calibration_lines"] == 24


def test_sync_new_block_wins_over_conflicting_legacy_mode():
    raw = {
        "data": {"coil_processing_mode": "rss"},
        "physics": {
            "coil_processing": {"compression": {"method": "svd", "num_virtual_coils": 2}}
        },
    }
    out = _sync_coil_processing_to_legacy(raw)
    assert out["data"]["coils"]["processing_mode"] == "svd"


def test_sync_none_leaves_legacy_mode_untouched():
    raw = {
        "data": {"coil_processing_mode": "rss"},
        "physics": {"coil_processing": {"compression": {"method": "none"}}},
    }
    out = _sync_coil_processing_to_legacy(raw)
    # Reads back the LEGACY spelling on purpose. Every other case here asserts
    # the canonical `data.coils.processing_mode`, because that is where the sync
    # writes; this one asserts the input is passed through UNTOUCHED, so it must
    # read the key the input used. The fold happens later, in the loader.
    assert out["data"]["coil_processing_mode"] == "rss"
    assert "coils" not in out["data"], "method=none must not synthesise a block"


def test_sync_gcc_raises_not_implemented():
    raw = {"physics": {"coil_processing": {"compression": {"method": "gcc"}}}}
    with pytest.raises(NotImplementedError, match="gcc"):
        _sync_coil_processing_to_legacy(raw)


def test_sync_noop_without_new_block():
    raw = {"data": {"coil_processing_mode": "svd"}}
    out = _sync_coil_processing_to_legacy(dict(raw))
    assert out == raw


def test_sync_noop_without_physics_block():
    raw = {"data": {"batch_size": 4}}
    out = _sync_coil_processing_to_legacy(dict(raw))
    assert out == raw


def test_sync_creates_data_block_when_absent():
    raw = {
        "physics": {
            "coil_processing": {"compression": {"method": "svd", "num_virtual_coils": 3}}
        }
    }
    out = _sync_coil_processing_to_legacy(raw)
    assert out["data"]["coils"]["processing_mode"] == "svd"
    assert out["data"]["coils"]["num_virtual_coils"] == 3


def test_sync_does_not_warn():
    """The sync must be silent — mriforge.* DeprecationWarnings are errors in
    tests, so a warning here would break every legacy config that loads."""
    import warnings

    raw = {"physics": {"coil_processing": {"compression": {"method": "svd"}}}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → failure
        _sync_coil_processing_to_legacy(raw)


# ---------------------------------------------------------------------------
# Anti-regrowth: no shipped arm may pair dataset_type: m4raw with svd.
# ---------------------------------------------------------------------------


def _resolved_coil_mode(raw: dict) -> str | None:
    """The mode as the DATASET will see it, after the loader's two-way sync.

    Reading ``data.coil_processing_mode`` off the raw YAML is not enough:
    ``_sync_coil_processing_to_legacy`` re-synthesizes it from
    ``physics.coil_processing.compression.method``, so an arm can look migrated
    and still arrive at the dataset as ``svd``. Two of the eight arms migrated
    in this change were exactly that case.
    """
    raw = _derive_coil_processing_from_legacy(raw) or raw
    raw = _sync_coil_processing_to_legacy(raw) or raw
    return (raw.get("data") or {}).get("coil_processing_mode")


def test_no_shipped_arm_pairs_m4raw_with_svd():
    """``svd`` on the m4raw path raises ``NotImplementedError`` at construction.

    SVD virtual-coil compression is wired only on ``dataset_type: kspace``
    (``UniversalMRIDataset``). On the m4raw path it used to be accepted and then
    silently ignored — a facade (pitfall #16) that trained on uncompressed coils
    while the YAML advertised compression. Now it fails loud, which means any arm
    still carrying the pairing is a train-time crash rather than a quiet lie.

    This sweep is the guard: ``mriforge audit`` is Tier 0/1 static and does NOT
    catch the pairing, so without it a regrown arm passes pre-flight and dies
    only after cluster submission.
    """
    import yaml

    offenders = []
    for path in tracked_yamls(pathlib.Path("experiments")):
        if "archive" in path.parts:
            continue
        try:
            raw = yaml.safe_load(path.read_text())
        except Exception:
            continue  # not our concern here; schema tests cover unparseable YAML
        if not isinstance(raw, dict):
            continue
        dataset_type = (raw.get("data") or {}).get("dataset_type")
        if dataset_type != "m4raw":
            continue
        if _resolved_coil_mode(raw) == "svd":
            offenders.append(str(path))

    assert offenders == [], (
        "these arms pair dataset_type: m4raw with an svd coil mode, which now "
        "raises at dataset construction — set coil_processing_mode: none AND "
        "physics.coil_processing.compression.method: none (the loader "
        "re-synthesizes the legacy key from the unified block):\n  "
        + "\n  ".join(offenders)
    )


class TestTheDeriveReaderFollowsTheDrain:
    """`_derive_coil_processing_from_legacy` must see a drained arm's key.

    It runs BEFORE the rename fold, so a drained arm presents only
    `data.coils.processing_mode`. Reading the legacy spelling alone makes the
    derive silently no-op — and because the derive is what CREATES the
    `physics` block, the whole block goes missing rather than one key. That is
    a resolved-document change produced by a rewrite that is supposed to be a
    no-op; the drain oracle caught it on
    `physics_driven/experiment_53_multi_echo_b0_fit.yaml`.
    """

    def test_the_canonical_spelling_drives_the_derive(self):
        raw = {"data": {"coils": {"processing_mode": "svd"}}}
        out = _derive_coil_processing_from_legacy(raw)
        assert out["physics"]["coil_processing"]["compression"]["method"] == "svd"

    def test_the_legacy_spelling_still_drives_it(self):
        """Un-drained cohorts must keep working: 58 of 59 still spell it this
        way, so losing this leg would break far more than it fixed."""
        raw = {"data": {"coil_processing_mode": "svd"}}
        out = _derive_coil_processing_from_legacy(raw)
        assert out["physics"]["coil_processing"]["compression"]["method"] == "svd"

    def test_the_canonical_spelling_wins_when_both_are_present(self):
        """Matches the fold: the canonical path is the resolved one."""
        raw = {
            "data": {
                "coils": {"processing_mode": "svd"},
                "coil_processing_mode": "rss",
            }
        }
        out = _derive_coil_processing_from_legacy(raw)
        assert out["physics"]["coil_processing"]["compression"]["method"] == "svd"

    def test_neither_spelling_is_still_a_no_op(self):
        """Anti-vacuity: a reader that built `physics` unconditionally would
        satisfy every assertion above and change every arm in the corpus."""
        raw = {"data": {"dataset_type": "nifti"}}
        assert "physics" not in _derive_coil_processing_from_legacy(raw)

    def test_the_svd_knobs_follow_the_drain_too(self):
        """`processing_mode` was not the only legacy read in this function.

        `num_virtual_coils` and `svd_calibration_lines` fold to `data.coils.*`
        as well, and fixing only the mode left them silently falling back to
        the schema default — an arm asking for 1 virtual coil resolved to 4.
        The drain oracle caught it on 24 arms across three cohorts AFTER the
        mode fix had already landed, which is why all three now route through
        one reader instead of three lookups that have to be remembered
        separately.
        """
        raw = {
            "data": {
                "coils": {
                    "processing_mode": "svd",
                    "num_virtual_coils": 1,
                    "svd_calibration_lines": 12,
                }
            }
        }
        comp = _derive_coil_processing_from_legacy(raw)["physics"]["coil_processing"][
            "compression"
        ]
        assert comp["num_virtual_coils"] == 1
        assert comp["calibration_lines"] == 12

    def test_the_legacy_svd_knobs_still_work(self):
        """Un-drained roots (active/, validated/, training/) still spell these
        flat, so losing this leg would break far more than it fixed."""
        raw = {
            "data": {
                "coil_processing_mode": "svd",
                "num_virtual_coils": 1,
                "svd_calibration_lines": 12,
            }
        }
        comp = _derive_coil_processing_from_legacy(raw)["physics"]["coil_processing"][
            "compression"
        ]
        assert comp["num_virtual_coils"] == 1
        assert comp["calibration_lines"] == 12

    def test_the_default_still_applies_when_neither_declares_it(self):
        """Anti-vacuity: a reader returning None would satisfy neither
        assertion above but could still look correct in isolation."""
        raw = {"data": {"coils": {"processing_mode": "svd"}}}
        comp = _derive_coil_processing_from_legacy(raw)["physics"]["coil_processing"][
            "compression"
        ]
        assert comp["num_virtual_coils"] == 4

