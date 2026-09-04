"""``scripts/migrations/normalise_metadata_status.py``: line surgery, verified."""

from __future__ import annotations

import pytest
import yaml

from tests.utils.repo_scripts import load_script_module

# `scripts/migrations/` is not in the public allowlist, so loading it by path raised
# `FileNotFoundError` at module scope in the export -- a collection error, which
# aborts the whole session. The helper skips there, and still fails loudly when the
# script goes missing from a private checkout. It also owns the `sys.modules`
# registration this block used to do by hand.
_mod = load_script_module(
    "scripts/migrations/normalise_metadata_status.py", "normalise_metadata_status"
)

_ARM = """config_version: '1.0'
metadata:
  name: arm
  description: >
    two lines of
    prose
  tags:
    paradigm: diffusion
    status: {tag}
  status: {top}
  note: keep me
model:
  model_type: unet
"""


def _plan(text: str):
    out = _mod.plan_for(text)
    assert out is not None and out[0] != "UNMAPPED", out
    return out


def test_tag_spelling_moves_to_top_level_and_maps() -> None:
    text = _ARM.replace("  status: {top}\n", "").format(tag="operational_stub")
    new_text, summary = _plan(text)
    doc = yaml.safe_load(new_text)
    assert doc["metadata"]["status"] == "blocked"
    assert doc["metadata"]["status_reason"] == "operational_stub"
    assert "status" not in doc["metadata"]["tags"]
    assert doc["metadata"]["tags"]["paradigm"] == "diffusion"
    assert doc["metadata"]["note"] == "keep me"
    assert summary["moved_from_tags"] == "True"


def test_top_level_wins_when_both_spellings_exist() -> None:
    text = _ARM.format(tag="operational_stub", top="needs_implementation")
    new_text, _ = _plan(text)
    doc = yaml.safe_load(new_text)
    assert doc["metadata"]["status"] == "needs_implementation"
    assert doc["metadata"]["status_reason"] == "tags.status was operational_stub"
    assert "status" not in doc["metadata"]["tags"]


def test_canonical_top_level_status_is_left_alone() -> None:
    text = _ARM.replace("    status: {tag}\n", "").format(top="ready")
    assert _mod.plan_for(text) is not None  # top status present -> planned
    new_text, _summary = _plan(text)
    assert yaml.safe_load(new_text) == yaml.safe_load(text)


def test_unmapped_token_is_reported_not_guessed() -> None:
    text = _ARM.replace("    status: {tag}\n", "").format(top="something_new")
    assert _mod.plan_for(text) == ("UNMAPPED", {"raw": "something_new"})


def test_arm_without_status_needs_nothing() -> None:
    text = _ARM.replace("    status: {tag}\n", "").replace("  status: {top}\n", "")
    assert _mod.plan_for(text) is None


def test_prose_outside_the_status_keys_is_byte_identical() -> None:
    text = _ARM.replace("  status: {top}\n", "").format(tag="needs_implementation")
    new_text, _ = _plan(text)
    before = [line for line in text.splitlines() if "status" not in line]
    after = [line for line in new_text.splitlines() if "status" not in line]
    assert before == after


@pytest.mark.parametrize("token", sorted(_mod.CANONICAL_FOR))
def test_every_mapped_token_lands_in_the_vocabulary(token: str) -> None:
    from spectramr.config.schemas.base import EXPERIMENT_STATUSES

    assert _mod.CANONICAL_FOR[token] in EXPERIMENT_STATUSES
