"""Regression tests for ``AttentionUNetGenerator`` attention_type dispatch.

Pitfall #9 / CLAUDE.md non-negotiable #3: an unrecognized ``attention_type``
must RAISE at construction time, never silently degrade to
``AttentionType.SPATIAL`` (which would train a different model than the YAML
requested and hide the config error from the audit).

Pre-fix: a typo such as ``attention_type="spacial"`` was swallowed and mapped
to SPATIAL. Post-fix: construction raises a ``ValueError`` enumerating the
advertised set, while every valid value still resolves byte-identically.
"""

from __future__ import annotations

import pytest

from mriforge.models.generators.unet_generator import AttentionUNetGenerator
from mriforge.models.reconstruction.unet import AttentionType


def test_unknown_attention_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown attention_type"):
        AttentionUNetGenerator(in_channels=1, out_channels=1, attention_type="spacial")


def test_unknown_attention_type_error_lists_valid_set() -> None:
    with pytest.raises(ValueError) as exc:
        AttentionUNetGenerator(in_channels=1, out_channels=1, attention_type="spacial")
    msg = str(exc.value)
    # The error must surface the advertised set so the audit/user can self-correct.
    assert "spatial" in msg
    assert "channel" in msg


def test_valid_attention_type_constructs() -> None:
    model = AttentionUNetGenerator(in_channels=1, out_channels=1, attention_type="spatial")
    # Resolved enum is honored (the valid path is unchanged by the fix).
    assert model.config.attention_type is AttentionType.SPATIAL
