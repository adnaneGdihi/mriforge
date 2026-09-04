"""SSOT for MAE/pretraining config access (issue #368).

The MAE strategy reads the typed ``training.mae`` block. That read used to be a
``getattr(self.config.training, "mae", None)`` fallback and is now guarded direct
access -- the schema declares ``training.mae`` (a first-class field), so the
getattr default was redundant SSOT. The legacy ``task_settings`` dict reads are
intentionally left as getattr: ``task_settings`` is NOT a schema field, so direct
access would raise; they are the two known-remaining entries the ssot ratchet
already accounts for. These tests pin both facts so a new config getattr-fallback
cannot creep in unnoticed (the global test_ssot_compliance guard is CI-advisory
and let pretraining.py drift past its allowance once).
"""

from __future__ import annotations

import inspect
import re


def _source() -> str:
    import spectramr.infrastructure.training.strategies.pretraining as pre

    return inspect.getsource(pre)


def test_mae_is_read_via_direct_attribute_access() -> None:
    src = _source()
    assert 'getattr(self.config.training, "mae"' not in src, (
        "training.mae is a first-class schema field; read it directly "
        "(self.config.training.mae), not via a getattr fallback."
    )
    assert re.search(r"self\.config\.training\.mae\b", src), (
        "expected a direct self.config.training.mae read after the conversion"
    )


def test_config_getattr_fallbacks_do_not_exceed_the_known_ratchet() -> None:
    # Same flat pattern the ssot ratchet uses. Only the two documented legacy
    # task_settings reads may remain; no new config getattr-fallbacks.
    src = _source()
    flat = re.findall(r"getattr\s*\(\s*[a-zA-Z_.]*config\.[a-zA-Z_.]+\s*,", src)
    assert len(flat) <= 2, f"pretraining.py config getattr-fallbacks grew: {flat}"
