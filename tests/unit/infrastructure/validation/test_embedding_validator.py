"""Unit tests for ``mriforge.infrastructure.validation.embedding_validator``.

These tests pin down the public contract of
:class:`EmbeddingDimensionValidator`:

* The validator runs purely on Python dicts (no torch/numpy required).
* Divisibility ``embed_dim % num_heads == 0`` is enforced for transformer
  models — silent fallback would violate CLAUDE.md pitfall #9.
* Non-positive / non-integer dimensions are rejected with an actionable
  ``ValueError`` rather than being coerced.
* Hierarchical models (Swin-style) using a ``list`` of ``num_heads`` are
  validated element-wise.
* Non-transformer models opt out cleanly (the validator returns
  ``(True, None)`` rather than failing on missing fields).

Source contract (see ``embedding_validator.py``):

* ``validate_model_config(model_type, model_kwargs, raise_on_error=True)``
  returns ``Tuple[bool, Optional[str]]`` and raises ``ValueError`` when
  ``raise_on_error`` is True and validation fails.
* ``validate_training_config(config, raise_on_error=True)`` extracts the
  nested ``model.model_type`` and ``model.model_kwargs`` keys and delegates.
* ``suggest_fixes(embed_dim, num_heads)`` returns a multi-line string with
  fix suggestions.
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.validation.embedding_validator import (
    EmbeddingDimensionValidator,
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestValidEmbeddingPasses:
    """Valid configurations must return ``(True, None)``."""

    def test_vit_with_divisible_embed_dim_passes(self):
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "vit",
            {"embed_dim": 768, "num_heads": 12},
        )
        assert ok is True
        assert err is None

    def test_transformer_with_explicit_compatible_heads_passes(self):
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "transformer",
            {"embed_dim": 256, "num_heads": 8},
        )
        assert ok is True
        assert err is None

    def test_swin_hierarchical_num_heads_all_divide(self):
        """Swin uses a list of head counts (one per stage).

        Each stage must individually divide ``embed_dim``; valid case
        passes silently.
        """
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "swin",
            {"embed_dim": 96, "num_heads": [3, 6, 12]},
        )
        assert ok is True
        assert err is None

    def test_no_embed_dim_skips_validation(self):
        """Configs that omit ``embed_dim`` are not auto-rejected.

        The validator only triggers when the field is present — caller
        is responsible for separately enforcing presence if required.
        """
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "vit",
            {"num_heads": 12},
        )
        assert ok is True
        assert err is None

    def test_non_transformer_model_is_skipped(self):
        """Non-transformer model types short-circuit to ``(True, None)``.

        This is the only place where ``embed_dim`` need not be divisible
        — UNets etc. have no MultiHeadAttention.
        """
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "unet",
            {"embed_dim": 13, "num_heads": 12},  # would normally fail
        )
        assert ok is True
        assert err is None

    def test_default_num_heads_used_when_omitted(self):
        """When ``num_heads`` is missing, validator uses ``DEFAULT_HEADS``."""
        # vit default is 12; 768 % 12 == 0
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "vit",
            {"embed_dim": 768},
        )
        assert ok is True
        assert err is None


# ---------------------------------------------------------------------------
# Failure modes — actionable errors, no silent fallback
# ---------------------------------------------------------------------------


class TestInvalidEmbeddingRaises:
    """Mis-configured embeddings must raise ``ValueError`` (default mode).

    These tests enforce CLAUDE.md pitfall #9: the validator must NOT
    silently coerce a bad config — that's exactly the kind of latent
    bug that surfaces as an opaque ``assert`` failure deep inside
    MultiHeadSelfAttention.
    """

    def test_non_divisible_embed_dim_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            EmbeddingDimensionValidator.validate_model_config(
                "vit",
                {"embed_dim": 100, "num_heads": 12},
            )
        msg = str(excinfo.value)
        # Actionable: mentions the offending values and suggests fixes.
        assert "100" in msg
        assert "12" in msg
        assert "embed_dim" in msg
        assert "num_heads" in msg

    def test_non_divisible_returns_tuple_when_raise_disabled(self):
        ok, err = EmbeddingDimensionValidator.validate_model_config(
            "vit",
            {"embed_dim": 100, "num_heads": 12},
            raise_on_error=False,
        )
        assert ok is False
        assert isinstance(err, str)
        assert "100" in err and "12" in err

    def test_negative_embed_dim_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingDimensionValidator.validate_model_config(
                "vit",
                {"embed_dim": -64, "num_heads": 4},
            )

    def test_zero_embed_dim_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingDimensionValidator.validate_model_config(
                "vit",
                {"embed_dim": 0, "num_heads": 4},
            )

    def test_non_integer_embed_dim_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingDimensionValidator.validate_model_config(
                "vit",
                {"embed_dim": 64.5, "num_heads": 4},
            )

    def test_non_integer_num_heads_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingDimensionValidator.validate_model_config(
                "vit",
                {"embed_dim": 64, "num_heads": "bad"},
            )

    def test_swin_one_bad_layer_rejected(self):
        """If any layer in the hierarchy fails divisibility, the whole
        config is rejected and the error names the offending layer.
        """
        with pytest.raises(ValueError) as excinfo:
            EmbeddingDimensionValidator.validate_model_config(
                "swin",
                {"embed_dim": 96, "num_heads": [3, 5, 12]},  # 96 % 5 != 0
            )
        assert "Layer" in str(excinfo.value)
        assert "5" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Training-config extraction
# ---------------------------------------------------------------------------


class TestValidateTrainingConfig:
    """``validate_training_config`` extracts ``model.model_type`` and
    ``model.model_kwargs`` from a dict-like config object."""

    def test_dict_training_config_passes(self):
        config = {
            "model": {
                "model_type": "vit",
                "model_kwargs": {"embed_dim": 768, "num_heads": 12},
            }
        }
        ok, err = EmbeddingDimensionValidator.validate_training_config(config)
        assert ok is True
        assert err is None

    def test_dict_training_config_failure_raises(self):
        config = {
            "model": {
                "model_type": "vit",
                "model_kwargs": {"embed_dim": 100, "num_heads": 12},
            }
        }
        with pytest.raises(ValueError):
            EmbeddingDimensionValidator.validate_training_config(config)

    def test_missing_model_type_is_skipped(self):
        """Without a model_type the validator cannot decide whether the
        constraint applies — returns ``(True, None)`` rather than
        raising, leaving enforcement to the schema layer."""
        ok, err = EmbeddingDimensionValidator.validate_training_config({"model": {}})
        assert ok is True
        assert err is None

    def test_empty_config_does_not_raise(self):
        ok, err = EmbeddingDimensionValidator.validate_training_config({})
        assert ok is True
        assert err is None


# ---------------------------------------------------------------------------
# Fix suggestions
# ---------------------------------------------------------------------------


class TestSuggestFixes:
    def test_suggest_fixes_returns_actionable_string(self):
        suggestions = EmbeddingDimensionValidator.suggest_fixes(
            embed_dim=100, num_heads=12
        )
        # Should at least mention setting embed_dim to a multiple of num_heads.
        assert "embed_dim" in suggestions
        assert isinstance(suggestions, str)
        assert suggestions  # non-empty

    def test_find_divisor_returns_in_preferred_range_when_possible(self):
        d = EmbeddingDimensionValidator._find_divisor(64)
        assert 1 <= d <= 16
        assert 64 % d == 0
