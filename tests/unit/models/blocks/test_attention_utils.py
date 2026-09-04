"""Tests for ``AttentionConfig``, ``AttentionBackend`` and helpers.

Targets ``spectramr.models.blocks.attention_utils``. The wrapper integrates
FlashAttention / xFormers / memory-efficient SDP with graceful
fallbacks. Heavy backend testing is in the integration tier; here we
exercise the configuration surface that doesn't require external libs.
"""

from __future__ import annotations

from spectramr.models.blocks.attention_utils import (
    AttentionBackend,
    AttentionConfig,
    AttentionWrapper,
    get_attention_wrapper,
)


# ---------------------------------------------------------------------------
# AttentionBackend enum-like
# ---------------------------------------------------------------------------


def test_attention_backend_lists_supported_options() -> None:
    """``AttentionBackend`` exposes 4 string constants."""
    assert AttentionBackend.PYTORCH == "pytorch"
    assert AttentionBackend.XFORMERS == "xformers"
    assert AttentionBackend.FLASH_ATTENTION == "flash_attention"
    assert AttentionBackend.MEMORY_EFFICIENT == "memory_efficient"


# ---------------------------------------------------------------------------
# AttentionConfig defaults & overrides
# ---------------------------------------------------------------------------


def test_default_attention_config() -> None:
    """Default AttentionConfig prefers PyTorch + memory-efficient SDP."""
    cfg = AttentionConfig()
    assert cfg.backend == AttentionBackend.PYTORCH
    assert cfg.use_flash_attention is False
    assert cfg.use_xformers is False
    assert cfg.use_memory_efficient is True
    assert cfg.dropout == 0.0
    assert cfg.causal is False


def test_explicit_attention_config_overrides() -> None:
    """Explicit values override defaults."""
    cfg = AttentionConfig(
        backend=AttentionBackend.MEMORY_EFFICIENT,
        dropout=0.1,
        causal=True,
        scale=0.125,
    )
    assert cfg.backend == AttentionBackend.MEMORY_EFFICIENT
    assert cfg.dropout == 0.1
    assert cfg.causal is True
    assert cfg.scale == 0.125


def test_sdp_toggles_default_on() -> None:
    """All three SDP backends are enabled by default."""
    cfg = AttentionConfig()
    assert cfg.enable_math_sdp is True
    assert cfg.enable_flash_sdp is True
    assert cfg.enable_mem_efficient_sdp is True


# ---------------------------------------------------------------------------
# AttentionWrapper construction
# ---------------------------------------------------------------------------


def test_attention_wrapper_default_construction() -> None:
    """Default wrapper uses default config."""
    wrapper = AttentionWrapper()
    assert isinstance(wrapper.config, AttentionConfig)
    assert wrapper.config.backend == AttentionBackend.PYTORCH


def test_attention_wrapper_with_explicit_config() -> None:
    """Explicit config is persisted on the wrapper."""
    cfg = AttentionConfig(dropout=0.5)
    wrapper = AttentionWrapper(cfg)
    assert wrapper.config.dropout == 0.5


# ---------------------------------------------------------------------------
# get_attention_wrapper helper
# ---------------------------------------------------------------------------


def test_get_attention_wrapper_returns_instance() -> None:
    """Helper returns an ``AttentionWrapper`` instance."""
    wrapper = get_attention_wrapper()
    assert isinstance(wrapper, AttentionWrapper)


def test_get_attention_wrapper_with_custom_config() -> None:
    """Helper forwards a custom config."""
    cfg = AttentionConfig(causal=True)
    wrapper = get_attention_wrapper(cfg)
    assert wrapper.config.causal is True
