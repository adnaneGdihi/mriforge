"""Lazy-encode wrapper for stage-2 latent training (Phase 4f, Amendment H).

Wraps an inner Dataset so every yielded sample is passed through a frozen
stage-1 encoder (VAE / VQ-VAE) on access. The encoded latent is attached
to the returned dict under the configured ``latent_key`` while the raw
image remains under ``input`` for downstream losses.

Optional disk caching: when ``cache_dir`` is set, latents are persisted
on first encounter and reloaded on subsequent epochs.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.config.schemas.data import LatentDiffusionConfigSchema

logger = logging.getLogger(__name__)


__all__ = ["LazyEncodeWrapper"]


class LazyEncodeWrapper(Dataset):
    """Wrap an inner Dataset, encoding each sample on access.

    Args:
        inner: The underlying Dataset (yields tio.Subject or dict).
        config: Populated :class:`LatentDiffusionConfigSchema`.
        encoder: Frozen encoder callable. Receives ``input`` tensor and
            returns the latent tensor. Loaded from
            ``config.stage1_checkpoint`` if not provided explicitly.
        device: Where to run the encoder.

    The encoder is invoked under ``torch.no_grad()`` — gradient flows
    only through the stage-2 path that consumes the latent.

    Cache key: ``sha256(file_id + sample-index)``. Keyed off ``file_id``
    when the inner Subject exposes it (universal dataset, quantitative
    dataset, cine dataset all do); falls back to a hash of the input
    tensor's data pointer for in-memory datasets.
    """

    def __init__(
        self,
        inner: Dataset,
        config: LatentDiffusionConfigSchema,
        encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
    ):
        if not config.enabled:
            raise ValueError("LazyEncodeWrapper requires latent_diffusion.enabled=True.")
        self.inner = inner
        self.config = config
        self.device = device or torch.device("cpu")
        self.encoder = encoder if encoder is not None else self._load_encoder()
        self.cache_dir = Path(config.cache_dir) if config.cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tio.Subject | dict[str, Any]:
        sample = self.inner[idx]

        # Locate the input tensor on the sample. Both tio.Subject and
        # plain dict-style samples expose it under ``input``.
        input_tensor = self._extract_input(sample)
        cache_key = self._cache_key(sample, idx, input_tensor)

        latent = self._load_cached(cache_key)
        if latent is None:
            with torch.no_grad():
                latent = self.encoder(input_tensor.to(self.device))
            self._store_cached(cache_key, latent)

        # Attach the latent under the configured key.
        if isinstance(sample, tio.Subject):
            sample[self.config.latent_key] = tio.ScalarImage(tensor=latent.cpu())
        elif isinstance(sample, dict):
            sample[self.config.latent_key] = latent.cpu()
        else:
            raise TypeError(
                f"LazyEncodeWrapper expects inner samples to be tio.Subject "
                f"or dict; got {type(sample).__name__}."
            )
        return sample

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_input(sample: Any) -> torch.Tensor:
        if isinstance(sample, tio.Subject):
            img = sample.get("input")
            if img is None:
                raise KeyError("LazyEncodeWrapper expects 'input' on the inner Subject.")
            return img.data
        if isinstance(sample, dict):
            if "input" in sample:
                tensor = sample["input"]
                return tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            raise KeyError("LazyEncodeWrapper expects an 'input' key on the dict sample.")
        raise TypeError(f"Unsupported sample type for LazyEncodeWrapper: {type(sample).__name__}")

    def _cache_key(
        self,
        sample: Any,
        idx: int,
        input_tensor: torch.Tensor,
    ) -> str:
        # Prefer file_id when the Subject exposes it (stable across runs).
        file_id = None
        if isinstance(sample, tio.Subject) or isinstance(sample, dict):
            file_id = sample.get("file_id") or sample.get("path")
        if file_id is not None:
            return hashlib.sha256(f"{file_id}-{idx}".encode()).hexdigest()
        # Fallback for in-memory datasets — input tensor stats.
        sig = f"{tuple(input_tensor.shape)}-{idx}"
        return hashlib.sha256(sig.encode()).hexdigest()

    def _load_cached(self, key: str) -> torch.Tensor | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Failed to load cached latent at %s: %s", path, e)
            return None

    def _store_cached(self, key: str, latent: torch.Tensor) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.pt"
        torch.save(latent.cpu(), str(path))

    def _load_encoder(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Lazily load the stage-1 checkpoint and freeze it.

        Returns a closure that runs the encoder on a single tensor.
        Subclasses can override this if the stage-1 checkpoint format
        requires different deserialization.
        """
        if self.config.stage1_checkpoint is None:
            raise ValueError("stage1_checkpoint must be set to load the encoder.")
        checkpoint_path = Path(self.config.stage1_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"stage1_checkpoint not found at {checkpoint_path}")
        state = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        # Two common checkpoint shapes:
        # 1) A full nn.Module saved via `torch.save(model, ...)`.
        # 2) A state_dict — caller must register the model class.
        if isinstance(state, torch.nn.Module):
            module = state.to(self.device)
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
            return module
        raise ValueError(
            "LazyEncodeWrapper auto-loading supports only torch.save(module, ...) "
            "checkpoints. For state_dict checkpoints, pass an explicit ``encoder`` "
            "callable to LazyEncodeWrapper(...)."
        )
