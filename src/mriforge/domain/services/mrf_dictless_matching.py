"""Dictionary-free MRF matching via KD-tree on a conformal embedding.

Implements the inference-time half of MRF §3 of
``TODO/audit_plan_novel_fmri.md``:

* Embed every dictionary entry through a trained
  :class:`ConformalFPEmbedding` model into 5-D.
* Build an O(D log D) KD-tree.
* For each in-vivo fingerprint, embed it and look up the
  nearest-neighbour dictionary entry in O(log D) (vs the standard
  O(V·D·N) cosine matching).

The KD-tree uses :class:`scipy.spatial.cKDTree`. The fallback is a
brute-force NumPy nearest-neighbour search; the matcher is therefore
usable even without SciPy.

Canonical home: this is a pure numpy/torch matching primitive with no
``application`` dependencies, so it lives in ``domain/services`` — the MRF
training strategies (``infrastructure/training/strategies``) import it
*rightward* (infrastructure → domain), respecting the layer-direction rule.
It previously lived under ``application/use_cases`` and was imported leftward by
the strategies (a pitfall #13 violation); that path is now a re-export shim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MRFDictionary:
    """Embedded MRF dictionary container."""

    embeddings: np.ndarray  # [D, embed_dim]
    parameters: np.ndarray  # [D, 5] — (T1, T2, M0, B0, B1)


class MRFDictlessMatcher:
    """KD-tree-backed nearest-neighbour matcher in conformal-embedding space."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        normalise: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.normalise = normalise
        self._dictionary: MRFDictionary | None = None
        self._tree = None

    @staticmethod
    def _normalise(emb: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-8)
        return emb / n

    def build_index(
        self,
        fingerprints: torch.Tensor,
        parameters: torch.Tensor,
        *,
        batch_size: int = 256,
    ) -> None:
        """Embed the full dictionary and build a KD-tree.

        Args:
            fingerprints: ``[D, N]`` real-valued dictionary entries.
            parameters: ``[D, 5]`` corresponding Bloch parameters.
            batch_size: Embedding batch size for the forward pass.
        """
        if fingerprints.shape[0] != parameters.shape[0]:
            raise ValueError(
                f"D mismatch: fingerprints {fingerprints.shape[0]} vs parameters {parameters.shape[0]}"
            )
        embeddings = []
        with torch.no_grad():
            for start in range(0, fingerprints.shape[0], batch_size):
                chunk = fingerprints[start : start + batch_size].to(self.device)
                e = self.model(chunk).detach().cpu().numpy()
                embeddings.append(e)
        emb = np.concatenate(embeddings, axis=0).astype("float32")
        if self.normalise:
            emb = self._normalise(emb)
        self._dictionary = MRFDictionary(
            embeddings=emb, parameters=parameters.detach().cpu().numpy().astype("float32")
        )
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(emb)
            logger.info("MRFDictlessMatcher: built cKDTree over %d entries", emb.shape[0])
        except ImportError:
            self._tree = None
            logger.warning(
                "MRFDictlessMatcher: scipy not available — falling back to "
                "brute-force NumPy nearest-neighbour search."
            )

    def query(
        self,
        fingerprints: torch.Tensor,
        *,
        k: int = 1,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        """Match in-vivo fingerprints to their nearest-dictionary entry.

        Args:
            fingerprints: ``[V, N]`` real-valued query fingerprints
                (one per voxel).
            k: Number of nearest neighbours to return.
            batch_size: Embedding batch size.

        Returns:
            Dict with::

                {
                    "indices":      [V, k] — dictionary-row indices,
                    "parameters":   [V, k, 5] — Bloch params for each NN,
                    "distances":    [V, k] — Euclidean distance in embedding space,
                }
        """
        if self._dictionary is None:
            raise RuntimeError("MRFDictlessMatcher: call build_index(...) before query(...)")
        emb_queries = []
        with torch.no_grad():
            for start in range(0, fingerprints.shape[0], batch_size):
                chunk = fingerprints[start : start + batch_size].to(self.device)
                emb_queries.append(self.model(chunk).detach().cpu().numpy())
        Q = np.concatenate(emb_queries, axis=0).astype("float32")
        if self.normalise:
            Q = self._normalise(Q)
        if self._tree is not None:
            distances, indices = self._tree.query(Q, k=k)
            if k == 1:
                indices = indices[:, None]
                distances = distances[:, None]
        else:
            # Brute force: |Q · D^T| via numpy.
            sim = Q @ self._dictionary.embeddings.T
            # nearest = largest similarity = smallest 2 - 2·sim distance.
            indices = np.argsort(-sim, axis=-1)[:, :k]
            distances = np.linalg.norm(Q[:, None] - self._dictionary.embeddings[indices], axis=-1)
        params = self._dictionary.parameters[indices]
        return {
            "indices": indices,
            "parameters": params,
            "distances": distances.astype("float32"),
        }


__all__ = ["MRFDictionary", "MRFDictlessMatcher"]
