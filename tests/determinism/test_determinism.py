"""Determinism harness — bit-exact reproducibility tests.

Three independent verification axes:

1. **Single-step parameter-delta hash** (``TestParamDeltaHash``):
   Build a tiny model (``ConvBlock2D`` with 4 channels), perform one
   forward+backward+optimiser step under a fixed seed, then SHA-256 hash the
   concatenated ``state_dict`` bytes.  Compare to a stored reference in
   ``tests/determinism/_reference_hashes.json``.

   Because the hash can only be calibrated on the *exact* hardware/CUDA/cuDNN
   version that will run it in production, the test **skips** if the reference
   key is absent rather than failing.  This prevents false negatives on fresh
   environments.

   To calibrate::

       pytest tests/determinism/test_determinism.py::TestParamDeltaHash -v -s

   The test will print the hash; paste it into ``_reference_hashes.json``
   under ``"param_delta_hash" → "<key>"``.

   The ``<key>`` encodes ``"<torch_version>/<cuda_available>/<device>"`` so
   different cluster generations can coexist in the same JSON.

2. **DataLoader worker-seed isolation** (``TestDataLoaderWorkerSeeds``):
   With ``num_workers > 0``, two independent DataLoader iterations over the
   same dataset must produce identical batch sequences.

3. **Forward-pass seed isolation** (``TestSeedIsolation``):
   Running the tiny model's forward pass twice under ``torch.manual_seed``
   gives identical outputs.

Markers
-------
- ``determinism``  : bit-exact reproducibility
- ``slow``         : tests 1 and 2 create subprocesses / multiple workers
- ``gpu``          : auto-tagged by conftest if CUDA paths detected
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from mriforge.models.blocks.convolutional import ConvBlock2D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASH_FILE = Path(__file__).parent / "_reference_hashes.json"
_SEED = 12345
_TINY_IN_CH = 4
_TINY_OUT_CH = 4
_TINY_SPATIAL = 16  # 16×16 feature maps → fast on CPU


def _make_tiny_model(seed: int = _SEED) -> ConvBlock2D:
    """Build a tiny deterministic model and reset its parameters from ``seed``."""
    torch.manual_seed(seed)
    model = ConvBlock2D(
        in_channels=_TINY_IN_CH,
        out_channels=_TINY_OUT_CH,
        kernel_size=3,
        padding=1,
        bias=True,
        norm_layer=None,
        activation=nn.ReLU(inplace=True),
    )
    # Explicit re-initialisation under controlled seed for cross-run stability
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.manual_seed(seed)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def _one_step(model: nn.Module, device: torch.device, seed: int = _SEED) -> dict[str, Any]:
    """Perform one fwd+bwd+SGD step and return the resulting ``state_dict``."""
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    x = torch.randn(
        1, _TINY_IN_CH, _TINY_SPATIAL, _TINY_SPATIAL, generator=torch.Generator().manual_seed(seed)
    ).to(device)
    optimiser.zero_grad(set_to_none=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimiser.step()
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _hash_state_dict(sd: dict[str, Any]) -> str:
    """SHA-256 of sorted state_dict tensors concatenated as raw bytes."""
    hasher = hashlib.sha256()
    for key in sorted(sd.keys()):
        buf = io.BytesIO()
        torch.save(sd[key], buf)
        hasher.update(buf.getvalue())
    return hasher.hexdigest()


def _calibration_key() -> str:
    """Unique key for the current environment (torch version + CUDA availability)."""
    cuda_str = "cuda" if torch.cuda.is_available() else "cpu"
    return f"{torch.__version__}/{cuda_str}"


def _load_reference_hashes() -> dict[str, str]:
    if not _HASH_FILE.exists():
        return {}
    with _HASH_FILE.open() as f:
        data = json.load(f)
    return data.get("param_delta_hash", {})


def _save_reference_hash(key: str, value: str) -> None:
    if not _HASH_FILE.exists():
        doc: dict[str, Any] = {"_schema": "1", "param_delta_hash": {}}
    else:
        with _HASH_FILE.open() as f:
            doc = json.load(f)
    doc.setdefault("param_delta_hash", {})[key] = value
    with _HASH_FILE.open("w") as f:
        json.dump(doc, f, indent=2)


# ---------------------------------------------------------------------------
# 1. Param-delta hash
# ---------------------------------------------------------------------------


class TestParamDeltaHash:
    """Single-step SGD parameter-delta is bit-exact across runs."""

    def _run_and_hash(self, device: torch.device) -> str:
        model = _make_tiny_model(_SEED)
        sd = _one_step(model, device, seed=_SEED)
        return _hash_state_dict(sd)

    @pytest.mark.determinism
    def test_two_runs_produce_same_hash_cpu(self) -> None:
        """Two independent CPU runs under the same seed produce the same hash."""
        device = torch.device("cpu")
        h1 = self._run_and_hash(device)
        h2 = self._run_and_hash(device)
        assert h1 == h2, (
            f"Param-delta hash differs between runs on CPU:\n"
            f"  run1={h1}\n  run2={h2}\n"
            "This indicates non-deterministic weight initialisation or "
            "non-deterministic operator selection."
        )

    @pytest.mark.determinism
    @pytest.mark.gpu
    def test_two_runs_produce_same_hash_cuda(self) -> None:
        """Two independent CUDA runs under the same seed produce the same hash."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        h1 = self._run_and_hash(device)
        h2 = self._run_and_hash(device)
        assert h1 == h2, (
            f"Param-delta hash differs between CUDA runs:\n"
            f"  run1={h1}\n  run2={h2}\n"
            "Ensure TORCH_CUDNN_V8_API_DISABLED=0 is unset and "
            "torch.backends.cudnn.deterministic=True is set."
        )

    @pytest.mark.determinism
    def test_hash_matches_reference_or_skip_for_calibration(self) -> None:
        """Compare to stored reference; skip (not fail) if uncalibrated.

        To calibrate on the cluster::

            pytest tests/determinism/test_determinism.py -v -s -k "calibrate"

        Or just run this test once; it will print the hash to store.

        To store it::

            python - <<'EOF'
            import json, hashlib, io, torch
            # … (see test body for _run_and_hash and _calibration_key) …
            EOF

        Or call the helper in a one-off script::

            from tests.determinism.test_determinism import (
                TestParamDeltaHash, _calibration_key, _save_reference_hash
            )
            t = TestParamDeltaHash()
            h = t._run_and_hash(torch.device("cpu"))
            _save_reference_hash(_calibration_key(), h)
            print(f"Stored hash for key '{_calibration_key()}': {h}")
        """
        key = _calibration_key()
        refs = _load_reference_hashes()
        if key not in refs:
            h = self._run_and_hash(torch.device("cpu"))
            pytest.skip(
                f"Reference hash not yet calibrated for environment key='{key}'.\n"
                f"Computed hash: {h}\n"
                f"Run once and store in {_HASH_FILE}:\n"
                f"  {{\"param_delta_hash\": {{\"{key}\": \"{h}\"}}}}"
            )

        expected = refs[key]
        actual = self._run_and_hash(torch.device("cpu"))
        assert actual == expected, (
            f"Param-delta hash regression detected for key='{key}'!\n"
            f"  expected : {expected}\n"
            f"  actual   : {actual}\n"
            "A PyTorch or library update may have changed operator numerics. "
            f"If intentional, update {_HASH_FILE}."
        )


# ---------------------------------------------------------------------------
# 2. DataLoader worker-seed isolation
# ---------------------------------------------------------------------------


class _SeededDataset(data.Dataset):
    """Simple deterministic dataset: item[i] = tensor of value i."""

    def __init__(self, n: int = 64) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(float(idx))


def _seed_worker(worker_id: int) -> None:
    """Worker init function that seeds each worker deterministically."""
    import random

    import numpy as np

    base = torch.initial_seed() % (2**32)
    np.random.seed(base)
    random.seed(base)


def _batch_sequence_hash(loader: data.DataLoader) -> str:
    """Consume a DataLoader and return SHA-256 of concatenated batch bytes."""
    hasher = hashlib.sha256()
    for batch in loader:
        hasher.update(batch.numpy().tobytes())
    return hasher.hexdigest()


class TestDataLoaderWorkerSeeds:
    """Two passes through the same DataLoader (num_workers>0) are identical."""

    @pytest.mark.determinism
    @pytest.mark.slow
    def test_multiworker_two_passes_identical(self) -> None:
        """Identical shuffle + seed → identical batch sequence across two epochs."""
        # Limit workers to avoid fork overhead in CI; 2 is enough to exercise
        # the seed-isolation path.
        n_workers = min(2, multiprocessing.cpu_count())

        dataset = _SeededDataset(n=64)
        g = torch.Generator()
        g.manual_seed(_SEED)

        loader = data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=n_workers,
            worker_init_fn=_seed_worker,
            generator=g,
            persistent_workers=False,
        )

        h1 = _batch_sequence_hash(loader)

        # Reset generator to the same state for the second pass
        g.manual_seed(_SEED)
        h2 = _batch_sequence_hash(loader)

        assert h1 == h2, (
            "DataLoader produced different batch sequences on two passes with "
            f"the same generator seed (num_workers={n_workers}).\n"
            f"  pass1={h1}\n  pass2={h2}\n"
            "Check worker_init_fn and torch.Generator usage."
        )

    @pytest.mark.determinism
    def test_single_worker_two_passes_identical(self) -> None:
        """Baseline: single-worker DataLoader is trivially reproducible."""
        dataset = _SeededDataset(n=32)
        g = torch.Generator()
        g.manual_seed(_SEED)

        loader = data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=g,
        )

        h1 = _batch_sequence_hash(loader)
        g.manual_seed(_SEED)
        h2 = _batch_sequence_hash(loader)

        assert h1 == h2, (
            f"Single-worker DataLoader hash mismatch: {h1} vs {h2}"
        )


# ---------------------------------------------------------------------------
# 3. Forward-pass seed isolation
# ---------------------------------------------------------------------------


class TestSeedIsolation:
    """torch.manual_seed → identical forward-pass outputs."""

    @pytest.mark.determinism
    def test_forward_pass_identical_under_same_seed(self) -> None:
        """Two model.eval() forward passes under the same seed are bit-identical."""
        model = _make_tiny_model(_SEED)
        model.eval()
        device = torch.device("cpu")
        model = model.to(device)

        torch.manual_seed(_SEED)
        x = torch.randn(1, _TINY_IN_CH, _TINY_SPATIAL, _TINY_SPATIAL)

        with torch.no_grad():
            out1 = model(x.clone())
            out2 = model(x.clone())

        assert torch.equal(out1, out2), (
            "model.eval() is non-deterministic: two forward passes on identical "
            "inputs produced different outputs.\n"
            f"  max diff = {(out1 - out2).abs().max().item()}"
        )

    @pytest.mark.determinism
    def test_two_independent_seeded_runs_identical(self) -> None:
        """Re-seeding to the same value reproduces the same forward output."""
        device = torch.device("cpu")

        def _run() -> torch.Tensor:
            model = _make_tiny_model(_SEED)
            model.eval()
            model = model.to(device)
            torch.manual_seed(_SEED)
            x = torch.randn(1, _TINY_IN_CH, _TINY_SPATIAL, _TINY_SPATIAL)
            with torch.no_grad():
                return model(x)

        out1 = _run()
        out2 = _run()

        assert torch.equal(out1, out2), (
            "Two independent seeded runs produced different outputs.\n"
            f"  max diff = {(out1 - out2).abs().max().item()}\n"
            "Weight initialisation in _make_tiny_model may not be fully seeded."
        )

    @pytest.mark.determinism
    def test_training_step_gradient_is_reproducible(self) -> None:
        """Gradient tensors after one backward pass are identical across seeded runs."""
        device = torch.device("cpu")

        def _grad_hash(seed: int) -> str:
            torch.manual_seed(seed)
            model = _make_tiny_model(seed)
            model.train()
            model = model.to(device)
            x = torch.randn(
                1, _TINY_IN_CH, _TINY_SPATIAL, _TINY_SPATIAL,
                generator=torch.Generator().manual_seed(seed),
            )
            y = model(x)
            y.mean().backward()
            hasher = hashlib.sha256()
            for p in model.parameters():
                if p.grad is not None:
                    buf = io.BytesIO()
                    torch.save(p.grad.cpu(), buf)
                    hasher.update(buf.getvalue())
            return hasher.hexdigest()

        h1 = _grad_hash(_SEED)
        h2 = _grad_hash(_SEED)
        assert h1 == h2, (
            f"Gradient hashes differ across seeded runs: {h1} vs {h2}"
        )
