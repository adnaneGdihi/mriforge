"""Skip markers for backends that are optional, or optional-and-CUDA-only.

Two distinct environment gaps live here. Both are gaps, not defects -- the
correct response is to skip, never to install a dependency so a number goes
down, and never to substitute a different implementation so a test goes green.

**torch-fidelity.** `FID`, `KID` and `IS` are torchmetrics wrappers around
``torch-fidelity``, which ships in the ``torchmetrics[image]`` extra and is NOT
part of this project's ``.[dev]`` install. Where it is absent the metric raises
at CONSTRUCTION::

    ModuleNotFoundError: Kernel Inception Distance metric requires that
    `Torch-fidelity` is installed. ...

Cluster job 17666023 reported 16 such failures. The probe is ``find_spec``,
evaluated once at import: the alternative, constructing the metric to see
whether it raises, downloads InceptionV3 weights on any machine that DOES have
the backend.

**mamba-ssm.** This one is a gap only in a *specific pairing*, which is why the
usual "is it installed" probe answers the wrong question. ``MambaBlock`` takes
the official selective-scan kernel whenever ``mamba_ssm`` imports, and that
kernel is CUDA-only -- it reaches ``causal_conv1d`` and raises ``Expected
x.is_cuda() to be true`` on CPU tensors. So a Mamba forward pass needs BOTH
halves, and each half alone is fine:

===================  ==================  ==============================
``mamba_ssm``        CUDA               outcome
===================  ==================  ==============================
absent               either             ``MambaBlock.__init__`` raises
                                        ImportError (fail-loud, #9)
present              available          the real kernel runs
present              **absent**         **kernel raises mid-forward**
===================  ==================  ==============================

The third row is the CPU test cluster, and it cost job 8000966 **78 failures
across 19 files** -- every one of them an opaque ``torch/_library`` dispatch
traceback that says nothing about CUDA. ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` does
NOT rescue it and is not meant to: that opt-in keys off the *ImportError* in row
1, and is scoped by CLAUDE.md to "boxes without the mamba_ssm CUDA kernel". This
box has the kernel. Reaching for it here would also swap a Gated-Conv+GRU in
under the "Mamba" label on 78 tests -- a green suite measuring something that is
not an SSM (pitfall #16).

``requires_cuda_for_mamba`` is therefore the honest marker: it declares where
that coverage actually lives rather than manufacturing it. Apply it per test or
per class, never per module -- these files also hold linearizer, registry and
permutation tests that are genuinely CPU-live and must keep running.

The probe mirrors ``MambaBlock.__init__``'s own two-step import rather than
``find_spec``, because an installed-but-unimportable kernel takes the fail-loud
path, not the CUDA one.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_TORCH_FIDELITY = importlib.util.find_spec("torch_fidelity") is not None

requires_torch_fidelity = pytest.mark.skipif(
    not HAS_TORCH_FIDELITY,
    reason=(
        "needs torch-fidelity (the torchmetrics[image] extra), which is not in "
        "this project's .[dev] install; FID/KID/IS raise at construction without it"
    ),
)


def _official_mamba_importable() -> bool:
    """Whether ``MambaBlock`` will dispatch to the official CUDA kernel.

    Mirrors the two-step import in
    ``spectramr.models.blocks.mamba_block.MambaBlock.__init__``; ``find_spec``
    alone would answer ``True`` for an installed-but-broken kernel, which takes
    the ImportError path instead and needs no CUDA. So the spec check only
    guards the real import rather than replacing it.

    That guard matters because this module is imported by ~21 test files,
    including the FID/KID ones that have nothing to do with Mamba. ``mamba_ssm``
    pulls in Triton and a compiled ``.so``; a failure there is not always an
    ``ImportError`` (``OSError`` on a missing shared object, ``RuntimeError``
    from CUDA init), and an escaping exception at module scope would take
    collection down for every consumer. Any failure to import means the block
    cannot dispatch to the kernel, which is exactly ``False``.
    """
    if importlib.util.find_spec("mamba_ssm") is None:
        return False
    try:
        from mamba_ssm import Mamba
    except ImportError:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except Exception:
            return False
    except Exception:
        return False
    return Mamba is not None


HAS_MAMBA_SSM = _official_mamba_importable()


def _cuda_available() -> bool:
    """``True`` only when torch reports a usable CUDA device.

    Broad ``except`` for the same reason as above: this runs at import of a
    module ~21 test files depend on, and ``torch.cuda.is_available()`` can raise
    on a half-configured driver. Unavailable-or-unaskable is ``False``, which is
    the conservative answer — it makes the marker skip rather than let a test
    reach a kernel that cannot run.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_cuda_for_mamba = pytest.mark.skipif(
    HAS_MAMBA_SSM and not _cuda_available(),
    reason="mamba_ssm installed → CUDA-only kernel; no GPU available",
)

__all__ = [
    "HAS_MAMBA_SSM",
    "HAS_TORCH_FIDELITY",
    "requires_cuda_for_mamba",
    "requires_torch_fidelity",
]
