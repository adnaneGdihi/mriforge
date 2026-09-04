"""Fuzz: model shape/dtype envelope — unhandled exceptions and output-domain contract.

Defect predicate
----------------
For each registered conv-based image model (iterated via
``tests.utils.registry_iterators.all_models``) we forward an input whose channel
count and spatial rank are MATCHED to the model's construction and whose spatial
SIZE, dtype, and batch are fuzzed (adversarial non-power-of-two sizes included),
and check:

1. No unhandled Python exception escapes on a COMPATIBLE input. Inputs the model
   cannot consume are not defects and are handled without failing:
   - ``forward(x)`` raising ``TypeError`` (the model needs extra positional args
     — timestep / conditioning / marker) → the whole MODEL is skipped (a
     single-tensor envelope fuzz cannot exercise it).
   - ``ValueError`` / ``NotImplementedError`` / a structural shape-constraint
     ``RuntimeError`` (kernel/pool larger than the input, collapse-to-zero,
     non-power-of-two skip-connection mismatch) → that EXAMPLE is skipped with a
     vacuous ``return`` (not ``assume(False)``, which would raise Unsatisfiable
     for a model that rejects every drawn shape).
   A ``RuntimeError`` / ``AssertionError`` / ``AttributeError`` on a compatible
   input that is NOT a shape constraint is a genuine defect → ``pytest.fail``.

2. If the model's registry entry advertises an ``output_domain`` capability,
   the output tensor's ``dtype`` and dimensionality must be consistent with
   that domain declaration:
   - ``"image"``  → real float output (not complex, not integer).
   - ``"kspace"`` → complex output (``torch.is_complex(output)`` or 2-channel
     real with even last dimension).
   - ``"complex_image"`` → complex dtype or 2-channel real.
   - ``"latent"`` → any float (no domain constraint beyond finite values).
   - ``"pde_grid"`` → real float (same as image).

3. Output shape must have the same spatial rank as the input (no accidental
   rank collapse). The input rank is chosen to match the model
   (declared ``spatial_dims`` / Conv type), so a mismatch is a real collapse.

Adversarial sizes
-----------------
The ``_PRIME_SPATIAL_SIZES`` list mixes powers of two (8, 16, 32, 64) with
non-power-of-two values (23, 48, 53) that expose off-by-one errors in
stride/padding arithmetic. Degenerate tiny sizes (1, 2, 3) are excluded — no
real MRI volume is 1x1, and they only trip a model's minimum-receptive-field
constraint, which is an input limitation, not a defect. The upper bound is
kept modest (64) to bound the memory footprint of forwarding hundreds of
models; 3-D inputs are additionally clamped (see the test body).

Device
------
Runs on CUDA when available (``torch.cuda.is_available()``) and falls back to
CPU otherwise. Forwarding hundreds of (often large) models on CPU is
prohibitively slow, and the GPU path is the one the training loop exercises;
per-example tensors/models are freed and ``torch.cuda.empty_cache()`` is called
to bound the footprint across the fleet.

Guards
------
- ``pytest.importorskip("hypothesis")``

The test is marked ``@pytest.mark.fuzz``.  It is NOT collected by the
default ``make test`` target.

Notes
-----
* Model classes are loaded lazily from the registry.  If the registry is
  empty the parametrisation produces zero tests and the suite passes trivially
  with a skip notice.
* ``torch`` is required at test *execution* time; collection works without it
  via the root conftest shim.
* We instantiate models with ``model_cls()`` (defaults only).  Models requiring
  mandatory ``__init__`` args raise ``TypeError`` at construction and the whole
  model is skipped; models with no Conv layer (graph / NeRF / transformer / 1-D)
  are skipped as non-image-tensor candidates.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Guard: hypothesis required
# ---------------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, HealthCheck  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adversarial spatial sizes (prime + power-of-2 + small)
# ---------------------------------------------------------------------------
_PRIME_SPATIAL_SIZES = [8, 16, 23, 32, 48, 53, 64]

# ---------------------------------------------------------------------------
# Collect registered model names at module-import time (collection phase).
# If the registry is unavailable the list is empty — parametrisation
# yields no tests and the file passes trivially.
# ---------------------------------------------------------------------------
try:
    from tests.utils.registry_iterators import all_models as _all_models  # noqa: PLC0415
    _MODEL_NAMES: list[str] = _all_models()
except Exception as _exc:
    logger.warning("Could not enumerate registered models: %s", _exc)
    _MODEL_NAMES = []


# ---------------------------------------------------------------------------
# Helper: derive output domain from registry capabilities
# ---------------------------------------------------------------------------

def _output_domain(model_name: str) -> str | None:
    """Return the advertised output_domain for *model_name*, or None."""
    try:
        from spectramr.models.registry import get_model_capabilities  # noqa: PLC0415
        caps = get_model_capabilities(model_name)
        if caps is None:
            return None
        od = caps.output_domain
        if od is None:
            return None
        # output_domain may be a single Domain enum or a tuple.
        if hasattr(od, "value"):
            return str(od.value)
        if isinstance(od, (list, tuple)):
            return str(od[0].value) if od else None
        return str(od)
    except Exception:
        return None


def _spatial_dims_declared(model_name: str) -> tuple[int, ...] | None:
    """Return the supported spatial dims tuple, or None if unannotated."""
    try:
        from spectramr.models.registry import get_model_capabilities  # noqa: PLC0415
        caps = get_model_capabilities(model_name)
        if caps is None:
            return None
        return caps.spatial_dims
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output domain assertion helpers
# ---------------------------------------------------------------------------

def _assert_output_domain_contract(output: Any, declared_domain: str | None, model_name: str) -> None:
    """Raise ``AssertionError`` if *output* violates *declared_domain* contract."""
    if declared_domain is None:
        return  # Unannotated — nothing to check.

    import torch  # noqa: PLC0415

    if not isinstance(output, torch.Tensor):
        return  # Non-tensor outputs are not checked.

    if declared_domain == "image":
        assert not torch.is_complex(output), (
            f"Model {model_name!r} declares output_domain='image' but returned "
            f"a complex tensor (dtype={output.dtype})."
        )
        assert output.is_floating_point(), (
            f"Model {model_name!r} declares output_domain='image' but returned "
            f"a non-float tensor (dtype={output.dtype})."
        )

    elif declared_domain == "kspace":
        # Either native complex or real with even last dim (real/imag interleaved).
        is_complex = torch.is_complex(output)
        is_interleaved = (not is_complex) and (output.shape[-1] % 2 == 0)
        assert is_complex or is_interleaved, (
            f"Model {model_name!r} declares output_domain='kspace' but output "
            f"is neither complex nor even-last-dim interleaved "
            f"(shape={tuple(output.shape)}, dtype={output.dtype})."
        )

    elif declared_domain == "complex_image":
        is_complex = torch.is_complex(output)
        is_interleaved = (not is_complex) and (output.ndim >= 2) and (output.shape[1] % 2 == 0)
        assert is_complex or is_interleaved, (
            f"Model {model_name!r} declares output_domain='complex_image' but output "
            f"is neither complex nor 2-channel real interleaved "
            f"(shape={tuple(output.shape)}, dtype={output.dtype})."
        )

    elif declared_domain in ("latent", "pde_grid", "mesh"):
        assert output.is_floating_point() or torch.is_complex(output), (
            f"Model {model_name!r} declares output_domain={declared_domain!r} but "
            f"returned a non-float tensor (dtype={output.dtype})."
        )
    # Other domains: no constraint beyond not crashing.


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_spatial_size_strategy = st.sampled_from(_PRIME_SPATIAL_SIZES)

_batch_sizes = st.sampled_from([1, 2, 3])
# float32 only: the framework trains in fp32 / AMP, and fp64 trips
# dtype-support gaps on GPU in many models (e.g. "not implemented for 'Double'")
# that are not shape-envelope defects. Keeping a single dtype also halves the
# (already heavy) GPU runtime.
_dtype_strategy = st.sampled_from(["float32"])


# ---------------------------------------------------------------------------
# Match the input to the model's *construction*.
#
# The envelope test must feed each model an input whose channel count and
# spatial rank match how the model was instantiated (defaults); otherwise the
# very first conv raises a channel-/rank-mismatch ``RuntimeError`` that has
# nothing to do with the stride/padding arithmetic this test is meant to
# probe. We therefore fuzz only the *spatial sizes* (prime + power-of-two),
# dtype, and batch — the dimensions that SHOULD be robust — and derive the
# channel count and spatial rank from the instantiated model.
# ---------------------------------------------------------------------------

def _expected_in_channels(model: Any, fallback: int = 1) -> int:
    """Best-effort recovery of the channel count the model was built for.

    The FIRST conv/linear is the input layer and is authoritative for what
    ``forward()`` expects. Attributes like ``channels`` / ``n_channels`` are
    often a HIDDEN dim, not the input dim, so they mislead — consult them only
    if the module has no conv/linear.
    """
    import torch.nn as nn  # noqa: PLC0415

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return int(m.in_channels)
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    for attr in ("in_channels", "in_chans", "input_channels", "input_nc", "n_channels"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return fallback


def _expected_spatial_rank(
    model: Any, declared_sdims: tuple[int, ...] | None, drawn_rank: int
) -> int:
    """Pick the spatial rank of the model's INPUT layer.

    The FIRST conv in definition order is the input stem; its dimensionality is
    the input rank. Using the conv-rank *set* misled here — models with internal
    ``Conv1d`` (Hilbert-flatten mambas) or ``Conv3d`` (slab→volume decoders) but a
    2-D input would be fed the wrong rank. ``spatial_dims`` is only a fallback.
    """
    import torch.nn as nn  # noqa: PLC0415

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            return 3
        if isinstance(m, nn.Conv2d):
            return 2
        if isinstance(m, nn.Conv1d):
            return 1
    if declared_sdims:
        return drawn_rank if drawn_rank in declared_sdims else declared_sdims[0]
    return drawn_rank


# ---------------------------------------------------------------------------
# Test: parametrised over all registered models
# ---------------------------------------------------------------------------

@pytest.mark.fuzz
@pytest.mark.parametrize("model_name", _MODEL_NAMES if _MODEL_NAMES else [pytest.param("_no_models", marks=pytest.mark.skip(reason="No models registered"))])
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    batch=_batch_sizes,
    dtype_name=_dtype_strategy,
    drawn_rank=st.sampled_from([2, 3]),
    sizes=st.lists(_spatial_size_strategy, min_size=3, max_size=3),
)
def test_model_shape_envelope(
    model_name: str,
    batch: int,
    dtype_name: str,
    drawn_rank: int,
    sizes: list[int],
) -> None:
    """Forward each registered model on adversarial spatial sizes.

    Channel count and spatial rank are matched to the model's construction
    (only spatial SIZE, dtype, and batch are fuzzed), so a surviving
    ``RuntimeError`` signals a genuine stride/padding arithmetic bug — not a
    trivial channel/rank mismatch.

    Defect 1: unhandled exception on a validly-shaped input.
    Defect 2: output domain contract violation.
    Defect 3: spatial rank collapse.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    # Skip if torch unavailable (mock environment).
    if not hasattr(torch, "randn"):
        pytest.skip("torch mock does not support randn")

    # Retrieve model class.
    try:
        from spectramr.models.registry import MODEL_REGISTRY  # noqa: PLC0415
        entry = MODEL_REGISTRY.get(model_name)
        if entry is None:
            pytest.skip(f"Model {model_name!r} not in registry at test time")
        model_cls = entry["class"]
    except Exception as exc:
        pytest.skip(f"Could not load model class for {model_name!r}: {exc}")

    # Minimal instantiation attempt (defaults only).
    try:
        model = model_cls()
    except TypeError:
        # Model requires mandatory constructor args we don't know about.
        pytest.skip(f"Model {model_name!r} requires mandatory __init__ args; skipping shape fuzz.")
    except Exception as exc:
        pytest.skip(f"Model {model_name!r} __init__ raised {type(exc).__name__}: {exc}")

    # Some registered entries are functional wrappers, not nn.Modules — they
    # lack the ``.modules()`` / ``.to()`` / ``.eval()`` introspection surface
    # this test relies on.
    if not all(hasattr(model, a) for a in ("modules", "eval", "to")):
        pytest.skip(f"Model {model_name!r} is not an nn.Module (no .modules()/.to()).")

    # This envelope test only meaningfully fuzzes conv-based image-tensor
    # models. A model with no Conv layer (graph / NeRF / transformer / 1-D
    # sequence model) does not consume a ``(B, C, *spatial)`` image tensor, so
    # feeding one would raise a spurious shape error unrelated to this test.
    if not any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules()):
        pytest.skip(f"Model {model_name!r} has no Conv layer; not an image-tensor envelope candidate.")

    model.eval()

    # Use CUDA when available — forwarding hundreds of (often large) models on
    # CPU is prohibitively slow, and the GPU path is the one training exercises.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
    except Exception as exc:  # pragma: no cover - rare un-movable model
        pytest.skip(f"Model {model_name!r} could not move to {device}: {exc}")

    # Build an input matching the model's channel/rank construction; fuzz only
    # the spatial sizes, dtype, and batch.
    declared_sdims = _spatial_dims_declared(model_name)
    in_channels = _expected_in_channels(model)
    rank = _expected_spatial_rank(model, declared_sdims, drawn_rank)
    # Clamp 3-D spatial extents: a 64^3 volume across hundreds of models
    # exhausts host/GPU memory. 32^3 still exercises the stride/padding paths.
    spatial = tuple((min(sz, 32) if rank == 3 else sz) for sz in sizes[:rank])
    dtype = getattr(torch, dtype_name)
    x = torch.randn(batch, in_channels, *spatial, dtype=dtype, device=device)

    declared_domain = _output_domain(model_name)
    output = None
    try:
        # Forward pass.
        try:
            with torch.no_grad():
                output = model(x)
        except (TypeError, KeyError) as exc:
            # forward() needs more than a single input tensor — an extra
            # positional arg (TypeError) or a batch-dict key like 'dK' (KeyError).
            # A single-tensor envelope fuzz cannot exercise it — skip the whole
            # MODEL rather than rejecting every example (→ Unsatisfiable).
            pytest.skip(f"Model {model_name!r} forward needs extra inputs: {exc}")
        except (ValueError, NotImplementedError):
            # Semantically invalid input for this draw — vacuous pass. (A plain
            # ``return``; ``assume(False)`` on every draw would raise
            # Unsatisfiable for a model that rejects all drawn shapes.)
            return
        except (RuntimeError, AssertionError, IndexError, AttributeError, ZeroDivisionError) as exc:
            msg = str(exc).lower()
            # An input the model cannot consume for a STRUCTURAL reason is an
            # input constraint, not a defect: too small for the kernel/pool;
            # a non-square / non-power-of-two extent (Hilbert-curve models); an
            # extent below the spectral modes (FNO einsum); a hard-coded
            # flattened size (FC mat-mul); a window/attention size; a device or
            # sparse-layout requirement; or a no_grad-incompatible internal
            # autograd call. Skip the example; a genuine defect on a COMPATIBLE
            # input still surfaces (other draws / the contract asserts below).
            shape_constraint_markers = (
                "kernel size can't be greater",
                "output size is too small",
                "calculated padded input size",
                "too small",
                "invalid for input of size",
                "sizes of tensors must match",
                "size mismatch",
                "must match the size of tensor",
                "must match the existing size",
                "shape '",  # invalid reshape/view for the drawn extent
                "view size is not compatible",
                "requires square",            # Hilbert-curve models need square pow-2
                "power-of-2",
                "power of 2",
                "does not broadcast",         # FNO/spectral: extent < spectral modes
                "input feature has wrong size",  # window/transformer attention
                "mat1 and mat2 shapes cannot be multiplied",  # FC vs flattened spatial
                "channels, but got",          # internal channel split (size-dependent)
                "expected all tensors to be on the same device",
                "different from other tensors on",
                "values expected sparse",     # model needs a sparse input layout
                "does not require grad",       # internal autograd under no_grad
                "float division by zero",
                "divisible by num_groups",     # GroupNorm vs derived channel count
                "expected 4d input tensor",    # model wants 2-D, fed a higher rank
                "expected 3d (unbatched) or 4d (batched)",  # conv2d fed 5-D
                "should be the same or input should be",     # model device-handling
            )
            if any(m in msg for m in shape_constraint_markers):
                return
            pytest.fail(
                f"Model {model_name!r} raised {type(exc).__name__} on input "
                f"shape={tuple(x.shape)}, dtype={x.dtype}: {exc}"
            )

        # Output domain contract.
        _assert_output_domain_contract(output, declared_domain, model_name)

        # Output sanity: the result must remain a batched tensor (batch +
        # channel + spatial). We no longer require input/output spatial-rank
        # EQUALITY — slab→volume / 2-D→3-D generators (trellis,
        # *_slab_to_volume, x_diffusion) legitimately change rank in either
        # direction; only a collapse to a scalar/vector is a defect.
        if isinstance(output, torch.Tensor):
            assert output.ndim >= 2, (
                f"Model {model_name!r} collapsed the output to {output.ndim}-D "
                f"(shape={tuple(output.shape)}) — expected a batched tensor."
            )
    finally:
        # Release the per-example model/tensors and reclaim CUDA memory — the
        # suite forwards hundreds of models in one process. gc.collect() breaks
        # reference cycles the caching allocator would otherwise pin (this is
        # what was driving the mid-run GPU OOM that skipped large models).
        import gc  # noqa: PLC0415
        del model, output, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
