"""Unit tests for core/metrics/spectroscopy_metrics.py.

Covers: SpectralLinewidth._find_peaks, SpectralLinewidth.forward.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.spectroscopy_metrics import SpectralLinewidth

# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spectral_linewidth_canary():
    """Single narrow peak → positive, finite FWHM value."""
    metric = SpectralLinewidth()
    spec = torch.zeros(1, 64)
    # Gaussian-like peak
    for i, v in enumerate([0.5, 0.8, 1.0, 0.8, 0.5]):
        spec[0, 28 + i] = v
    result = metric(spec)
    assert torch.isfinite(result)
    assert result.item() > 0.0


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "batch_size,n_freq",
    [
        (1, 64),
        (2, 128),
        (4, 32),
    ],
)
def test_forward_various_batch_sizes(batch_size, n_freq):
    """forward handles various [B, F] shapes without crashing."""
    metric = SpectralLinewidth()
    spec = torch.zeros(batch_size, n_freq)
    # Put a clear peak in each batch
    for b in range(batch_size):
        spec[b, n_freq // 2] = 1.0
        spec[b, n_freq // 2 - 1] = 0.6
        spec[b, n_freq // 2 + 1] = 0.6
    result = metric(spec)
    assert torch.isfinite(result)


@pytest.mark.unit
@pytest.mark.parametrize("peak_threshold", [0.3, 0.5, 0.7])
def test_peak_threshold_affects_detection(peak_threshold):
    """Different thresholds do not crash and return finite results."""
    metric = SpectralLinewidth(peak_threshold=peak_threshold)
    spec = torch.zeros(1, 64)
    spec[0, 32] = 1.0
    spec[0, 31] = 0.8
    spec[0, 33] = 0.8
    spec[0, 30] = 0.4
    spec[0, 34] = 0.4
    try:
        result = metric(spec)
        if result is not None:
            assert torch.isfinite(result)
    except TypeError:
        # None returned when no peaks found — acceptable
        pass


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zero_spectrum_no_peaks():
    """All-zero spectrum → no peaks → None or zero tensor returned (not crash)."""
    metric = SpectralLinewidth()
    spec = torch.zeros(1, 64)
    # Should not raise; may return None or a zero tensor
    try:
        result = metric(spec)
    except TypeError:
        pass  # None returned is acceptable


@pytest.mark.unit
def test_custom_freq_axis():
    """Providing an explicit freq_axis does not crash."""
    metric = SpectralLinewidth()
    n_freq = 64
    spec = torch.zeros(1, n_freq)
    spec[0, 32] = 1.0
    spec[0, 31] = 0.9
    spec[0, 33] = 0.9
    freq_axis = torch.linspace(0.0, 10.0, n_freq)
    result = metric(spec, freq_axis=freq_axis)
    if result is not None:
        assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wrong_ndim_raises():
    """1D input (not [B, F]) should raise an error (not silently proceed)."""
    metric = SpectralLinewidth()
    spec_1d = torch.zeros(64)
    with pytest.raises((ValueError, RuntimeError)):
        metric(spec_1d)


# ---------------------------------------------------------------------------
# Regression: quoted jaxtyping dims on the forward annotations (WS-2 fix)
#
# Three forward annotations used to carry malformed BARE-NAME jaxtyping dims
# (`Float[Tensor, freq]` / `Float[Tensor, batch]`). Under `from __future__
# import annotations` (PEP-563) every annotation is stored as a *string*, so
# the bug was invisible at import time; but when jaxtyping/beartype later
# evaluate the forward annotation, the bare name `freq` / `batch` is an
# undefined symbol -> NameError -> jaxtyping fail-soft SILENTLY swallows it
# and the dimension constraint becomes a no-op (no type check at all).
#
# The fix quotes the inner dim ("freq" / "batch" / "batch freq") so the stored
# annotation string `eval`s to a real `Float[...]` spec instead of raising.
# These tests assert the load-bearing property directly: the stored annotation
# strings carry the quoted form AND evaluate without NameError, which is the
# exact thing the bare-name bug broke. We also pin the happy path (a correctly
# shaped float freq_axis is accepted).
#
# Enforcement-form note: jaxtyping is installed in this env, but the source
# gates BOTH `jaxtyped` and `beartype` on one `try: import ...` block, and
# beartype is NOT installed here -> the decorators degrade to identity no-ops
# (JAXTYPING_AVAILABLE is False), so passing a wrong-dtype/wrong-length
# freq_axis does NOT raise a *type-check* error at runtime in this env. Per
# the task's explicit fallback, we therefore assert the structural fix (quoted
# dims that evaluate to a real jaxtyping spec, vs. the NameError-raising
# bare-name form) plus the happy path — NOT a no-op. This is strictly stronger
# than a runtime probe: it would FAIL on the pre-fix bare-name annotations
# regardless of whether beartype is present.
# ---------------------------------------------------------------------------

import inspect as _inspect  # noqa: E402

from spectramr.core.metrics.spectroscopy_metrics import (  # noqa: E402
    CRLB,
    FrequencyDomainSNR,
)


def _forward_annotation(cls, param_name):
    """Return the raw (PEP-563 string) annotation for a forward parameter."""
    sig = _inspect.signature(cls.forward)
    return sig.parameters[param_name].annotation


@pytest.mark.unit
@pytest.mark.audit_regression
@pytest.mark.parametrize(
    "cls,param,dim_token",
    [
        (SpectralLinewidth, "spectrum", "'batch freq'"),
        (SpectralLinewidth, "freq_axis", "'freq'"),
        (CRLB, "spectrum", "'batch freq'"),
        (CRLB, "peak_amplitude", "'batch'"),
        (CRLB, "model_spectrum", "'batch freq'"),
        (FrequencyDomainSNR, "spectrum", "'batch freq'"),
        (FrequencyDomainSNR, "freq_axis", "'freq'"),
    ],
)
def test_forward_jaxtyping_dims_are_quoted(cls, param, dim_token):
    """Regression: forward jaxtyping dims are QUOTED strings, not bare names.

    Pre-fix the annotation read ``Float[Tensor, freq]`` (bare ``freq``); under
    PEP-563 that stringifies to ``"Float[Tensor, freq]"`` with no quotes around
    the dim. The fix stores ``"Float[Tensor, 'freq']"`` (quoted). We assert the
    quoted token is present in the stored annotation string.
    """
    ann = _forward_annotation(cls, param)
    assert isinstance(ann, str), (
        f"expected PEP-563 string annotation for {cls.__name__}.{param}, got {ann!r}"
    )
    assert "Float[Tensor," in ann
    # The quoted dim token must appear; the buggy bare-name form would NOT
    # contain the surrounding single quotes.
    assert dim_token in ann, (
        f"{cls.__name__}.forward({param}) annotation {ann!r} is missing the "
        f"quoted jaxtyping dim {dim_token} (regressed to a bare-name no-op?)"
    )


@pytest.mark.unit
@pytest.mark.audit_regression
@pytest.mark.parametrize(
    "cls,param",
    [
        (SpectralLinewidth, "spectrum"),
        (SpectralLinewidth, "freq_axis"),
        (CRLB, "spectrum"),
        (CRLB, "peak_amplitude"),
        (CRLB, "model_spectrum"),
        (FrequencyDomainSNR, "spectrum"),
        (FrequencyDomainSNR, "freq_axis"),
    ],
)
def test_forward_annotation_evaluates_to_real_jaxtyping_spec(cls, param):
    """Regression: the stored annotation string `eval`s to a real Float[...] spec.

    This is the property the bug actually broke: with a BARE-NAME dim, eval of
    the annotation string raises ``NameError`` (``name 'freq' is not defined``),
    which jaxtyping fail-soft swallows into a silent no-op type check. With the
    quoted dim the string evaluates cleanly to a ``jaxtyping.Float[...]`` object.
    """
    jaxtyping = pytest.importorskip("jaxtyping")
    from torch import Tensor

    ann = _forward_annotation(cls, param)
    # Strip the optional ``| None`` union so we eval just the Float[...] core,
    # which is where the dim lives.
    core = ann.split("|")[0].strip()
    namespace = {"Float": jaxtyping.Float, "Tensor": Tensor}
    try:
        evaluated = eval(core, namespace)
    except NameError as e:  # pragma: no cover - this is the regressed path
        pytest.fail(
            f"{cls.__name__}.forward({param}) annotation {core!r} raised "
            f"NameError on eval ({e}) — bare-name jaxtyping dim regressed; "
            "jaxtyping would fail-soft this into a silent no-op type check."
        )
    # A real jaxtyping array-type carries a dim string; the bare-name form
    # could never reach here.
    assert "Float" in repr(evaluated)


@pytest.mark.unit
@pytest.mark.audit_regression
def test_bare_name_dim_would_raise_nameerror_control():
    """Control: the PRE-FIX bare-name form raises NameError on eval.

    Confirms the test above discriminates the fix from the bug (rather than
    passing vacuously): the buggy annotation string is constructed here and
    evaluating it must raise NameError, while the fixed quoted form must not.
    """
    jaxtyping = pytest.importorskip("jaxtyping")
    from torch import Tensor

    namespace = {"Float": jaxtyping.Float, "Tensor": Tensor}
    # Fixed (quoted) form evaluates fine.
    assert "Float" in repr(eval("Float[Tensor, 'freq']", namespace))
    # Buggy (bare-name) form raises NameError.
    with pytest.raises(NameError):
        eval("Float[Tensor, freq]", namespace)


@pytest.mark.unit
@pytest.mark.audit_regression
def test_happy_path_float_freq_axis_accepted():
    """Happy path: a correctly-shaped float32 freq_axis is accepted on all arms.

    The quoting fix must not break the working call. We feed each metric a
    matching-length float ``freq_axis`` and require a finite scalar result.
    """
    torch.manual_seed(0)
    n_freq = 64

    spec = torch.zeros(2, n_freq)
    for b in range(2):
        spec[b, 32] = 1.0
        spec[b, 31] = 0.7
        spec[b, 33] = 0.7
    freq_axis = torch.linspace(0.0, 10.0, n_freq)  # float32, len == n_freq
    assert freq_axis.dtype == torch.float32

    lw = SpectralLinewidth()(spec, freq_axis=freq_axis)
    assert lw.ndim == 0 and torch.isfinite(lw)

    snr = FrequencyDomainSNR()(spec, freq_axis=freq_axis)
    assert snr.ndim == 0 and torch.isfinite(snr)

    amp = torch.tensor([1.0, 1.0])  # [batch]
    crlb = CRLB()(spec, amp)
    assert crlb.ndim == 0 and torch.isfinite(crlb)


def test_crlb_scales_inversely_with_amplitude() -> None:
    """%CRLB ∝ 1/amplitude (the Cramér-Rao bound on amplitude is σ_noise/A).

    Doubling the amplitude must halve %CRLB. Pre-fix the code divided by the
    amplitude twice, giving a non-physical ``σ_noise / A^{1.5}`` whose ratio
    under doubling is ``2^{-1.5} ≈ 0.354``, not ``0.5``.
    """
    torch.manual_seed(0)
    n_freq = 64
    spec = torch.zeros(1, n_freq)
    spec[0, 32] = 1.0
    spec[0, -3:] = 0.05  # small noise floor in the high-ppm tail

    metric = CRLB(noise_estimation="baseline")
    crlb_a1 = metric(spec, torch.tensor([1.0]))
    crlb_a2 = metric(spec, torch.tensor([2.0]))

    assert crlb_a1 > 0
    assert (crlb_a2 / crlb_a1).item() == pytest.approx(0.5, rel=1e-3)
