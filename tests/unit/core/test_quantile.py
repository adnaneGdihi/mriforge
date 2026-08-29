"""Tests for the shared robust-quantile helper (data-layer audit D2).

``torch.quantile`` raises once the reduced dimension exceeds 2**24 elements.
Three copies of a decimating workaround had accumulated; the data layer had
none at all, and its five raw calls sit on the robust-scale path that decides
what intensity the model sees.
"""

from __future__ import annotations

import struct

import pytest
import torch

from mriforge.core.quantile import (
    QUANTILE_MAX_ELEMS,
    _fast_select_quantile,
    robust_quantile,
)


class TestBelowTheCapItIsATransparentDelegate:
    """The guard must not perturb any number it was not needed for."""

    @pytest.mark.parametrize("q", [0.0, 0.5, 0.9, 0.99, 1.0])
    def test_flat_matches_torch_exactly(self, q: float) -> None:
        torch.manual_seed(0)
        x = torch.rand(10_000)
        assert torch.equal(robust_quantile(x, q), torch.quantile(x, q))

    def test_dim_matches_torch_exactly(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(16, 512)
        assert torch.equal(robust_quantile(x, 0.75, dim=1), torch.quantile(x, 0.75, dim=1))


class TestAboveTheCapItDecimatesInsteadOfRaising:
    """Exercised with a small ``max_elems`` — allocating 2**24 floats per test
    would be 64 MB and slow, and the branch is the same one."""

    def test_flat_path_survives(self) -> None:
        x = torch.arange(10_000, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            # Prove the real cap exists in torch, at a size we can afford: the
            # library's own error is what this helper is here to avoid.
            torch.quantile(torch.empty(QUANTILE_MAX_ELEMS + 1), 0.5)
        out = robust_quantile(x, 0.5, max_elems=100)
        assert out.isfinite()

    def test_dim_path_survives(self) -> None:
        x = torch.arange(4 * 10_000, dtype=torch.float32).reshape(4, 10_000)
        out = robust_quantile(x, 0.5, dim=1, max_elems=100)
        assert out.shape == (4,)
        assert out.isfinite().all()

    def test_the_decimated_answer_stays_close(self) -> None:
        """Uniform subsampling leaves these quantiles statistically unchanged —
        the assumption the decimation rests on, so it is asserted rather than
        trusted.

        Tested at a 10x decimation. In production the ratio is far gentler: the
        stride is ``ceil(n / 2**24)``, so a tensor must exceed ~33.5M elements
        before even 2x kicks in, and the surviving sample count never drops
        below 16.7M. A 100x decimation (500 surviving samples) has a median
        standard error of ~0.022 and fails a 0.02 tolerance on sampling noise
        alone — which says something about the test, not the helper.
        """
        torch.manual_seed(0)
        x = torch.rand(50_000)
        for q in (0.5, 0.9, 0.99):
            exact = torch.quantile(x, q)
            approx = robust_quantile(x, q, max_elems=5_000)
            assert torch.allclose(exact, approx, atol=0.02), f"q={q}"

    def test_decimation_is_deterministic(self) -> None:
        """An even stride, never ``randperm``: a random draw here would perturb
        ``initialize_accelerator`` seeding and break reproducibility."""
        torch.manual_seed(0)
        x = torch.rand(50_000)
        first = robust_quantile(x, 0.9, max_elems=500)
        for _ in range(3):
            assert torch.equal(robust_quantile(x, 0.9, max_elems=500), first)


class TestTheDuplicatesDelegate:
    """One implementation, not three (#5: `data/` cannot import `infrastructure/`)."""

    def test_digital_twin_simulator_reexports_the_shared_helper(self) -> None:
        from mriforge.infrastructure.physics.digital_twin_simulator import (
            _QUANTILE_MAX_ELEMS,
            _robust_quantile,
        )

        assert _robust_quantile is robust_quantile
        assert _QUANTILE_MAX_ELEMS == QUANTILE_MAX_ELEMS

    def test_the_normalization_ssot_has_no_raw_quantile_calls_left(self) -> None:
        """The five data-layer sites were the unguarded ones."""
        import inspect

        from mriforge.data.transforms import normalization

        source = inspect.getsource(normalization)
        assert "torch.quantile(" not in source, (
            "a raw torch.quantile reappeared in the normalization SSOT; route "
            "it through mriforge.core.quantile.robust_quantile"
        )


def _bits(t: torch.Tensor) -> bytes:
    """Bit pattern of a 0-dim tensor.

    ``torch.equal`` accepts ``-0.0`` for ``0.0``, and that is exactly one of the
    two divergences these tests exist to catch — so identity is compared as
    bytes, never with ``==``.
    """
    return struct.pack("<d", float(t))


#: ``(n, q, seed)`` where evaluating the rank ``q * (n - 1)`` in float64 instead
#: of the tensor dtype changes the answer. Found by sweeping sizes x quantiles x
#: seeds and keeping the triples where the tensor-dtype rank matches
#: ``torch.quantile`` and the float64 rank does not -- so each one goes red if
#: the rank arithmetic in ``_fast_select_quantile`` is ever "simplified" back to
#: Python float. 4096 is the smallest discriminating size, which keeps these
#: fast; the effect is not exotic or large-tensor-only.
_RANK_DTYPE_DISCRIMINATORS = [
    (4096, 0.99, 0),
    (4096, 0.001, 1),
    (8192, 0.95, 2),
    (8192, 0.99, 0),
    (16384, 0.9, 1),
    (16384, 0.95, 0),
]


class TestTheFastPathIsBitIdenticalWhereItEngages:
    """#1537 proposed this route and rejected it as inexact above ~2**18
    elements. The rejection was mis-attributed: with the rank in the tensor's
    dtype the route is exact, including at the sizes the issue flagged."""

    @pytest.mark.parametrize("n", [4096, 16384, 102400, 262144, 1048576])
    @pytest.mark.parametrize("q", [0.0, 0.001, 0.5, 0.9, 0.99, 0.999, 1.0])
    def test_matches_torch_bit_for_bit(self, n: int, q: float) -> None:
        torch.manual_seed(0)
        x = torch.rand(n)
        assert _bits(robust_quantile(x, q)) == _bits(torch.quantile(x, q))

    @pytest.mark.parametrize(("n", "q", "seed"), _RANK_DTYPE_DISCRIMINATORS)
    def test_rank_is_evaluated_in_the_tensor_dtype(self, n: int, q: float, seed: int) -> None:
        """Regression pin for the actual cause of #1537's divergence.

        Each triple is a case where a float64 rank gives a different float32
        answer, so this fails if the ``torch.tensor(q, dtype=flat.dtype)`` in
        ``_fast_select_quantile`` is replaced by a bare ``q``.
        """
        torch.manual_seed(seed)
        x = torch.rand(n)
        assert _bits(robust_quantile(x, q)) == _bits(torch.quantile(x, q))

    def test_gradient_magnitude_shaped_data(self) -> None:
        """The GRI call site's real distribution: mostly zeros with a sparse
        tail, which is where ties are densest and selection most likely to
        disagree with a sort."""
        torch.manual_seed(0)
        x = torch.rand(102_400)
        x[x < 0.6] = 0.0
        for q in (0.5, 0.9, 0.95, 0.99):
            assert _bits(robust_quantile(x, q)) == _bits(torch.quantile(x, q)), q

    def test_float64_input(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(16_384, dtype=torch.float64)
        for q in (0.5, 0.9, 0.99):
            assert _bits(robust_quantile(x, q)) == _bits(torch.quantile(x, q)), q

    def test_non_contiguous_input(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(2, 16_384)[1]
        assert _bits(robust_quantile(x, 0.9)) == _bits(torch.quantile(x, 0.9))


class TestEachGuardDelegatesRatherThanReturnAWrongNumber:
    """One test per precondition, asserting both halves: the guard declines
    (``None``), and the public result still matches ``torch.quantile``. Testing
    only the public result would stay green if the guard were removed but the
    fast path happened to agree on that input."""

    def test_nan_is_not_silently_replaced_by_a_plausible_number(self) -> None:
        """``kthvalue`` does not propagate NaN. Without the guard this returns
        ~0.9039 where torch returns NaN -- a signal turned into a normal-looking
        value, the exact shape non-negotiable 3 forbids."""
        torch.manual_seed(0)
        x = torch.rand(16_384)
        x[17] = float("nan")

        assert _fast_select_quantile(x, 0.9) is None
        assert robust_quantile(x, 0.9).isnan()
        assert torch.quantile(x, 0.9).isnan()

    def test_infinities_delegate(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(16_384)
        x[3] = float("inf")
        x[9] = float("-inf")

        assert _fast_select_quantile(x, 0.9) is None
        assert _bits(robust_quantile(x, 0.9)) == _bits(torch.quantile(x, 0.9))

    def test_negative_zero_delegates(self) -> None:
        """``sort`` and ``kthvalue`` order tied -0.0/+0.0 differently, so the
        fast path can return +0.0 where torch returns -0.0. Numerically equal,
        which is why this needs a bitwise assertion to catch at all."""
        x = torch.zeros(16_384)
        x[:8192] = -0.0

        assert _fast_select_quantile(x, 0.999) is None
        assert _bits(robust_quantile(x, 0.999)) == _bits(torch.quantile(x, 0.999))

    def test_small_tensors_delegate(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(64)
        assert _fast_select_quantile(x, 0.9) is None
        assert _bits(robust_quantile(x, 0.9)) == _bits(torch.quantile(x, 0.9))

    def test_unsupported_dtype_delegates_so_torch_still_raises(self) -> None:
        """float16 is accepted by ``kthvalue`` but rejected by
        ``torch.quantile``. An ungated fast path would silently start accepting
        input this function currently raises on -- a widened contract, not a
        speedup."""
        x = torch.rand(16_384, dtype=torch.float16)
        assert _fast_select_quantile(x, 0.9) is None
        with pytest.raises(RuntimeError):
            robust_quantile(x, 0.9)

    def test_above_the_cap_delegates_so_torch_still_raises(self) -> None:
        """The float16 test's shape on the other precondition torch enforces and
        ``kthvalue`` does not: size. Unreachable with the default ``max_elems``
        -- decimation guarantees ``n <= max_elems`` before the call -- but a
        caller may widen ``max_elems`` past the cap, and an ungated fast path
        then answers where this function raises. Measured on ``rand``, the
        ungated route returned ``0.9000321626663208``; zeros here keep the 64
        MiB allocation cheap, since a size guard cannot read the contents."""
        x = torch.zeros(QUANTILE_MAX_ELEMS + 1, dtype=torch.float32)

        # Execute the premise rather than assert it in prose: above the cap
        # there is no reference value for the fast path to be identical *to*.
        with pytest.raises(RuntimeError):
            torch.quantile(x, 0.9)

        assert _fast_select_quantile(x, 0.9) is None
        with pytest.raises(RuntimeError):
            robust_quantile(x, 0.9, max_elems=QUANTILE_MAX_ELEMS + 1)

    @pytest.mark.parametrize("q", [-0.1, 1.5, float("nan")])
    def test_out_of_range_q_raises_through_the_one_owner(self, q: float) -> None:
        """``nan`` is here because the guard declines it *incidentally*: the
        chained ``0.0 <= q <= 1.0`` is ``False`` for NaN by IEEE rule, while the
        equivalent-looking ``not (q < 0.0 or q > 1.0)`` is ``True`` and would let
        it through to ``kthvalue``. torch agrees it belongs -- its own message is
        ``q must be in the range [0, 1] but got nan`` -- so a future rewrite of
        the guard to the other spelling must turn this red, not pass."""
        torch.manual_seed(0)
        x = torch.rand(16_384)
        assert _fast_select_quantile(x, q) is None
        with pytest.raises(RuntimeError):
            robust_quantile(x, q)


class TestTheFastPathComposesWithDecimation:
    """The two concerns in this module are sequential, not alternative: above
    the cap the tensor is decimated and the fast path then applies to the
    survivors. Both must hold at once."""

    def test_decimated_result_is_bit_identical_to_quantile_of_the_survivors(
        self,
    ) -> None:
        torch.manual_seed(0)
        x = torch.rand(60_000)
        max_elems = 20_000
        step = (60_000 + max_elems - 1) // max_elems
        expected = torch.quantile(x[::step], 0.9)
        assert _bits(robust_quantile(x, 0.9, max_elems=max_elems)) == _bits(expected)

    def test_decimation_is_still_deterministic_with_the_fast_path(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(60_000)
        first = robust_quantile(x, 0.9, max_elems=20_000)
        for _ in range(3):
            assert _bits(robust_quantile(x, 0.9, max_elems=20_000)) == _bits(first)
