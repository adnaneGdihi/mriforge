"""MRF dictionary-matching loss — the honest backing for mri_fingerprinting.

The regime had no real loss at all before this: every MRF-*named* loss in the
repo is a themed name over generic maths (``tropical_mrf_consistency``'s
reference fan is ``torch.randn``). These tests pin that this one actually does
the physics — dictionary matching — and that it refuses to run without a
dictionary rather than degrading into something else under the same name.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.config.schemas.enums import Regime
from spectramr.models.losses.mrf_losses import (
    MRFDictionaryMatchLoss,
    soft_dictionary_match,
)


def _dictionary(
    n_entries: int = 16, length: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """A toy Bloch-like dictionary: decaying trajectories indexed by (T1, T2)."""
    t = torch.linspace(0, 1, length).reshape(1, length)
    t1 = torch.linspace(300.0, 2000.0, n_entries).reshape(-1, 1)
    t2 = torch.linspace(30.0, 200.0, n_entries).reshape(-1, 1)
    entries = (1 - torch.exp(-t * 1000.0 / t1)) * torch.exp(-t * 1000.0 / t2)
    params = torch.cat([t1, t2], dim=1)  # [D, 2]
    return entries, params


class TestSoftDictionaryMatch:
    def test_an_exact_dictionary_entry_recovers_its_own_parameters(self) -> None:
        """The defining MRF operation, at low temperature.

        Hand it a fingerprint that IS dictionary entry k and the soft match must
        return entry k's parameters. This is the property that makes the loss
        physics rather than a regulariser.
        """
        dictionary, params = _dictionary()
        for k in (0, 5, 15):
            matched = soft_dictionary_match(
                dictionary[k : k + 1], dictionary, params, temperature=1e-4
            )
            assert torch.allclose(matched, params[k : k + 1], rtol=1e-2)

    def test_amplitude_is_ignored_because_proton_density_is_not_indexed(self) -> None:
        """Normalisation makes the match about trajectory SHAPE.

        PD scales a fingerprint; the dictionary indexes (T1, T2). A match that
        keyed on amplitude would return a different tissue for the same tissue at
        a different proton density.
        """
        dictionary, params = _dictionary()
        scaled = dictionary[3:4] * 17.0
        matched = soft_dictionary_match(scaled, dictionary, params, temperature=1e-4)
        assert torch.allclose(matched, params[3:4], rtol=1e-2)

    def test_global_phase_is_ignored_for_complex_fingerprints(self) -> None:
        """Receive phase is arbitrary and carries no tissue information."""
        dictionary, params = _dictionary()
        cplx = dictionary.to(torch.complex64)
        rotated = cplx[7:8] * torch.exp(torch.tensor(1j * 0.9))
        matched = soft_dictionary_match(rotated, cplx, params, temperature=1e-4)
        assert torch.allclose(matched, params[7:8], rtol=1e-2)

    def test_it_is_differentiable(self) -> None:
        """The whole reason for the softmax relaxation: argmax has no gradient."""
        dictionary, params = _dictionary()
        query = dictionary[4:5].clone().requires_grad_(True)
        soft_dictionary_match(
            query, dictionary, params, temperature=0.05
        ).sum().backward()
        assert query.grad is not None
        assert torch.isfinite(query.grad).all()
        assert float(query.grad.abs().sum()) > 0

    def test_zero_temperature_raises_rather_than_dividing_by_zero(self) -> None:
        dictionary, params = _dictionary()
        with pytest.raises(ValueError, match="temperature must be > 0"):
            soft_dictionary_match(dictionary[:1], dictionary, params, temperature=0.0)

    def test_fingerprint_length_mismatch_raises(self) -> None:
        dictionary, params = _dictionary(length=32)
        with pytest.raises(ValueError, match="does not match the"):
            soft_dictionary_match(torch.randn(1, 16), dictionary, params)

    def test_dictionary_parameter_row_mismatch_raises(self) -> None:
        """Otherwise the matched parameters come from the wrong tissue."""
        dictionary, params = _dictionary(n_entries=16)
        with pytest.raises(ValueError, match="dictionary_params"):
            soft_dictionary_match(dictionary[:1], dictionary, params[:8])


class TestMRFDictionaryMatchLoss:
    def test_a_prediction_agreeing_with_the_match_scores_zero(self) -> None:
        dictionary, params = _dictionary()
        loss = MRFDictionaryMatchLoss(temperature=1e-4)
        value = loss(
            pred_params=params[2:5],
            fingerprints=dictionary[2:5],
            dictionary=dictionary,
            dictionary_params=params,
        )
        assert float(value) == pytest.approx(0.0, abs=1e-2)

    def test_a_disagreeing_prediction_scores_higher(self) -> None:
        dictionary, params = _dictionary()
        loss = MRFDictionaryMatchLoss(temperature=1e-4)
        good = loss(
            pred_params=params[2:5],
            fingerprints=dictionary[2:5],
            dictionary=dictionary,
            dictionary_params=params,
        )
        bad = loss(
            pred_params=params[10:13],
            fingerprints=dictionary[2:5],
            dictionary=dictionary,
            dictionary_params=params,
        )
        assert float(bad) > float(good)

    def test_it_is_self_supervised_needing_no_ground_truth_maps(self) -> None:
        """The dictionary IS the supervision.

        Real MRF acquisitions have no ground-truth (T1, T2) maps; this is what
        makes the regime trainable on real data, exactly as tofts_residual does
        for perfusion. Asserted by the signature: nothing named target/gt.
        """
        import inspect

        names = set(inspect.signature(MRFDictionaryMatchLoss.forward).parameters)
        assert {
            "pred_params",
            "fingerprints",
            "dictionary",
            "dictionary_params",
        } <= names
        assert not {"target", "ground_truth", "gt_params"} & names

    def test_a_broadcastable_shape_mismatch_raises_rather_than_returning_a_number(
        self,
    ) -> None:
        """[V,1] vs [V,2] would broadcast to a plausible scalar and never warn.

        This is the shape the earlier review caught in the flow strategy: a
        silent broadcast comparing every voxel against every other voxel, giving
        a perfectly ordinary-looking loss.
        """
        dictionary, params = _dictionary()
        loss = MRFDictionaryMatchLoss()
        with pytest.raises(ValueError, match="does not match"):
            loss(
                pred_params=params[2:5, :1],  # [3, 1] against a [3, 2] match
                fingerprints=dictionary[2:5],
                dictionary=dictionary,
                dictionary_params=params,
            )

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight must be >= 0"):
            MRFDictionaryMatchLoss(weight=-1.0)

    def test_no_dictionary_argument_is_optional(self) -> None:
        """There is deliberately no "no dictionary, fall back to L1" branch.

        A fingerprinting loss without a dictionary is not a degraded
        fingerprinting loss — it is a different loss wearing the name, which is
        precisely how beltrami_epi_residual came to report plain L1 under the
        name of physics it was not doing.
        """
        dictionary, params = _dictionary()
        with pytest.raises(TypeError):
            MRFDictionaryMatchLoss()(
                pred_params=params[:1], fingerprints=dictionary[:1]
            )


def test_registered_and_tagged_for_fingerprinting() -> None:
    """It backs mri_fingerprinting's LIVE claim; a lost tag must fail here."""
    from spectramr.models.losses.registry import LossRegistry

    meta = LossRegistry._loss_domains["mrf_dictionary_match"]
    assert meta["workflows"] == frozenset({Regime.FINGERPRINTING})


def test_selectable_from_yaml_via_the_loss_weight_ssot() -> None:
    """Registered is not enough — it must be reachable from a config.

    resolve_loss_weight RAISES on a name with neither a declaration nor a schema
    default, so without `lambda_mrf_dictionary_match` the loss could be
    registered and tagged yet impossible to actually request.
    """
    from spectramr.models.losses.weights import _schema_defaults

    defaults = _schema_defaults()
    assert "mrf_dictionary_match" in defaults
    section, value = defaults["mrf_dictionary_match"]
    assert section == "physics"
    assert value == 0.0  # 0.0 = "not requested"


class TestComplexDtypePromotion:
    """#346: the complex branch used to hardcode ``complex64``.

    The real branch preserves the input dtype implicitly (float64 stays
    float64), so the two paths disagreed about precision purely on whether the
    fingerprints were complex — silently, and exactly where a float64
    dictionary would have been reached for: chasing a matching discrepancy.
    """

    @staticmethod
    def _inputs(dtype, param_dtype):
        return (
            torch.randn(4, 8, dtype=dtype),
            torch.randn(6, 8, dtype=dtype),
            torch.randn(6, 3, dtype=param_dtype),
        )

    def test_complex128_is_not_downcast(self) -> None:
        f, d, p = self._inputs(torch.complex128, torch.float64)
        out = soft_dictionary_match(f, d, p, temperature=0.1)
        assert out.dtype == torch.float64, (
            "complex128 in must not come back float32 — the hardcoded "
            "complex64 cast is what #346 reports"
        )

    def test_complex64_still_yields_float32(self) -> None:
        f, d, p = self._inputs(torch.complex64, torch.float32)
        assert soft_dictionary_match(f, d, p, temperature=0.1).dtype == torch.float32

    def test_complex_and_real_paths_agree_on_precision(self) -> None:
        """The asymmetry the issue names: same width in, same width out."""
        fc, dc, pc = self._inputs(torch.complex128, torch.float64)
        fr, dr, pr = self._inputs(torch.float64, torch.float64)
        complex_out = soft_dictionary_match(fc, dc, pc, temperature=0.1)
        real_out = soft_dictionary_match(fr, dr, pr, temperature=0.1)
        assert complex_out.dtype == real_out.dtype

    def test_mixed_widths_promote_to_the_wider(self) -> None:
        f = torch.randn(4, 8, dtype=torch.complex64)
        d = torch.randn(6, 8, dtype=torch.complex128)
        p = torch.randn(6, 3, dtype=torch.float64)
        assert soft_dictionary_match(f, d, p, temperature=0.1).dtype == torch.float64
