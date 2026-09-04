"""Unit tests for the tissue-property registry.

Targets :mod:`spectramr.infrastructure.physics.tissue_properties`. The
registry maps ``(field_strength, tissue_type)`` pairs to relaxation
constants and converts a segmentation map to a per-tissue PD/T1/T2
field map used by the Bloch simulator.

The tests pin:

* The published T1/T2/PD values for each (field, tissue) pair — these
  are derived from literature and must not drift silently
* ``get_properties`` raises on unknown field strengths (CLAUDE.md #9)
* ``get_map_tensor`` converts ``(B,1,H,W)`` int segs to ``(B,3,H,W)``
  float maps with the documented PD/T1/T2 channel ordering
* Background pixels stay at zero
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.tissue_properties import (
    TissueFieldStrength,
    RelaxationProperties,
    TissuePropertyRegistry,
    TissueType,
)


# ── Enums and dataclasses ──────────────────────────────────────────


class TestEnums:
    def test_tissue_type_integer_values(self) -> None:
        # The seg-map integers feed directly into mask comparisons so
        # they must be stable.
        assert TissueType.BACKGROUND.value == 0
        assert TissueType.CSF.value == 1
        assert TissueType.GM.value == 2
        assert TissueType.WM.value == 3

    def test_field_strength_values_are_strings(self) -> None:
        assert TissueFieldStrength.mT64.value == "64mT"
        assert TissueFieldStrength.T1_5.value == "1.5T"
        assert TissueFieldStrength.T3_0.value == "3.0T"

    def test_relaxation_properties_is_frozen(self) -> None:
        p = RelaxationProperties(t1=1.0, t2=2.0, pd=0.5)
        with pytest.raises((AttributeError, TypeError)):
            # frozen=True must reject post-construction mutation.
            p.t1 = 999.0  # type: ignore[misc]


# ── get_properties ─────────────────────────────────────────────────


class TestGetProperties:
    @pytest.mark.parametrize(
        "field, t1_csf",
        [
            ("64mT", 4000.0),
            ("1.5T", 4000.0),
            ("3.0T", 4000.0),
        ],
    )
    def test_csf_t1_constant_across_fields(self, field: str, t1_csf: float) -> None:
        # CSF T1 is documented as ≈4000 ms across the three field
        # strengths in this registry.
        props = TissuePropertyRegistry.get_properties(field)
        assert props[TissueType.CSF].t1 == t1_csf

    def test_white_matter_t1_increases_with_field(self) -> None:
        wm_64 = TissuePropertyRegistry.get_properties("64mT")[TissueType.WM].t1
        wm_15 = TissuePropertyRegistry.get_properties("1.5T")[TissueType.WM].t1
        wm_3 = TissuePropertyRegistry.get_properties("3.0T")[TissueType.WM].t1
        # Monotone with field — captures the literature trend.
        assert wm_64 < wm_15 < wm_3

    def test_gm_t1_increases_with_field(self) -> None:
        gm_64 = TissuePropertyRegistry.get_properties("64mT")[TissueType.GM].t1
        gm_15 = TissuePropertyRegistry.get_properties("1.5T")[TissueType.GM].t1
        gm_3 = TissuePropertyRegistry.get_properties("3.0T")[TissueType.GM].t1
        assert gm_64 < gm_15 < gm_3

    def test_background_properties_are_zero(self) -> None:
        for field in ("64mT", "1.5T", "3.0T"):
            bg = TissuePropertyRegistry.get_properties(field)[TissueType.BACKGROUND]
            assert bg.t1 == 0.0
            assert bg.t2 == 0.0
            assert bg.pd == 0.0

    def test_enum_form_equivalent_to_string(self) -> None:
        a = TissuePropertyRegistry.get_properties(TissueFieldStrength.mT64)
        b = TissuePropertyRegistry.get_properties("64mT")
        assert a is b  # same dict reference

    def test_unknown_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown field strength"):
            TissuePropertyRegistry.get_properties("0.1T")


# ── get_map_tensor ─────────────────────────────────────────────────


class TestGetMapTensor:
    def test_output_shape_is_B3HW(self) -> None:
        seg = torch.zeros(2, 1, 8, 8, dtype=torch.long)
        out = TissuePropertyRegistry.get_map_tensor(seg)
        # 3 channels: PD, T1, T2.
        assert out.shape == (2, 3, 8, 8)

    def test_background_stays_zero(self) -> None:
        seg = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        out = TissuePropertyRegistry.get_map_tensor(seg)
        # All voxels are BG → output is all zeros.
        assert torch.all(out == 0)

    def test_wm_voxels_receive_wm_properties_64mT(self) -> None:
        seg = torch.full((1, 1, 4, 4), TissueType.WM.value, dtype=torch.long)
        out = TissuePropertyRegistry.get_map_tensor(seg, field_strength="64mT")
        wm = TissuePropertyRegistry.get_properties("64mT")[TissueType.WM]
        # PD channel.
        assert torch.allclose(out[:, 0], torch.full((1, 4, 4), wm.pd))
        # T1 channel.
        assert torch.allclose(out[:, 1], torch.full((1, 4, 4), wm.t1))
        # T2 channel.
        assert torch.allclose(out[:, 2], torch.full((1, 4, 4), wm.t2))

    def test_mixed_tissue_segmentation(self) -> None:
        seg = torch.tensor(
            [
                [[0, 1], [2, 3]],
            ],
            dtype=torch.long,
        ).unsqueeze(0)
        # Shape: (1, 1, 2, 2).
        out = TissuePropertyRegistry.get_map_tensor(seg, field_strength="3.0T")
        props = TissuePropertyRegistry.get_properties("3.0T")
        # T1 values per location.
        expected_t1 = torch.tensor([
            [props[TissueType.BACKGROUND].t1, props[TissueType.CSF].t1],
            [props[TissueType.GM].t1, props[TissueType.WM].t1],
        ])
        assert torch.allclose(out[0, 1], expected_t1)

    def test_get_map_tensor_unknown_field_raises(self) -> None:
        seg = torch.zeros(1, 1, 2, 2, dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown field strength"):
            TissuePropertyRegistry.get_map_tensor(seg, field_strength="0.1T")
