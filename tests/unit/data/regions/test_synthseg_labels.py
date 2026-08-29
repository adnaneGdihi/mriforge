"""SynthSeg label map -> tissue ordinals, and the raise on an undeclared label."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.data.regions.synthseg_labels import (  # noqa: E402
    DK_PARCELLATION_LABELS,
    LABEL_TO_TISSUE,
    TissueClass,
    UnmappedSynthSegLabelError,
    brain_mask,
    label_map_to_tissue,
    structure_mask,
    tissue_mask,
)


class TestTheParcellationLabelsAreCortex:
    """Under ``--parc`` SynthSeg STOPS emitting the whole-cortex labels 3/42 and emits
    the 68 Desikan-Killiany parcels instead. A table that knows 3/42 but not 1001+ either
    raises on every parcel, or -- if someone "fixes" that by bucketing them to background
    -- silently deletes the ENTIRE CORTEX from ``brain()`` while the run goes on producing
    confident numbers.
    """

    def test_every_dk_parcel_is_cortical_gm(self) -> None:
        assert len(DK_PARCELLATION_LABELS) == 70  # 1001-1035 and 2001-2035
        for label in DK_PARCELLATION_LABELS:
            assert LABEL_TO_TISSUE[label] is TissueClass.CORTICAL_GM

    def test_brain_is_identical_with_and_without_parc(self) -> None:
        """The invariant. The same anatomy segmented with --parc must give the same
        brain mask and the same cortical_gm mask -- only the RESOLUTION of the label ids
        changes, never the tissue they resolve to."""
        plain = torch.tensor([[0, 3, 42, 2]])
        parc = torch.tensor([[0, 1001, 2035, 2]])  # cortex, now parcellated

        assert torch.equal(
            brain_mask(label_map_to_tissue(plain)),
            brain_mask(label_map_to_tissue(parc)),
        )
        assert torch.equal(
            tissue_mask(label_map_to_tissue(plain), TissueClass.CORTICAL_GM),
            tissue_mask(label_map_to_tissue(parc), TissueClass.CORTICAL_GM),
        )

    def test_a_parc_label_does_not_fit_in_a_uint8(self) -> None:
        """Why the cache had to move to uint16 -- not an optimisation, a
        representability requirement."""
        assert max(DK_PARCELLATION_LABELS) > 255


class TestStructureMask:
    def test_it_unions_raw_freesurfer_ids(self) -> None:
        labels = torch.tensor([[17, 53, 10, 0]])
        assert torch.equal(
            structure_mask(labels, [17, 53]), torch.tensor([[True, True, False, False]])
        )

    def test_it_separates_what_the_tissue_classes_merge(self) -> None:
        """Thalamus (10) and hippocampus (17) are BOTH subcortical_gm. The tissue view
        cannot tell them apart; the raw-label view can, which is the entire point of the
        v2 cache."""
        labels = torch.tensor([[10, 17]])
        ords = label_map_to_tissue(labels)

        assert torch.equal(
            tissue_mask(ords, TissueClass.SUBCORTICAL_GM), torch.tensor([[True, True]])
        )
        assert torch.equal(structure_mask(labels, [17]), torch.tensor([[False, True]]))

    def test_it_needs_at_least_one_label(self) -> None:
        with pytest.raises(ValueError, match="at least one FreeSurfer label"):
            structure_mask(torch.zeros(2, 2, dtype=torch.int64), [])


class TestTheLookupIsVectorisedAndStillStrict:
    def test_a_label_above_the_table_raises_rather_than_indexing_out_of_bounds(
        self,
    ) -> None:
        """The LUT gather must not be allowed to silently wrap or crash on an id larger
        than the table: an undeclared label is a region nobody can name."""
        with pytest.raises(UnmappedSynthSegLabelError):
            label_map_to_tissue(torch.tensor([[99999]]))

    def test_a_negative_label_raises(self) -> None:
        with pytest.raises(UnmappedSynthSegLabelError):
            label_map_to_tissue(torch.tensor([[-1]]))

    def test_a_gap_inside_the_table_raises(self) -> None:
        """Id 9 sits between declared ids and is NOT a SynthSeg output -- the LUT holds a
        sentinel there, not a default."""
        with pytest.raises(UnmappedSynthSegLabelError):
            label_map_to_tissue(torch.tensor([[9]]))

    def test_an_empty_map_does_not_raise(self) -> None:
        assert label_map_to_tissue(torch.zeros(0, dtype=torch.int64)).numel() == 0


class TestLabelMapping:
    def test_left_and_right_fold_into_one_class(self) -> None:
        """The region axis asks whether a metric behaves differently in WM than in
        GM -- not differently in LEFT WM. Laterality is still recoverable from the
        column centroid, which is what the contralateral control uses."""
        assert LABEL_TO_TISSUE[2] is LABEL_TO_TISSUE[41] is TissueClass.CEREBRAL_WM
        assert LABEL_TO_TISSUE[3] is LABEL_TO_TISSUE[42] is TissueClass.CORTICAL_GM

    def test_ventricles_are_csf(self) -> None:
        for fs_id in (4, 43, 14, 15, 24):
            assert LABEL_TO_TISSUE[fs_id] is TissueClass.CSF

    def test_conversion_produces_uint8_ordinals(self) -> None:
        labels = torch.tensor([[0, 2], [3, 4]], dtype=torch.int64)
        out = label_map_to_tissue(labels)
        assert out.dtype is torch.uint8
        assert out[0, 0] == TissueClass.BACKGROUND.ordinal
        assert out[0, 1] == TissueClass.CEREBRAL_WM.ordinal
        assert out[1, 0] == TissueClass.CORTICAL_GM.ordinal
        assert out[1, 1] == TissueClass.CSF.ordinal

    def test_ordinals_are_stable(self) -> None:
        """They are written into the cache; reordering them would silently
        reinterpret every cached volume."""
        assert TissueClass.BACKGROUND.ordinal == 0
        assert TissueClass.CORTICAL_GM.ordinal == 1
        assert TissueClass.CEREBRAL_WM.ordinal == 2
        assert TissueClass.CSF.ordinal == 3


class TestUndeclaredLabelRaises:
    def test_an_unknown_freesurfer_id_raises(self) -> None:
        """Silently bucketing it would create a region whose anatomical meaning
        nobody can state."""
        with pytest.raises(UnmappedSynthSegLabelError, match="999"):
            label_map_to_tissue(torch.tensor([[0, 999]], dtype=torch.int64))

    def test_the_error_names_the_likely_cause(self) -> None:
        with pytest.raises(UnmappedSynthSegLabelError, match="out of distribution"):
            label_map_to_tissue(torch.tensor([[777]], dtype=torch.int64))


class TestMasks:
    def test_tissue_mask_unions_its_classes(self) -> None:
        ords = torch.tensor(
            [
                [TissueClass.CORTICAL_GM.ordinal, TissueClass.CEREBRAL_WM.ordinal],
                [TissueClass.SUBCORTICAL_GM.ordinal, TissueClass.BACKGROUND.ordinal],
            ],
            dtype=torch.uint8,
        )
        gm = tissue_mask(ords, TissueClass.CORTICAL_GM, TissueClass.SUBCORTICAL_GM)
        assert gm.tolist() == [[True, False], [True, False]]

    def test_brain_mask_is_everything_not_background(self) -> None:
        ords = torch.tensor(
            [[TissueClass.BACKGROUND.ordinal, TissueClass.CSF.ordinal]],
            dtype=torch.uint8,
        )
        assert brain_mask(ords).tolist() == [[False, True]]

    def test_tissue_mask_needs_at_least_one_class(self) -> None:
        with pytest.raises(ValueError, match="at least one TissueClass"):
            tissue_mask(torch.zeros(2, 2, dtype=torch.uint8))
