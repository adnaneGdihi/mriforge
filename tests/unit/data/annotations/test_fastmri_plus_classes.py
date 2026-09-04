"""The label map: no 'other' bucket, and not every annotated box is a lesion.

The declared strings are pinned against the SHIPPED ``brain.csv``
(microsoft/fastmri-plus @ 67ed9a6, 7,570 box rows, 21 distinct box labels) -- **not**
against the paper. The two disagree badly, and the paper won until 2026-07-13: ten of the
21 box labels were undeclared (1,692 boxes, 22%), so ``raise_if_unmapped`` fired and no
manifest could be built at all.
"""

from __future__ import annotations

import pytest

from spectramr.data.annotations.fastmri_plus_classes import (
    NON_PARENCHYMAL_GROUPS,
    LesionGroup,
    UnmappedLabelError,
    declared_labels,
    group_for,
    is_poolable,
    normalize_label,
)

# Every distinct label on a BOX row (study_level=No) of the pinned CSV, with its count.
# Read off the file, not the paper. If this list and the CSV ever disagree, the CSV wins.
SHIPPED_BOX_LABELS: dict[str, int] = {
    "Nonspecific white matter lesion": 1826,
    "Posttreatment change": 1262,
    "Craniotomy": 1025,
    "Nonspecific lesion": 757,
    "Possible artifact": 505,
    "Mass": 380,
    "Edema": 369,
    "Dural thickening": 351,
    "Enlarged ventricles": 300,
    "Resection cavity": 199,
    "Encephalomalacia": 161,
    "Lacunar infarct": 113,
    "Extra-axial mass": 104,
    "Normal variant": 73,
    "Craniectomy with Cranioplasty": 43,
    "Paranasal sinus opacification": 40,
    "Craniectomy": 32,
    "Likely cysts": 17,
    "Intraventricular substance": 8,
    "Absent septum pellucidum": 3,
    "Pineal cyst": 2,
}


class TestEveryShippedLabelIsDeclared:
    """The reconciliation that had never been run.

    If any of these raises, the manifest build dies on the real CSV -- which is exactly
    what happened, silently, until someone actually downloaded the file.
    """

    @pytest.mark.parametrize("label", sorted(SHIPPED_BOX_LABELS))
    def test_the_label_maps_to_a_group(self, label: str) -> None:
        assert isinstance(group_for(label), LesionGroup)

    def test_the_box_counts_sum_to_the_published_total(self) -> None:
        """7,570 box rows -- the number the fastMRI+ paper states. If this drifts, the
        pinned CSV changed under us and every downstream count moved with it."""
        assert sum(SHIPPED_BOX_LABELS.values()) == 7570


class TestTheTaxonomyWasFiction:
    """Regressions for claims the code made that the shipped CSV does not support."""

    def test_craniatomy_is_not_a_real_label(self) -> None:
        """The map declared 'craniatomy' with the comment '# sic -- misspelling in the
        shipped CSV'. There is no such misspelling. The provenance claim was invented, so
        the key is gone -- and an invented label must raise like any other."""
        with pytest.raises(UnmappedLabelError):
            group_for("Craniatomy")

    def test_there_is_no_hemorrhage_group(self) -> None:
        """LesionGroup.HEMORRHAGE existed with two declared strings. The shipped brain CSV
        contains no hemorrhage label of any spelling -- it was a group with no members.
        """
        assert not hasattr(LesionGroup, "HEMORRHAGE")
        with pytest.raises(UnmappedLabelError):
            group_for("Intracranial hemorrhage")

    def test_the_sinus_label_carries_its_real_name(self) -> None:
        """Declared as 'sinus opacification'; the CSV says 'Paranasal sinus
        opacification'. Close enough to read past, and it does not match."""
        assert group_for("Paranasal sinus opacification") is LesionGroup.EXTRACRANIAL


class TestNotEveryBoxIsALesion:
    """The two label classes that would have poisoned the money table."""

    def test_possible_artifact_is_an_artifact_not_a_lesion(self) -> None:
        """505 boxes. sim2rank INJECTS artifacts and ranks metrics on how well they track
        the injected severity -- so pre-existing artifact ROIs inside the 'lesion' arm
        confound the axis with itself."""
        assert group_for("Possible artifact") is LesionGroup.ARTIFACT
        assert not is_poolable(LesionGroup.ARTIFACT)

    def test_normal_variant_is_normal_anatomy(self) -> None:
        assert group_for("Normal variant") is LesionGroup.NORMAL
        assert not is_poolable(LesionGroup.NORMAL)

    def test_extracranial_is_excluded_because_its_mirror_is_not_brain(self) -> None:
        """A sinus opacification's contralateral 'control in normal-appearing tissue' is
        the OTHER SINUS. The paired test would still return a confident number."""
        assert not is_poolable(LesionGroup.EXTRACRANIAL)

    def test_real_pathology_is_poolable(self) -> None:
        for g in (
            LesionGroup.EDEMA,
            LesionGroup.MASS,
            LesionGroup.WHITE_MATTER,
            LesionGroup.INFARCT,
            LesionGroup.NONSPECIFIC,
        ):
            assert is_poolable(g)

    def test_postsurgical_is_not_excluded_by_label(self) -> None:
        """Deliberately. The group is mixed: a craniotomy sits on the skull flap (mirror =
        intact skull, useless) but a resection cavity sits in parenchyma (mirror = normal
        brain, exactly right). Same label, opposite answers -- so it is settled by MEASURED
        brain coverage in the source, not guessed here."""
        assert is_poolable(LesionGroup.POSTSURGICAL)

    def test_non_parenchymal_groups_are_exactly_the_three(self) -> None:
        expected = {LesionGroup.EXTRACRANIAL, LesionGroup.ARTIFACT, LesionGroup.NORMAL}
        assert set(NON_PARENCHYMAL_GROUPS) == expected


class TestNoOtherBucket:
    def test_an_undeclared_label_raises(self) -> None:
        """`.get(label, 'other')` would pool an undeclared pathology into a bucket whose
        clinical meaning nobody can state, and it would do so silently."""
        with pytest.raises(UnmappedLabelError, match="Wormholes"):
            group_for("Wormholes")

    def test_the_error_says_exactly_how_to_fix_it(self) -> None:
        with pytest.raises(UnmappedLabelError, match="_LABEL_TO_GROUP"):
            group_for("Some New Finding")

    def test_no_group_is_named_other(self) -> None:
        assert "other" not in {g.value for g in LesionGroup}


class TestNormalisation:
    def test_case_and_whitespace_are_normalised(self) -> None:
        assert group_for("  EDEMA  ") is LesionGroup.EDEMA
        assert group_for("Edema") is LesionGroup.EDEMA
        assert normalize_label("Extra-Axial   Collection") == "extra-axial collection"

    def test_spelling_is_not_repaired(self) -> None:
        """Repairing a typo means guessing what a human meant -- the judgement the raise
        exists to escalate. A misspelling that silently maps to a real class is how one
        class quietly becomes two underpowered halves."""
        with pytest.raises(UnmappedLabelError):
            group_for("Edmea")


class TestGroupSemantics:
    def test_every_declared_label_maps_to_a_real_group(self) -> None:
        for label in declared_labels():
            assert isinstance(group_for(label), LesionGroup)

    def test_nonspecific_lesion_is_not_folded_into_white_matter(self) -> None:
        """757 boxes. The annotator deliberately did NOT say white matter; inventing it
        for them would inflate the WM group by 40%."""
        assert group_for("Nonspecific lesion") is LesionGroup.NONSPECIFIC
        assert group_for("Nonspecific white matter lesion") is LesionGroup.WHITE_MATTER
