"""fastMRI+ raw label -> coarse lesion group.

Why a raise and not ``.get(label, "other")``
--------------------------------------------
An ``"other"`` bucket is a silent fallback (pitfall #9) wearing a taxonomy. A
label the map has never seen is one of exactly two things -- a pathology class
nobody declared, or a **typo/whitespace variant of a class that IS declared**
(the published fastMRI+ CSV is known to carry at least one misspelling). Both
must stop the build:

* an undeclared class silently pooled into ``other`` shows up in the leaderboard
  as a region whose clinical meaning nobody can state;
* a typo variant silently *splits* one class in two, halving the n of both and
  quietly making every per-group cell underpowered.

So :func:`group_for` raises, and the fix is one line in ``_LABEL_TO_GROUP``.

Reconciled against the real CSV (2026-07-13)
--------------------------------------------
The strings below are **read off the shipped ``brain.csv``**, at the pinned upstream
commit (``scripts/data/fetch_fastmri_plus_annotations.py``): 7,570 box rows carrying
**21 distinct labels**.

They previously came from the *paper*, not the file, and the two disagree badly. Ten of
the 21 box labels were undeclared -- 1,692 boxes, 22% of the corpus -- so
:meth:`~mriforge.data.annotations.fastmri_plus.AnnotationParseReport.raise_if_unmapped`
fired and **no manifest could be built at all**. Worse, several *declared* strings do not
exist in the CSV, including ``"craniatomy"``, which carried the comment
``# sic -- misspelling in the shipped CSV``. There is no such misspelling. That was a
fabricated provenance claim, and it is the reason this module now cites a commit.

The reconciliation workflow the docstring always demanded, and which had never been run::

    python scripts/data/fetch_fastmri_plus_annotations.py --out <brain.csv>
    python scripts/data/build_fastmri_plus_manifest.py --annotations <brain.csv> --report-labels

which prints every distinct label with its count and never raises. The raise in this
module is what forces that reconciliation to happen *before* a manifest is built, rather
than after a leaderboard has been published.

Not every annotated box is a lesion
------------------------------------
Two of the CSV's box labels are not pathology at all, and pooling them into
``path:lesion_any`` would have been a quiet catastrophe:

* ``"Possible artifact"`` -- **505 boxes** -- is an *imaging artifact*. sim2rank's entire
  method is to inject artifacts and rank metrics on how well they track the injected
  severity. Putting 505 pre-existing artifact regions inside the "lesion" arm confounds
  the axis with itself.
* ``"Normal variant"`` -- **73 boxes** -- is, explicitly, *normal anatomy*.

They are declared (so the parser does not raise) and grouped into
:attr:`LesionGroup.ARTIFACT` / :attr:`LesionGroup.NORMAL`, which
:data:`NON_PARENCHYMAL_GROUPS` keeps out of the pooled lesion region. Discarding them at
parse time instead would have hidden them; the point is that they are *counted and
visible* in the manifest's provenance, and *excluded* from the region.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

__all__ = [
    "NON_PARENCHYMAL_GROUPS",
    "LesionGroup",
    "UnmappedLabelError",
    "declared_labels",
    "group_for",
    "is_poolable",
    "normalize_label",
]


class LesionGroup(StrEnum):
    """Coarse grouping of fastMRI+ brain pathology labels.

    The sweep's default lesion region is the pooled union of every box on a
    slice (``lesion_any``); these groups exist for the **opt-in** per-group
    regions. With ~16 raw classes x 4 contrasts, most per-class cells hold
    fewer than 30 lesions -- a per-class default would be underpowered by
    construction, which is why pooling is the default and grouping is coarse.
    """

    MASS = "mass"
    """Space-occupying lesions: mass, pineal cyst, likely cysts."""

    WHITE_MATTER = "white_matter"
    """Nonspecific / age-related white-matter signal abnormality. The largest group."""

    NONSPECIFIC = "nonspecific"
    """``"Nonspecific lesion"`` -- a real finding the radiologist declined to classify.

    757 boxes: the fourth-largest group, and it has no analogue in the paper's taxonomy.
    Kept as its own group rather than folded into ``WHITE_MATTER``: the annotator
    deliberately did **not** say white matter, and inventing that for them would silently
    inflate the WM group by 40%.
    """

    INFARCT = "infarct"
    """Lacunar infarct."""

    EDEMA = "edema"
    """Vasogenic or cytotoxic oedema."""

    ENCEPHALOMALACIA = "encephalomalacia"
    """Established tissue loss / gliosis."""

    VENTRICULAR = "ventricular"
    """Ventricular enlargement, intraventricular material, midline variants."""

    POSTSURGICAL = "postsurgical"
    """Craniotomy, craniectomy (+/- cranioplasty), resection cavity, posttreatment change.

    **Mixed by anatomy, deliberately.** A craniotomy box sits on the skull flap; a
    resection cavity sits in brain parenchyma. That distinction is invisible to a label
    and decides whether a contralateral-mirror control is meaningful, so it is settled by
    *measurement* downstream -- see :data:`NON_PARENCHYMAL_GROUPS`.
    """

    EXTRA_AXIAL = "extra_axial"
    """Collections, masses and dural findings outside the brain parenchyma."""

    ARTIFACT = "artifact"
    """``"Possible artifact"`` -- an IMAGING artifact, not a pathology. **505 boxes.**

    Never pooled into the lesion region. sim2rank *injects* artifacts and ranks metrics on
    how well they track the injected severity; a "lesion" arm containing 505 pre-existing
    artifact regions confounds the axis with itself.
    """

    NORMAL = "normal"
    """``"Normal variant"`` -- explicitly normal anatomy. **73 boxes.**

    Never pooled into the lesion region, for the obvious reason.
    """

    EXTRACRANIAL = "extracranial"
    """Findings outside the cranial vault (e.g. sinus opacification).

    Kept as its own group deliberately: these boxes do **not** sit on brain
    tissue, so they are the one group for which the contralateral-mirror
    matched control is meaningless. Pooling them with parenchymal lesions
    would quietly corrupt the ``lesion_vs_control`` comparison.

    This is enforced -- see :data:`NON_PARENCHYMAL_GROUPS`. Until 2026-07-13 it was
    only *declared*, and the source pooled these boxes anyway.
    """


NON_PARENCHYMAL_GROUPS: frozenset[LesionGroup] = frozenset(
    {
        LesionGroup.EXTRACRANIAL,  # 40 boxes: outside the cranial vault
        LesionGroup.ARTIFACT,  # 505 boxes: an imaging artifact, not a finding
        LesionGroup.NORMAL,  # 73 boxes: explicitly normal anatomy
    }
)
"""Groups whose boxes may never enter ``path:lesion_any``. **618 boxes, 8% of the corpus.**

Two distinct reasons, both fatal to the money table:

* **It is not a lesion.** ``ARTIFACT`` ("Possible artifact") and ``NORMAL`` ("Normal
  variant") are, by the annotator's own word, not pathology. A lesion region containing
  them measures nothing. The artifact case is worse than merely useless: sim2rank
  *injects* artifacts and ranks metrics on how well they track injected severity, so
  pre-existing artifact ROIs inside the lesion arm confound the axis with itself.
* **Its mirror is not a control.** ``EXTRACRANIAL`` (sinus opacification) does not sit on
  brain, so its contralateral mirror is *the other sinus* -- and ``lesion_vs_control``
  then compares two non-brain ROIs and still returns a confident number.

**Why ``POSTSURGICAL`` is NOT in here**, despite being the obvious candidate: the group is
genuinely *mixed*. A craniotomy box sits on the skull flap (mirror = intact skull,
useless); a resection cavity sits in brain parenchyma (mirror = normal brain, exactly
right). Excluding the group would throw away real parenchymal lesions; keeping it whole
would smuggle skull boxes in.

That is not a call a *label* can make, so this table does not try. The source applies a
**measured** gate instead -- a box whose SynthSeg brain coverage falls below
``MIN_LESION_BRAIN_COVERAGE`` is not a parenchymal lesion whatever it is called (see
:mod:`mriforge.data.sources.fastmri_brain_source`). Measurement separates the craniotomy
from the resection cavity; a taxonomy cannot.
"""


def is_poolable(group: LesionGroup) -> bool:
    """May this group's boxes enter the pooled ``path:lesion_any`` region?

    A label-level gate only. It cannot see where the box actually landed, so it is
    necessary but **not sufficient** -- the source additionally gates on measured brain
    coverage when a SynthSeg cache is present.
    """
    return group not in NON_PARENCHYMAL_GROUPS


# Provenance: read off Annotations/brain.csv at microsoft/fastmri-plus @ 67ed9a6
# (sha256 38c8c9d9..., fetched 2026-07-13). Every one of the 21 distinct BOX labels is
# here; the trailing comment is that label's box count in the shipped CSV.
#
# Study-level rows (study_level=Yes, empty geometry) carry a further 9 labels -- "Normal
# for age", "Motion artifact", "Global label: ..." and friends. They are dropped before
# they ever reach group_for, so they are deliberately NOT declared: if one ever shows up
# as a *box*, that is a change in the data and it should stop the build.
_LABEL_TO_GROUP: Mapping[str, LesionGroup] = {
    "nonspecific white matter lesion": LesionGroup.WHITE_MATTER,  # 1826
    "posttreatment change": LesionGroup.POSTSURGICAL,  # 1262
    "craniotomy": LesionGroup.POSTSURGICAL,  # 1025
    "nonspecific lesion": LesionGroup.NONSPECIFIC,  # 757
    "possible artifact": LesionGroup.ARTIFACT,  # 505 -- NOT pathology
    "mass": LesionGroup.MASS,  # 380
    "edema": LesionGroup.EDEMA,  # 369
    "dural thickening": LesionGroup.EXTRA_AXIAL,  # 351
    "enlarged ventricles": LesionGroup.VENTRICULAR,  # 300
    "resection cavity": LesionGroup.POSTSURGICAL,  # 199
    "encephalomalacia": LesionGroup.ENCEPHALOMALACIA,  # 161
    "lacunar infarct": LesionGroup.INFARCT,  # 113
    "extra-axial mass": LesionGroup.EXTRA_AXIAL,  # 104
    "normal variant": LesionGroup.NORMAL,  # 73 -- NOT pathology
    "craniectomy with cranioplasty": LesionGroup.POSTSURGICAL,  # 43
    "paranasal sinus opacification": LesionGroup.EXTRACRANIAL,  # 40
    "craniectomy": LesionGroup.POSTSURGICAL,  # 32
    "likely cysts": LesionGroup.MASS,  # 17
    "intraventricular substance": LesionGroup.VENTRICULAR,  # 8
    "absent septum pellucidum": LesionGroup.VENTRICULAR,  # 3
    "pineal cyst": LesionGroup.MASS,  # 2
    # Study-level-only in the pinned CSV, but a plausible box label and previously
    # declared -- kept so a future export does not trip the gate on a known class.
    "extra-axial collection": LesionGroup.EXTRA_AXIAL,  # 0 boxes (9 study-level)
}


class UnmappedLabelError(KeyError):
    """A fastMRI+ label with no declared :class:`LesionGroup`."""

    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__(
            f"fastMRI+ label {label!r} has no declared LesionGroup. This is not a "
            "bug to route around: either it is a pathology class nobody declared, "
            "or it is a spelling variant of a class that IS declared (which would "
            "silently split one class into two underpowered halves). Run "
            "`build_fastmri_plus_manifest.py --report-labels` to see every label in "
            "your CSV with its count, then add this one to `_LABEL_TO_GROUP` in "
            "mriforge/data/annotations/fastmri_plus_classes.py. There is deliberately "
            "no 'other' bucket."
        )


def normalize_label(label: str) -> str:
    """Casefold + collapse whitespace. Does **not** repair spelling.

    Normalising case and stray whitespace is safe -- ``"Edema "`` and ``"edema"``
    are the same class by anyone's reading. Repairing spelling is not: it would
    be this module guessing what a human meant, which is exactly the judgement
    the raise exists to escalate.
    """
    return " ".join(label.split()).casefold()


def group_for(label: str) -> LesionGroup:
    """Map one raw CSV label to its group, or raise :class:`UnmappedLabelError`."""
    try:
        return _LABEL_TO_GROUP[normalize_label(label)]
    except KeyError:
        raise UnmappedLabelError(label) from None


def declared_labels() -> frozenset[str]:
    """Every normalised label the map knows. Used by the coverage test."""
    return frozenset(_LABEL_TO_GROUP)
