"""The production-plan row classifier, watched failing on every shape it claims to catch.

``TODO/production_plan/tools/row_status.py`` decides, for each of the 249 fix rows,
whether an implementer is told to execute it, rewrite it, or leave it alone.  It had no
tests: it lived inside ``gen_exclusions.py``, which reads ``sys.argv`` at module scope
and so cannot be imported.  Splitting it out was the precondition for testing it at all.

Every case below is a **planted** status string, and each precedence case is one that a
plausible wrong ordering misfiles (non-negotiable 15).  The precedence cases matter more
than the plain ones: a real status almost always matches two patterns, so the classes are
decided by *order*, and an order is exactly what a single-pattern test cannot see.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
TOOLS = REPO / "TODO/production_plan/tools"


def _load():
    """Import row_status.py by path -- ``TODO/`` is not a package, by design."""
    spec = importlib.util.spec_from_file_location("row_status", TOOLS / "row_status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not (TOOLS / "row_status.py").exists(),
    reason="TODO/ is excluded from the public branch; the classifier is dev-only",
)


class TestEachClassIsReached:
    """One plant per class.  A class no string reaches is a class that does nothing."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("**filed #1406**", "skip"),
            ("top of D04", "skip"),
            ("R9 ANCHOR CONFIRMED at settings.py:562", "skip"),
            ("**FIXED (PR #1384)** — the raise now fires", "done"),
            ("**GATE DONE / KNOBS PART-DONE (PR #1411)** — 8 of 10 wired", "part"),
            ("**FIX REFUTED (R2)** — the proposal rebuilds MODEL_REGISTRY", "refuted"),
            ("**COUNT CORRECTED (R8)** — 28 auto-fixable, not ~60", "corrected"),
            ("no verdict keyword at all", "corrected"),
        ],
    )
    def test_it_classifies(self, status: str, expected: str) -> None:
        assert _load().classify(status) == expected


class TestPrecedenceIsLoadBearing:
    """Real statuses match several patterns; the class is decided by order.

    Each case here is misfiled by the obvious wrong ordering, and each misfile has a
    concrete cost stated in the assertion message.
    """

    def test_a_delivered_row_that_also_carries_a_correction_is_delivered(self) -> None:
        s = "**FIXED (PR #1384)** — the row's stated COUNT CORRECTED to 3 sites"
        assert _load().classify(s) == "done", (
            "checked after 'corrected', a shipped row lands in the re-size table and "
            "gets re-executed"
        )

    def test_a_part_done_row_that_also_carries_a_correction_is_part(self) -> None:
        s = "**GATE DONE / KNOBS PART-DONE (PR #1411)** — mechanism CORRECTED, count too"
        assert _load().classify(s) == "part", (
            "this is the misfile that motivated the class: a part-built row reading as "
            "'the fix stands, a number does not'"
        )

    def test_a_refuted_row_that_also_carries_a_correction_is_refuted(self) -> None:
        s = "**FIX REFUTED (R2)** — and the CORRECTED figure would have been 2, not 14"
        assert _load().classify(s) == "refuted", (
            "the worst misfile in the set: an unexecutable proposal presented as merely "
            "mis-counted, so it gets executed as written"
        )

    def test_a_part_done_row_whose_other_half_was_refuted_is_still_part(self) -> None:
        """A row can ship one half and have the other half struck.

        Ordering ``refuted`` ahead of ``part`` sends it to the do-not-execute table and
        the delivered half gets rebuilt.  Found by a surviving mutant, not by review.
        """
        s = "**GATE DONE / KNOBS PART-DONE (PR #1411)** — the k-space half is FIX REFUTED"
        assert _load().classify(s) == "part"

    def test_part_done_is_not_swallowed_by_delivered(self) -> None:
        s = "**GATE DONE / KNOBS PART-DONE (PR #1411)**"
        assert _load().classify(s) != "done", (
            "'do not re-execute' on a part-built row leaves the remainder unbuilt"
        )

    def test_a_filed_issue_never_reads_as_delivery(self) -> None:
        """Reading `filed #N` as progress under-reported Wave 0 as 5 done of 20."""
        assert _load().classify("**filed #1409**") == "skip"


class TestTheCanonicalSpellingsAreRequired:
    """The prefixes are anchored, so a mention inside prose cannot claim a class.

    The anchoring is ``re.match``'s, not a ``^`` in the pattern -- carrying both made it
    undetectable, because no single edit could unanchor it.  These two tests now die on
    a ``.match`` -> ``.search`` change, which is the edit that would actually break it.
    """

    def test_delivered_must_lead_the_string(self) -> None:
        """The prose must not open with a skip prefix, or it never reaches DELIVERED.

        The first version of this plant read ``"see also **FIXED (PR #1)**..."`` and a
        ``.match`` -> ``.search`` mutant survived it: ``"see "`` is a SKIP prefix, so the
        string was classed ``skip`` two branches earlier and the assertion held for the
        wrong reason.  A plant must reach the code it targets.
        """
        assert _load().classify("context: **FIXED (PR #1)** covers the sibling") != "done"

    def test_a_trailing_delivered_marker_is_caught_by_the_tripwire(self) -> None:
        """The other half of the case above, once the tripwire exists (#1429).

        This plant used to read ``"... **FIXED (PR #1)** shipped the sibling"`` and
        assert ``!= "done"``.  It still would not be ``done`` -- but it now raises
        instead of returning, because "shipped" is a delivery claim with no canonical
        prefix, which is precisely the misfile the tripwire exists for.  Split in two
        rather than reworded away: the anchoring proof above must stay free of delivery
        vocabulary so it keeps killing the ``.match`` -> ``.search`` mutant on its own,
        and this half pins the interaction that made the original ambiguous.
        """
        with pytest.raises(ValueError, match="canonical delivery prefix"):
            _load().classify("context: **FIXED (PR #1)** shipped the sibling")

    def test_part_done_must_lead_the_string(self) -> None:
        """Delimiters included -- without ``**`` the pattern cannot match either way."""
        assert _load().classify("blocked; sibling is **PART-DONE (PR #1)**") != "part"

    def test_part_done_must_sit_inside_the_bold_prefix(self) -> None:
        """``[^*]*``, not ``.*``: the marker cannot be claimed from later prose.

        With a greedy ``.*`` the pattern spans the closing ``**`` of the real prefix and
        picks up a PART-DONE mentioned anywhere afterwards, so a row that merely refers
        to a sibling's partial delivery is filed as partly delivered itself.
        """
        s = "**COUNT CORRECTED (R8)** — the sibling row is PART-DONE (PR #2)**"
        assert _load().classify(s) == "corrected"

    def test_part_done_must_name_a_pr(self) -> None:
        assert _load().classify("**PART-DONE**") != "part", (
            "an unattributable part-delivery claim is not auditable"
        )

    def test_delivered_must_name_a_pr(self) -> None:
        assert _load().classify("**FIXED**") != "done"


class TestTheTwoDeliveryClassesAreDisjoint:
    """Why the ``done``-before-``part`` order is free, stated rather than assumed.

    Reordering those two survives every mutation, which is not a gap in the plants: no
    string can match both, so the order genuinely carries no information.  That is worth
    a test rather than a silence -- if a future spelling makes them overlap, the order
    becomes load-bearing and this goes red first.
    """

    CANDIDATES = (
        "**FIXED (PR #1)** — shipped",
        "**GATE DONE / KNOBS PART-DONE (PR #1411)** — 8 of 10",
        "**FIXED / PART-DONE (PR #1)**",
        "**PART-DONE, then FIXED (PR #1)**",
    )

    @pytest.mark.parametrize("status", CANDIDATES)
    def test_no_status_matches_both_delivery_patterns(self, status: str) -> None:
        mod = _load()
        both = bool(mod.DELIVERED.match(status)) and bool(mod.PART_DELIVERED.match(status))
        assert not both, (
            f"{status!r} matches both delivery patterns, so classify()'s ordering "
            "between them now decides the answer and must be defended by a test"
        )


class TestAgainstTheRealAnnotations:
    """Plants prove the predicate; only the real corpus proves it is wired to anything."""

    @staticmethod
    def _statuses() -> list[tuple[str, str]]:
        ann = json.loads((TOOLS / "annotations.json").read_text())
        return [
            (f"{dossier.split('_')[0]}#{rid}", st)
            for dossier, body in sorted(ann.items())
            for rid, st in body.get("status", {}).items()
        ]

    def test_every_real_status_classifies(self) -> None:
        classify = _load().classify
        valid = {"skip", "done", "part", "refuted", "corrected"}
        assert {classify(st) for _, st in self._statuses()} <= valid

    def test_the_corpus_reaches_more_than_the_default_class(self) -> None:
        """Anti-vacuity: if everything fell through, the classifier would be decoration."""
        classify = _load().classify
        reached = {classify(st) for _, st in self._statuses()}
        assert len(reached - {"corrected"}) >= 3, f"only reached {sorted(reached)}"

    def test_the_row_that_motivated_the_part_class_is_in_it(self) -> None:
        classify = _load().classify
        by_row = dict(self._statuses())
        assert "D04#1" in by_row, "D04#1 lost its annotation"
        assert classify(by_row["D04#1"]) == "part"


class TestTheDeliveryClaimTripwire:
    """The fall-through's one blind spot, watched failing on both its vocabularies.

    ``corrected`` is reached by falling through, not by matching, so a row that claims
    a delivery in non-canonical prose lands there silently and is scheduled a second
    time.  Three real rows did (D04#8, D08#6, D13#9, #1429), and nothing could see it:
    ``check_fidelity.py`` compares each row's *text* against its annotation -- both
    sides derive from the same string, so they agree by construction -- and never its
    *class*.  A checker only fails on the dimension it inspects.

    Each plant below is a string the pre-fix classifier returned ``corrected`` for.
    """

    @pytest.mark.parametrize(
        "status",
        [
            "**PREMISE CORRECTED — and SHIPPED, so this row is no longer scheduled work.**",
            "**PARTIALLY SHIPPED (3 of 4 files) — the count was wrong.**",
            "**COUNT CORRECTED (R8)** — the mount was DELIVERED in the same change",
            "shipped last week, no prefix at all",
        ],
    )
    def test_a_non_canonical_delivery_claim_raises(self, status: str) -> None:
        with pytest.raises(ValueError, match="canonical delivery prefix"):
            _load().classify(status)

    def test_the_message_names_both_canonical_spellings(self) -> None:
        """An error that does not say how to fix it sends the reader to the source."""
        with pytest.raises(ValueError) as exc:
            _load().classify("**SHIPPED** — but spelled how?")
        msg = str(exc.value)
        assert "**FIXED (PR #N)**" in msg
        assert "PART-DONE (PR #N)" in msg

    @pytest.mark.parametrize(
        "status",
        [
            "**FIXED (PR #1415)** — PREMISE ALSO CORRECTED, SHIPPED whole",
            "**MOUNT DONE / CENSUS PART-DONE (PR #1413)** — the mount SHIPPED",
        ],
    )
    def test_a_canonical_row_never_reaches_the_tripwire(self, status: str) -> None:
        """Delivery classes return two branches above it, so their prose is free."""
        assert _load().classify(status) in {"done", "part"}

    def test_a_refuted_row_mentioning_delivery_never_reaches_the_tripwire(self) -> None:
        """Precedence, not the tripwire, owns this case -- and it must stay that way.

        Ordering the tripwire ahead of ``REFUTED`` turns a legitimately struck row into
        a hard error and the plan stops compiling.  This is the plant that dies on that
        reordering; without it the tripwire's position carries no test at all.
        """
        s = "**FIX REFUTED (R2)** — the sibling row SHIPPED, this one cannot"
        assert _load().classify(s) == "refuted"

    def test_a_skip_row_mentioning_delivery_never_reaches_the_tripwire(self) -> None:
        assert _load().classify("**filed #1409** — the fix SHIPPED elsewhere") == "skip"

    @pytest.mark.parametrize(
        "status",
        [
            "**COUNT CORRECTED (R8)** — 28 auto-fixable, not ~60",
            "**MECHANISM CORRECTED** — the cited assertion does not exist",
            "no verdict keyword at all",
            "**RANK CORRECTED** — this belongs in W3, not W1",
        ],
    )
    def test_ordinary_corrected_prose_still_falls_through(self, status: str) -> None:
        """Anti-over-reach: a tripwire that fires on the common case is a broken gate."""
        assert _load().classify(status) == "corrected"

    def test_the_tripwire_vocabulary_is_not_a_sixth_class(self) -> None:
        """It raises; it never returns.  A class would let non-canonical prose settle."""
        mod = _load()
        assert not hasattr(mod, "DELIVERY_CLAIM_CLASS")
        assert mod.DELIVERY_CLAIM.search("SHIPPED")
        assert mod.DELIVERY_CLAIM.search("delivered")
        assert not mod.DELIVERY_CLAIM.search("shipping container")
        assert not mod.DELIVERY_CLAIM.search("undelivered")


class TestTheCorpusClearsTheTripwire:
    """A ratchet is only a ratchet if the corpus is at zero when it lands."""

    def test_no_real_status_trips_it(self) -> None:
        classify = _load().classify
        tripped = []
        for row, st in TestAgainstTheRealAnnotations._statuses():
            try:
                classify(st)
            except ValueError:
                tripped.append(row)
        assert not tripped, (
            f"{tripped} claim a delivery in non-canonical prose; canonicalise the "
            "annotation rather than widening the tripwire"
        )

    @pytest.mark.parametrize(
        ("row", "expected"),
        [("D13#9", "done"), ("D04#8", "part"), ("D08#6", "part")],
    )
    def test_the_three_misfiled_rows_are_now_in_their_true_classes(
        self, row: str, expected: str
    ) -> None:
        """D08#6 is ``part``, not ``done``: it shipped 3 of 4 files.

        Worth pinning per-row rather than as a count -- ``done`` and ``part`` both move
        a row out of ``corrected``, so a tally alone cannot tell a correct split from a
        wrong one that happens to sum right.
        """
        by_row = dict(TestAgainstTheRealAnnotations._statuses())
        assert _load().classify(by_row[row]) == expected

    def test_the_class_split_is_what_the_plan_publishes(self) -> None:
        """00_MASTER.md is hand-authored and republishes these numbers (NN17).

        It has no generator, so nothing else keeps it honest; this is the check that
        goes red when the tally moves and master is not updated with it.
        """
        classify = _load().classify
        got = collections.Counter(
            classify(st) for _, st in TestAgainstTheRealAnnotations._statuses()
        )
        assert dict(got) == {
            "skip": 2,
            "done": 27,
            "part": 4,
            "refuted": 14,
            "corrected": 25,
        }, f"class split moved to {dict(got)} -- update 00_MASTER.md in the same change"
