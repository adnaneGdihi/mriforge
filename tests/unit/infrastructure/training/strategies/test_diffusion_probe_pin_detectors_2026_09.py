"""Planted violations for the t=0 probe's two source-scanning pins.

Non-negotiable 15: a detector is only a gate for the violation shape it has
been watched to fail on, and every gate in this repo that turned out blind had
gone red many times -- on the easy shape. So each core below is fed both the
violation it claims to catch AND the near-misses that must NOT trip it: a name
in a comment, in a docstring, in a string literal, or referenced without being
called. A substring check passes all four; that is exactly why these pins parse.

The cores live in `test_diffusion_t0_predc_probe_2026_09.py` beside the real
pins that call them on `inspect.getsource(...)`. Committing helper-level plants
here does not on its own discharge the rule -- a helper-only plant scores a
call-site plant green -- so the call sites were separately mutated in the
working tree and each of the 12 mutations was confirmed to turn its pin red.
"""

from __future__ import annotations

import textwrap

from tests.unit.infrastructure.training.strategies.test_diffusion_t0_predc_probe_2026_09 import (
    image_logger_call_counts,
    probe_call_site_count,
)

LOGGER = "_log_validation_images_to_tensorboard"
PROBE = "_t0_pre_dc_probe_metrics"


def _method(body: str) -> str:
    """Wrap a body as an indented method, the shape `inspect.getsource` returns."""
    return "    def _compute_validation_metrics(self):\n" + textwrap.indent(
        textwrap.dedent(body).strip("\n"), "        "
    )


class TestImageLoggerGuardPinSeesTheShapesItClaimsTo:
    """Plants for `image_logger_call_counts`."""

    def test_unguarded_call_is_reported(self):
        """The plant: the guard is simply not there."""
        assert image_logger_call_counts(_method(f"self.{LOGGER}(x)")) == (1, 0)

    def test_guarded_call_is_reported_as_guarded(self):
        assert image_logger_call_counts(
            _method(f"if emit_reports:\n    self.{LOGGER}(x)")
        ) == (1, 1)

    def test_if_true_wrapper_does_not_count_as_a_guard(self):
        """The plant that motivated parsing: `if True:` keeps the source shape."""
        assert image_logger_call_counts(_method(f"if True:\n    self.{LOGGER}(x)")) == (1, 0)

    def test_guard_on_a_different_name_does_not_count(self):
        assert image_logger_call_counts(_method(f"if emit_images:\n    self.{LOGGER}(x)")) == (
            1,
            0,
        )

    def test_compound_condition_is_deliberately_not_a_guard(self):
        """`if emit_reports and x:` is a BoolOp -- refused, not guessed at.

        Chosen, not accidental: a compound condition can be false for reasons
        the probe does not control, so the call is not provably suppressed.
        """
        assert image_logger_call_counts(
            _method(f"if emit_reports and x:\n    self.{LOGGER}(x)")
        ) == (1, 0)

    def test_one_guarded_and_one_unguarded_call_is_caught_by_the_total(self):
        """The shape the `total == guarded` clause exists for.

        A pin that only asserted "some call is guarded" would pass here while
        the probe's scoring pass still wrote a second set of renders.
        """
        total, guarded = image_logger_call_counts(
            _method(f"if emit_reports:\n    self.{LOGGER}(x)\nself.{LOGGER}(y)")
        )
        assert (total, guarded) == (2, 1)

    def test_call_nested_deeper_inside_the_guard_still_counts(self):
        """Documents the shape: guarded-ness is by containment, not adjacency."""
        assert image_logger_call_counts(
            _method(f"if emit_reports:\n    for _ in r:\n        self.{LOGGER}(x)")
        ) == (1, 1)

    def test_name_in_a_comment_is_not_a_call(self):
        assert image_logger_call_counts(_method(f"pass  # self.{LOGGER}(x)")) == (0, 0)

    def test_name_in_a_docstring_is_not_a_call(self):
        assert image_logger_call_counts(_method(f'"""calls self.{LOGGER}(x)."""')) == (0, 0)

    def test_name_as_a_string_literal_is_not_a_call(self):
        assert image_logger_call_counts(_method(f'k = "{LOGGER}"')) == (0, 0)

    def test_name_referenced_without_being_called_is_not_a_call(self):
        assert image_logger_call_counts(_method(f"fn = self.{LOGGER}")) == (0, 0)

    def test_bare_name_call_is_not_counted_and_that_is_documented(self):
        """Attribute calls only. The production shape is always `self.<name>`;
        matching a bare name would let an unrelated local satisfy the pin."""
        assert image_logger_call_counts(_method(f"{LOGGER}(x)")) == (0, 0)


class TestProbeCallSitePinSeesTheShapesItClaimsTo:
    """Plants for `probe_call_site_count`."""

    def test_no_call_site_is_reported_as_zero(self):
        """The plant: the probe was dropped from a return path."""
        assert probe_call_site_count(_method("return {}")) == 0

    def test_a_single_call_site_is_reported_as_one(self):
        """The plant that matters: wired on one return path, not both."""
        assert probe_call_site_count(_method(f"m.update(self.{PROBE}(a))\nreturn m")) == 1

    def test_both_call_sites_are_counted(self):
        assert (
            probe_call_site_count(
                _method(f"if c:\n    return self.{PROBE}(a)\nreturn self.{PROBE}(b)")
            )
            == 2
        )

    def test_call_nested_in_a_branch_is_counted(self):
        assert probe_call_site_count(_method(f"if c:\n    self.{PROBE}(a)")) == 1

    def test_name_in_a_comment_is_not_counted(self):
        assert probe_call_site_count(_method(f"pass  # self.{PROBE}(a)")) == 0

    def test_name_in_a_docstring_is_not_counted(self):
        assert probe_call_site_count(_method(f'"""wires self.{PROBE}(a) twice."""')) == 0

    def test_name_as_a_string_literal_is_not_counted(self):
        assert probe_call_site_count(_method(f'k = "{PROBE}"')) == 0

    def test_name_referenced_without_being_called_is_not_counted(self):
        assert probe_call_site_count(_method(f"fn = self.{PROBE}")) == 0

    def test_bare_name_call_is_not_counted_and_that_is_documented(self):
        assert probe_call_site_count(_method(f"{PROBE}(a)")) == 0
