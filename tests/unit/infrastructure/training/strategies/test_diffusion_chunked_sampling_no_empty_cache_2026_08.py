"""``_sample_multistep_chunked`` must not call ``empty_cache`` per chunk.

A Scalene profile of ``experiment_11_attention_none`` (300 iterations,
V100-PCIE-32GB, ``1bde7f3f66c5``) charged the per-chunk
``torch.cuda.empty_cache()`` 1.99 % of a 927 s run (~18 s), and an in-loop
``empty_cache`` is a non-negotiable 9 violation. Its docstring justified it as
what "keeps peak at one chunk's footprint" -- that rationale was false, and
:class:`TestTheAllocatorClaim` is the measurement that says so.

Three things are pinned here, and each detector is shown to discriminate rather
than merely to be green (non-negotiable 15):

* the per-chunk call is gone -- asserted **on the AST**, because the corrected
  docstring now mentions ``empty_cache`` three times and would satisfy any
  ``assert "empty_cache" not in source`` pin;
* the two *deliberate* sites in ``validation_step`` survive -- over-deletion is
  as much a defect as the original call, and they carry a real OOM history;
* peak **allocated** memory across a chunked loop is identical with and without
  ``empty_cache`` -- the empirical claim the rewritten docstring now makes.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

DIFFUSION_SOURCE = (
    pathlib.Path(__file__).resolve().parents[5]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "training"
    / "strategies"
    / "diffusion.py"
)

# The exact two lines this change deleted, at the indentation they had inside
# the chunk loop. Planted at the *call site* shape, not a convenient helper:
# a detector proven on a tidier shape is not proven on this one.
DELETED_CALL_SITE = """
def _sample_multistep_chunked(self, measurement):
    parts = []
    for meas_c in measurement.split(1, dim=0):
        parts.append(gen.sample(measurement=meas_c))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)
"""

# Prose only: a docstring and a comment that both name the call. An
# ``assert "empty_cache" not in src`` pin goes red on this; the AST does not.
PROSE_ONLY = '''
def _sample_multistep_chunked(self, measurement):
    """This loop used to call ``torch.cuda.empty_cache()`` between chunks."""
    # torch.cuda.empty_cache() would go here, but it is a sync.
    return measurement
'''


def _empty_cache_calls(node: ast.AST) -> list[int]:
    """Line numbers of every ``*.empty_cache()`` **call** under ``node``."""
    return [
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "empty_cache"
    ]


def _functions_named(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    """**Every** definition of ``name`` -- a list, never a name-keyed dict.

    Eight strategy files in this repo define ``_compute_losses_impl`` two or
    three times; keying by name silently keeps the last and undercounts.
    """
    return [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]


@pytest.fixture(scope="module")
def diffusion_tree() -> ast.Module:
    return ast.parse(DIFFUSION_SOURCE.read_text(encoding="utf-8"))


class TestTheDetectorDiscriminates:
    """Before trusting a green result, watch the detector go red."""

    def test_it_sees_the_call_this_change_deleted(self) -> None:
        found = _empty_cache_calls(ast.parse(DELETED_CALL_SITE))
        assert len(found) == 1, "detector is blind to the shape it must catch"

    def test_prose_alone_does_not_satisfy_it(self) -> None:
        assert _empty_cache_calls(ast.parse(PROSE_ONLY)) == []
        # ...while the source-text pin this replaces would have been fooled:
        assert "empty_cache" in PROSE_ONLY


class TestChunkedSampler:
    def test_exactly_one_definition_exists(self, diffusion_tree: ast.Module) -> None:
        defs = _functions_named(diffusion_tree, "_sample_multistep_chunked")
        assert len(defs) == 1, (
            f"a second definition would shadow the one under test; found {len(defs)}"
        )

    def test_no_empty_cache_call_in_the_chunk_loop(self, diffusion_tree: ast.Module) -> None:
        (chunked,) = _functions_named(diffusion_tree, "_sample_multistep_chunked")
        assert _empty_cache_calls(chunked) == [], (
            "the per-chunk empty_cache is back: it forces a device synchronise "
            "once per validation chunk and cannot lower peak allocated memory "
            "(non-negotiable 9)"
        )

    def test_the_docstring_still_mentions_it(self, diffusion_tree: ast.Module) -> None:
        """The corrected rationale is load-bearing prose -- do not silently drop it."""
        (chunked,) = _functions_named(diffusion_tree, "_sample_multistep_chunked")
        doc = ast.get_docstring(chunked) or ""
        assert "empty_cache" in doc, (
            "the docstring explaining why the call was removed is gone; a future "
            "reader will re-add it"
        )


class TestDeliberateSitesSurvive:
    """Over-deletion is a defect too. These two calls are kept on purpose."""

    def test_validation_step_keeps_its_two_calls(self, diffusion_tree: ast.Module) -> None:
        (validation_step,) = _functions_named(diffusion_tree, "validation_step")
        found = _empty_cache_calls(validation_step)
        assert len(found) == 2, (
            "validation_step should keep exactly two empty_cache calls -- the "
            "one-shot training->validation phase boundary (cross-phase "
            "fragmentation) and the per-rung boundary (documented OOM history); "
            f"found {len(found)} at lines {found}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestTheAllocatorClaim:
    """Measure the claim the rewritten docstring makes.

    ``empty_cache`` returns *cached* blocks to the driver. It moves
    ``max_memory_reserved``; it cannot move ``max_memory_allocated``, which is
    what "peak at one chunk's footprint" actually means.
    """

    @staticmethod
    def _chunked_loop(*, call_empty_cache: bool) -> tuple[int, int]:
        device = torch.device("cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        parts: list[torch.Tensor] = []
        for _ in range(4):
            # A chunk's activations: large, transient, freed each iteration.
            activations = torch.empty(64 * 1024 * 1024 // 4, device=device)
            parts.append(activations[:16].clone())
            del activations
            if call_empty_cache:
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return (
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )

    def test_peak_allocated_is_identical_without_empty_cache(self) -> None:
        with_call = self._chunked_loop(call_empty_cache=True)
        without_call = self._chunked_loop(call_empty_cache=False)
        assert without_call[0] == with_call[0], (
            "peak ALLOCATED differs, so the removed empty_cache was load-bearing "
            f"after all: {without_call[0]} vs {with_call[0]}"
        )

    def test_the_measurement_is_not_vacuous(self) -> None:
        """A loop that never frees anything must show a *rising* peak.

        Without this, ``test_peak_allocated_is_identical`` would pass on a
        harness that allocates nothing at all.
        """
        device = torch.device("cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        held: list[torch.Tensor] = []
        peaks: list[int] = []
        for _ in range(3):
            held.append(torch.empty(64 * 1024 * 1024 // 4, device=device))
            peaks.append(torch.cuda.max_memory_allocated(device))
        torch.cuda.synchronize()
        assert peaks[0] < peaks[1] < peaks[2], (
            f"the harness allocates nothing measurable; peaks={peaks}"
        )
