"""Regression guard: ``spectramr.main`` must NOT import the heavy pipeline graph at
module top-level (performance follow-up, 2026-06-19).

``from spectramr.pipelines import run_training_pipeline`` pulls the entire pipeline
→ model-registry → monai/torchio graph (tens of seconds cold). It used to sit at
module top, so ``from spectramr.main import _parse_value`` (and every other
lightweight reach into the module — e.g. the ``ablation`` verb) paid the full
cold-import cost: the "after the import line it waits for a whole minute"
symptom. It is now imported lazily inside the two functions that use it
(``__common_train_setup`` / ``experiment_command``).

This guard parses the source with :mod:`ast` (it never imports the module, so no
torch is required) and pins the contract so a future refactor cannot quietly
re-hoist the import.
"""

from __future__ import annotations

import ast
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[2] / "src" / "spectramr" / "main.py"


def _module_level_imports(source: str) -> list[str]:
    """Dotted module names imported at MODULE level (excludes function-local)."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:  # only top-level statements — nested defs are skipped
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def test_pipelines_not_imported_at_module_level():
    top = _module_level_imports(MAIN_PY.read_text())
    offenders = [
        m
        for m in top
        if m == "spectramr.pipelines" or m.startswith("spectramr.pipelines.")
    ]
    assert not offenders, (
        f"spectramr.main imports the heavy pipeline graph at module level: {offenders}. "
        "Import run_training_pipeline lazily inside the function that uses it so "
        "`import spectramr.main` stays cheap."
    )


def test_run_training_pipeline_is_imported_lazily():
    # The deferral must actually be present (not merely deleted): both call sites
    # import it function-locally.
    source = MAIN_PY.read_text()
    lazy = source.count("from spectramr.pipelines import run_training_pipeline")
    assert lazy >= 2, (
        "expected run_training_pipeline to be imported lazily in "
        f"__common_train_setup and experiment_command, found {lazy} local import(s)"
    )


def test_cuda_alloc_conf_set_before_torch_import():
    """``PYTORCH_CUDA_ALLOC_CONF`` MUST be assigned before ``import torch``.

    PyTorch reads this env var exactly once, when the CUDA caching allocator is
    first initialised (at/just after ``import torch``). An assignment placed
    *after* the import is silently inert — ``expandable_segments:True`` never
    takes effect and fragmentation OOMs are not mitigated despite the setting
    being present (the ``experiment_11_attention_wavelet_freq`` 2026-07-02 crash:
    2.88 GiB free on a 31 GiB GPU yet a 132 MiB alloc failed). This guard pins
    the ordering so a future edit cannot re-sink the assignment below the import.
    Uses raw source-line indices (never imports the module → no torch required).
    """
    lines = MAIN_PY.read_text().splitlines()
    alloc_line = next(
        (i for i, ln in enumerate(lines) if 'os.environ["PYTORCH_CUDA_ALLOC_CONF"]' in ln),
        None,
    )
    torch_import_line = next(
        (i for i, ln in enumerate(lines) if ln.strip() == "import torch"),
        None,
    )
    assert alloc_line is not None, "PYTORCH_CUDA_ALLOC_CONF assignment not found in main.py"
    assert torch_import_line is not None, "`import torch` not found in main.py"
    assert alloc_line < torch_import_line, (
        "PYTORCH_CUDA_ALLOC_CONF is assigned AFTER `import torch` "
        f"(line {alloc_line + 1} vs {torch_import_line + 1}); PyTorch reads it only "
        "at allocator init, so expandable_segments would be silently inert. Move "
        "the CUDA memory-config block above `import torch`."
    )


def test_hpo_command_is_not_defined():
    # The `hpo` verb is owned entirely by ``spectramr.cli.app.hpo_cmd`` (which drives
    # HPOUseCase directly). ``main.py`` used to carry a byte-orphaned near-duplicate
    # ``hpo_command`` that nothing imported — ``cli.app`` never wired it. Removed
    # 2026-07-02; this guard stops it silently reappearing. Uses ``ast`` so the
    # module is never imported (no torch required).
    tree = ast.parse(MAIN_PY.read_text())
    defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "hpo_command" not in defs, (
        "spectramr.main.hpo_command is dead (cli.app uses its own hpo_cmd). "
        "Do not re-add it here — extend cli.app.hpo_cmd instead."
    )
