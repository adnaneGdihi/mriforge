"""Sphinx configuration for the spectramr Read the Docs site."""

from __future__ import annotations

import os
import sys

# Make the package importable so autoapi / autodoc can walk it.
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

project = "spectraMR"
author = "Adnane Gdihi"
copyright = "2026, Adnane Gdihi"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",  # `..  automodule::` directives in docs/api/*.rst
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]


# Optional extensions: enabled only when installed. This keeps the
# Sphinx config importable both in the `[docs]`-extra environment used by
# Read the Docs and in a minimal local environment.
def _try_extension(name: str) -> None:
    try:
        __import__(name)
        extensions.append(name)
    except ImportError:
        pass


_try_extension("sphinxcontrib.mermaid")
_try_extension("sphinx_copybutton")

# NOTE: `autoapi.extension` is deliberately NOT loaded.
# The API tree is kept under `docs/api/*.rst` as a sphinx-apidoc-style
# checkpointed tree (git-trackable, navigable in the repo). autoapi would
# regenerate the same files at build time and collide. To switch back to
# autoapi, delete every `docs/api/*.rst`, drop `sphinx.ext.autodoc` above,
# and uncomment the autoapi block in `archive/autoapi_config.py`.

# Heavy native deps are mocked so the build works on a CPU-only Read the
# Docs runner without GPU wheels. But mocking `torch` breaks type-hint
# evaluation in any module that uses `str | torch.device` syntax (PEP 604
# unions require real types, not MagicMock). So we mock ONLY the packages
# that aren't actually installed in the current environment — the local
# `.venv` and the cluster get the real packages; RtD's bare CPU runner
# falls back to mocks.
_HEAVY_DEPS = [
    "torch",
    "torchio",
    "monai",
    "torchkbnufft",
    "h5py",
    "nibabel",
    "scipy",
    "matplotlib",
    "tensorboard",
    "wandb",
    "einops",
    "timm",
    "lpips",
    "kornia",
    "torch_fidelity",
    "clean_fid",
    "piq",
    "torchmetrics",
    "torchaudio",
    "torchvision",
    "pandas",
    "seaborn",
    "optuna",
    "omegaconf",
    "hydra",
    "rich",
    "click",
    "typer",
    "pydicom",
    "skimage",
]
autodoc_mock_imports: list[str] = []
for _name in _HEAVY_DEPS:
    try:
        __import__(_name)
    except ImportError:
        autodoc_mock_imports.append(_name)

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "fieldlist",
]

# Auto-generate anchor IDs for headings up to ##### (h5). This is what makes
# `[link](page.md#section-name)` references work across MyST documents.
# Without it, MyST silently refuses to resolve fragment links to headings.
myst_heading_anchors = 5

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The comprehensive hand-written index (formerly ``index_legacy.rst``, renamed
# to ``index.rst`` on 2026-05-26 per user request) is the landing page in the
# private tree. The slim v0.1.0 ``index.md`` is excluded below to avoid a root
# collision.
#
# The public export ships ``index_public.rst`` and denies ``index.rst``. It has
# to: the comprehensive index names pages that are internal-only (the private CI
# lane, the export procedure, the issue taxonomy, the site's cluster layout), and
# a toctree entry pointing at a document that is not there is a warning -- which
# ``.readthedocs.yaml`` turns into a FAILED build, because it sets
# ``fail_on_warning: true``. Selecting the root from what is actually present
# keeps one conf.py serving both trees instead of forking it, and the fork is
# what would drift. Whichever index is not the root is excluded, or it builds as
# an orphan and costs the warning this exists to avoid.
_HERE = os.path.dirname(os.path.abspath(__file__))
# ``index.rst`` is the PUBLIC landing page and is the root doc in both trees.
# Read the Docs serves ``index.html`` at the site root, so the root doc has to be
# named ``index`` -- an earlier revision selected ``index_public`` when
# ``index.rst`` was absent, which built successfully and produced no
# ``index.html`` at all. The comprehensive internal index is ``index_internal``,
# marked ``:orphan:`` and denied from the export.
root_doc = "index"

# Restored to the original Read-the-Docs theme (the pre-v0.1.0 "old"
# documentation style). Falls back gracefully if not installed.
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_title = "spectraMR"
html_static_path = ["_static"]

# Navigation behaviour. The tree is wide (14 captioned sections over ~415
# documents), so the defaults work against it:
#
#   * `collapse_navigation: True` (the default) collapses every section that
#     isn't the current one, so the sidebar cannot be used to see what else
#     exists — you have to already know where you're going.
#   * `navigation_depth: 4` (the default) is deeper than this tree is
#     meaningful; 3 reaches "section -> page -> heading" and stops.
#   * `prev_next_buttons_location` defaults to "bottom" only. Putting them at
#     both ends makes sequential reading (the tutorials especially) work
#     without scrolling back.
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "titles_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Without a timeout a build on a host that cannot reach these inventories hangs
# instead of degrading: intersphinx has no default deadline. Unresolved external
# refs are a warning, not a failure, so a slow/offline host still builds.
intersphinx_timeout = 15

# Keep this list PRECISE. Every pattern here silently deletes a page from the
# build, and Sphinx only *warns* when a `toctree` still links it — so an
# over-broad glob produces a sidebar entry that resolves to nothing while the
# build still exits 0. That is exactly how this file drifted: a blanket
# `physics_*.rst` aimed at `physics_rigor_audit_*` also deleted the 45 KB
# `physics_math.rst`, and `experiment_11_*.md` deleted the very Markdown paper
# that the `experiment_11_kspace_cold_diffusion.rst` exclusion below exists to
# make room for. At the 2026-07-16 audit, 87 of the 210 index entries (41%)
# were dead this way.
#
# Before adding a pattern, run `python scripts/ci/check_docs_navigation.py`;
# it fails on dead links, duplicates, and orphans. Prefer an exact filename
# over a glob, and never add a glob without checking what else it catches.
exclude_patterns = [
    # (no landing-page exclusion: `index.rst` is the root doc in BOTH trees, and
    # the internal index is `index_internal.rst`, which carries `:orphan:`.)
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
    "**/archive",
    "**/.backup_*",
    # Vendored Lean toolchain packages ship ~65 README files under
    # ``docs/lean/**/.lake/``. ``.lake`` is gitignored, so Read the Docs never
    # sees them, but a local build scans them and reports 65 orphans.
    "lean/**/.lake/**",
    # The Lean proof packages under ``docs/lean/`` are retained as proof
    # artifacts, not as pages: ``tests/unit/docs/test_sim2rank_*_layer.py``
    # read the ``.lean`` sources at runtime and assert they are free of
    # ``sorry``/``admit``/global ``axiom``. Their READMEs are not part of the
    # usage documentation, so they are excluded from the build rather than
    # wired into the sidebar.
    "lean/**/README.md",
    # The 16 design-compliance dossiers (+ their master) are the *compiler input*
    # for `TODO/production_plan/`, not pages: `compile_plan.py` transcribes each
    # one's §9 fix set into a plan file. They are dated point-in-time evidence
    # for one audit, nobody navigates to them, and `check_docs_navigation.py`
    # reports all 17 as orphans otherwise. They are deliberately unshipped to the
    # public export too — see the NOT-SHIPPED foot of
    # `scripts/release/public_allowlist.txt`.
    #
    # Excluding a page normally suppresses every check on it. These are the
    # exception: `TODO/production_plan/tools/check_fidelity.py` re-parses all 249
    # §9 rows independently of the compiler and fails on any drift between
    # dossier and plan file. That is a stronger and far more relevant check than
    # Sphinx applies to an orphan page — this exclusion removes them from
    # *navigation*, not from *verification*.
    "audits/**",
]

# Don't fail the build on missing references inside autoapi-generated pages —
# they cross-link aggressively and any stale ref blocks RTD CI.
nitpicky = False

# Warning suppressions — each one bounded + justified.
#
# `ref.python` —  `__init__.py` re-exports classes (e.g. `MRIVolume`) from
#   submodules. Sphinx's `automodule::` registers the class both under the
#   submodule path (`spectramr.domain.entities.entities_3d.MRIVolume`) and under
#   the package path (`spectramr.domain.entities.MRIVolume`). A bare
#   `:py:attr:`shape\`` in a docstring then matches both, producing an
#   ambiguity warning. The targets are identical Python objects — the
#   ambiguity is structural, not a real ref bug.
#
# `docutils` — many Python docstrings in this repo are written in
#   Markdown (with `###` headings and ``` ```mermaid ``` ``` fenced code
#   blocks). The default `automodule::` extractor parses docstrings as RST,
#   which mis-interprets the `###`/fenced-code syntax as "Unexpected
#   indentation". The docstrings render fine in IDE tooltips and via
#   `mypy --strict-help`; only Sphinx complains. Migrating 60+ files to RST
#   syntax is out of scope; suppressing the class is documented as a known
#   limitation in docs/CLAUDE.md.
#
# Add NEW suppressions sparingly. Each one masks a class of real bugs.
suppress_warnings = [
    "ref.python",
    "docutils",
    # Many submodules of this repo import `torch.utils.tensorboard` at the
    # top level. When `tensorboard` is mocked (because it's not installed in
    # the RtD CPU runner), `torch.utils.tensorboard.__init__` does
    # `Version(tensorboard.__version__)` — that fails because the mock's
    # `__version__` is a MagicMock, not a string. The result: autodoc
    # cannot import the submodule and emits a warning. The build still
    # produces correct docs for every module that DOES import cleanly; the
    # mocked-cascade subset just lacks its auto-generated content. We
    # suppress the warning class until either (a) tensorboard moves into
    # the docs extra or (b) the affected import sites are wrapped in
    # `try / except ImportError` guards.
    "autodoc.import_object",
]
