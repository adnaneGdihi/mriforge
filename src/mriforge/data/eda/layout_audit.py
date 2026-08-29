"""Audit dataset manifests against a physical ``tree -J`` snapshot of ``databases/``.

The EDA (:mod:`mriforge.data.eda.catalog`) classifies datasets from manifest *metadata*
only — it trusts ``status`` / ``data_root`` / ``records[]`` and never sees the real disk on
a remote cluster. This module closes that gap: given the JSON emitted by ``tree -J`` run in
the cluster ``databases/`` directory, it reconciles every manifest's *declared* location with
what is *physically* present and emits a per-dataset verdict.

It exists because the manifest ``status`` field is **not ground truth** — the external stubs
ship ``status: not_downloaded`` with empty ``records[]`` until a populate step is run where the
data lives (see ``scripts/data/gen_external_dataset_manifests.py``). A manifest can therefore
say "not downloaded" while the data is on disk, or say "downloaded" while pointing at a
``data_root`` that does not exist on this machine. The audit measures, it does not trust.

Verdicts (``Verdict``):

- ``ok`` — loadable data files present at (or under) the declared ``data_root``.
- ``data_relocated`` — no data at the declared path, but loadable files found elsewhere in the
  tree (e.g. data is in ``extracted/`` while ``data_root`` points at ``raw/``, or the cluster
  layout differs from the repo-relative path). ``found_at`` carries the real location.
- ``archive_only`` — archives present (``.zip`` / ``.7z`` / ``.tar.xz`` / …) but never
  extracted; no loadable files anywhere under the dataset dir → needs an extract step.
- ``metadata_only`` — only ``.json`` / ``.html`` / ``.xlsx`` / ``.pdf`` … and no data or
  archive → a failed or placeholder download (e.g. a landing-page ``index.html``).
- ``empty`` — the dataset directory exists but contains no files.
- ``absent`` — the dataset directory is nowhere in the tree → genuinely not downloaded.
- ``unresolvable`` — the manifest has no ``data_root`` and no records to locate it by, so the
  physical state cannot even be checked (a manifest bug, e.g. ``data_root: null``).
- ``no_manifest`` — (reverse check) a tree directory holding real data that no manifest covers.

Independently of the verdict, two ``status``-reconciliation flags are reported:
``present_but_marked_not_downloaded`` and ``claims_downloaded_but_no_data``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# --- extension classification -------------------------------------------------------------
# Grounded in the actual ext histogram of the cluster databases/ tree (.dcm/.ima/.h5/.nii.gz/
# .cfl/.mat/.npy/.mrd/.png dominate the imaging data; .zip/.7z/.tgz/.tar.xz the archives).
_DATA_SUFFIXES = (".nii.gz",)  # compound data suffix checked before the bare-ext table
_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2", ".tbz")
_DATA_EXTS = frozenset(
    {
        ".dcm",
        ".ima",
        ".h5",
        ".hdf5",
        ".nii",
        ".nib",
        ".mnc",
        ".mgz",
        ".par",
        ".rec",
        ".cfl",
        ".mat",
        ".npy",
        ".npz",
        ".mrd",
        ".dat",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".bin",
        ".pt",
        ".pth",
        ".raw",
        ".img",
    }
)
_ARCHIVE_EXTS = frozenset({".zip", ".7z", ".gz", ".xz", ".bz2", ".rar", ".tar", ".z"})

# Generic *container* path segments — a directory whose basename is one of these does not
# identify a dataset (``raw`` / ``extracted`` hold one dataset's files; ``data`` / ``datasets``
# hold many). The dataset-prefix walk skips them so a missing ``external/<x>/raw`` resolves to
# the dataset dir ``external/<x>`` (and its ``extracted/`` sibling), never up to the shared
# ``external/`` container — which would fabricate a bogus relocation for a genuinely absent set.
_GENERIC_SEGMENTS = frozenset(
    {
        "raw",
        "extracted",
        "datasets",
        "data",
        "nifti",
        "nifti_reconstructed",
        "kspace",
        "images",
        "image",
        "anat",
        "files",
        "derivatives",
        "sourcedata",
        "func",
        "dwi",
    }
)


def classify_ext(name: str) -> str:
    """Classify a filename as ``"data"`` (loadable imaging/array), ``"archive"``, or ``"other"``.

    A purely numeric extension (``IM.4983353``, ``vol.06666``) is treated as ``data`` — these
    are DICOM-instance / split-volume frames the download pipeline leaves un-suffixed.
    """
    low = name.lower()
    for s in _DATA_SUFFIXES:
        if low.endswith(s):
            return "data"
    for s in _ARCHIVE_SUFFIXES:
        if low.endswith(s):
            return "archive"
    dot = low.rfind(".")
    if dot <= 0:  # no extension, or a dotfile like ".gitkeep"
        return "other"
    ext = low[dot:]
    if ext in _DATA_EXTS:
        return "data"
    if ext in _ARCHIVE_EXTS:
        return "archive"
    if ext[1:].isdigit():  # numeric ext → DICOM frame / split file
        return "data"
    return "other"


# --- tree index ---------------------------------------------------------------------------
@dataclass
class Stats:
    """Recursive file-kind counts for one directory subtree (relative to the tree root)."""

    data: int = 0
    archive: int = 0
    other: int = 0
    data_dir: str | None = None  # a descendant dir holding immediate data files (for ``found_at``)
    sample_data: tuple[str, ...] = ()


class TreeIndex:
    """An index over a ``tree -J`` snapshot, queryable by directory path relative to the root.

    The root node (``tree -J`` labels it ``"."``) maps to the empty path ``""``; its children
    are ``"external"``, ``"brain"``, … so a manifest's ``databases/external/x/raw`` maps to the
    key ``"external/x/raw"`` (see :func:`tree_relative`).
    """

    def __init__(self) -> None:
        self.dirs: dict[str, Stats] = {}
        self.files_by_dir: dict[str, frozenset[str]] = {}
        self.basename_index: dict[str, list[str]] = {}

    def subtree(self, relpath: str) -> Stats | None:
        return self.dirs.get(relpath.strip("/"))

    def dataset_prefix(self, relpath: str) -> str | None:
        """Deepest existing **dataset-identifying** prefix of ``relpath`` (≥2 segments).

        Walks the declared path from the leaf upward, skipping segments whose basename is a
        generic container (``raw`` / ``extracted`` / ``data`` …), and never returns a single
        top-level segment. So ``external/<x>/raw`` → ``external/<x>``, ``m4raw/data/<split>/
        <split>_image/kspace`` → ``m4raw/data/<split>``, and a missing ``external/calgary/raw``
        → ``None`` (it does NOT fall back to the shared ``external/`` container).
        """
        segs = [s for s in relpath.strip("/").split("/") if s]
        for k in range(len(segs), 1, -1):  # require ≥2 segments — never anchor at a top-level
            if segs[k - 1].lower() in _GENERIC_SEGMENTS:
                continue
            cand = "/".join(segs[:k])
            if cand in self.dirs:
                return cand
        return None

    def unique_data_dir(self, basename: str) -> str | None:
        """The directory named ``basename`` *iff* exactly one such dir holds data (else ``None``).

        Conservative on purpose: a name shared by two data dirs (``fastmri`` under both
        ``brain/`` and ``knee/``) returns ``None`` rather than guessing the wrong one.
        """
        hits = [h for h in self.basename_index.get(basename, []) if self.dirs[h].data > 0]
        return hits[0] if len(hits) == 1 else None

    def locate_basenames(self, names: list[str]) -> list[str]:
        """Nested dirs whose basename is any of ``names`` (for honest 'see also' hints)."""
        out: list[str] = []
        for nm in names:
            out += [h for h in self.basename_index.get(nm, []) if "/" in h]
        return sorted(set(out))

    def file_exists(self, file_relpath: str) -> bool:
        rel = file_relpath.strip("/")
        if "/" not in rel:
            return rel in self.files_by_dir.get("", frozenset())
        dirpart, fname = rel.rsplit("/", 1)
        return fname in self.files_by_dir.get(dirpart, frozenset())

    def data_files_under(self, relpath: str) -> list[str]:
        """Tree-relative paths of every ``data``-kind file in the subtree rooted at ``relpath``."""
        rel = relpath.strip("/")
        prefix = rel + "/"
        out: list[str] = []
        for d, names in self.files_by_dir.items():
            if d == rel or d.startswith(prefix):
                out += [f"{d}/{n}" if d else n for n in names if classify_ext(n) == "data"]
        return sorted(out)


def build_tree_index(tree_json: list | dict) -> TreeIndex:
    """Build a :class:`TreeIndex` from parsed ``tree -J`` output (its list or root-dict form)."""
    root = tree_json
    if isinstance(tree_json, list):  # tree -J emits [root_dir_node, {"type": "report", ...}]
        root = next(
            (n for n in tree_json if isinstance(n, dict) and n.get("type") == "directory"), None
        )
    if not isinstance(root, dict):
        raise ValueError("tree_json is not a tree -J directory node or list")

    idx = TreeIndex()

    def walk(node: dict, rel: str) -> Stats:
        st = Stats()
        filenames: list[str] = []
        immediate_data: list[str] = []
        child_data: list[tuple[int, str]] = []
        for c in node.get("contents", []) or []:
            name = c.get("name", "")
            crel = f"{rel}/{name}" if rel else name
            if c.get("type") == "directory":
                cst = walk(c, crel)
                st.data += cst.data
                st.archive += cst.archive
                st.other += cst.other
                if cst.data_dir:
                    child_data.append((cst.data, cst.data_dir))
            else:
                filenames.append(name)
                kind = classify_ext(name)
                if kind == "data":
                    st.data += 1
                    immediate_data.append(crel)
                elif kind == "archive":
                    st.archive += 1
                else:
                    st.other += 1
        if immediate_data:
            st.data_dir = rel
            st.sample_data = tuple(immediate_data[:3])
        elif child_data:
            st.data_dir = max(child_data)[1]
        idx.dirs[rel] = st
        idx.files_by_dir[rel] = frozenset(filenames)
        idx.basename_index.setdefault(rel.rsplit("/", 1)[-1] if rel else ".", []).append(rel)
        return st

    walk(root, "")
    return idx


def tree_relative(data_root: str | Path | None) -> str | None:
    """Reduce a manifest ``data_root`` to a path relative to the ``databases/`` tree root.

    Handles cluster-absolute (``/project/.../databases/m4raw/...``), repo-relative
    (``databases/m4raw/...``), and already-relative (``m4raw/...``) forms.
    """
    if not data_root:
        return None
    p = str(data_root).replace("\\", "/")
    marker = "/databases/"
    if marker in p:
        return p.split(marker, 1)[1].strip("/")
    if p.startswith("databases/"):
        return p[len("databases/") :].strip("/")
    return p.lstrip("./").strip("/")


# --- per-dataset audit --------------------------------------------------------------------
_PRESENT_VERDICTS = {"ok", "data_relocated"}
_NO_DATA_VERDICTS = {"archive_only", "metadata_only", "empty", "absent", "unresolvable"}


@dataclass
class AuditResult:
    dataset_id: str
    manifest_path: str
    declared_root: str | None  # tree-relative data_root the manifest declares
    status: str  # manifest-declared status
    verdict: str
    found_at: str | None = None  # real data location when relocated
    data_files: int = 0
    archive_files: int = 0
    other_files: int = 0
    n_records: int = 0
    records_checked: int = 0
    records_missing: int = 0
    status_flag: str | None = (
        None  # present_but_marked_not_downloaded | claims_downloaded_but_no_data
    )
    note: str = ""

    @property
    def is_problem(self) -> bool:
        return self.verdict != "ok" or self.status_flag is not None or self.records_missing > 0


def _record_tree_paths(rec: object, declared_root: str | None) -> list[str]:
    """Tree-relative file paths a record/file entry points at (best-effort over schemas)."""
    out: list[str] = []
    vals: list[str] = []
    if isinstance(rec, str):
        vals = [rec]
    elif isinstance(rec, dict):
        for k in ("relative_path", "primary_path", "input_path", "target_path", "path", "file"):
            v = rec.get(k)
            if isinstance(v, str):
                vals.append(v)
    for v in vals:
        if "databases/" in v or v.startswith("/"):
            tr = tree_relative(v)  # full repo-/cluster-relative path
        elif declared_root:
            tr = f"{declared_root}/{v}".strip("/")  # relative to data_root
        else:
            tr = v.strip("/")
        if tr:
            out.append(tr)
    return out


def audit_entry(
    *,
    dataset_id: str,
    declared_root: str | None,
    status: str,
    records: list | tuple = (),
    files: list | tuple = (),
    index: TreeIndex,
    manifest_path: str = "",
    record_sample: int = 50,
) -> AuditResult:
    """Reconcile one manifest's declared location against the physical tree."""
    declared = declared_root.strip("/") if declared_root else None
    res = AuditResult(
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        declared_root=declared,
        status=status or "unknown",
        verdict="absent",
        n_records=len(records) or len(files),
    )

    # 1. Declared data_root exists and holds data → present exactly where the manifest says.
    if declared and declared in index.dirs and index.dirs[declared].data > 0:
        st = index.dirs[declared]
        res.data_files, res.archive_files, res.other_files = st.data, st.archive, st.other
        res.verdict, res.found_at = "ok", st.data_dir
        return _finalize(res, index, declared, records, files, record_sample)

    # 2. Resolve to the dataset directory (skips generic raw/extracted/data segments; never a
    #    shared top-level container) and classify what is physically there.
    anchor = index.dataset_prefix(declared) if declared else None
    if anchor is not None:
        st = index.dirs[anchor]
        res.data_files, res.archive_files, res.other_files = st.data, st.archive, st.other
        if st.data > 0:
            res.verdict = "ok" if anchor == declared else "data_relocated"
            res.found_at = st.data_dir or anchor
            if res.verdict == "data_relocated":
                res.note = f"data present under '{anchor}', not at declared '{declared}'"
        elif st.archive > 0:
            res.verdict = "archive_only"
            res.note = f"{st.archive} archive file(s) under '{anchor}', none extracted"
        elif st.other > 0:
            res.verdict = "metadata_only"
            res.note = f"only non-data files under '{anchor}' (failed/placeholder download)"
        else:
            res.verdict = "empty"
            res.note = f"'{anchor}' exists but holds no files"
        return _finalize(res, index, declared, records, files, record_sample)

    # 3. Declared path absent. Try a conservative, unambiguous locate by the dataset's own name.
    relocated = index.unique_data_dir(dataset_id)
    if relocated:
        st = index.dirs[relocated]
        res.verdict, res.found_at = "data_relocated", st.data_dir or relocated
        res.data_files, res.archive_files, res.other_files = st.data, st.archive, st.other
        res.note = "located by dataset name; declared data_root does not resolve on this machine"
        return _finalize(res, index, declared, records, files, record_sample)

    # 4. Genuinely unlocatable. Don't fabricate a location — list honest 'see also' candidates.
    tokens = [
        t
        for t in dataset_id.replace("-", "_").split("_")
        if len(t) >= 4 and t.lower() not in _GENERIC_SEGMENTS
    ]
    if declared:
        tokens += [
            s for s in declared.split("/") if len(s) >= 4 and s.lower() not in _GENERIC_SEGMENTS
        ]
    # Keep only shallow, plausible dataset locations (depth ≤ 3, not deep ``extracted/`` cruft).
    seealso = [
        h for h in index.locate_basenames(tokens) if h.count("/") <= 2 and "/extracted/" not in h
    ]
    seealso = sorted(seealso, key=lambda h: (h.count("/"), h))[:4]
    res.verdict = "unresolvable" if not declared else "absent"
    res.note = (
        "manifest declares no data_root and the name is nowhere in the tree"
        if not declared
        else f"declared data_root '{declared}' and dataset name are nowhere in the tree"
    )
    if seealso:
        res.note += f" (see also: {', '.join(seealso)})"
    return _finalize(res, index, declared, records, files, record_sample)


def _finalize(
    res: AuditResult,
    index: TreeIndex,
    declared_root: str | None,
    records: list | tuple,
    files: list | tuple,
    record_sample: int,
) -> AuditResult:
    # Sample record/file paths and confirm they resolve to real files in the tree.
    entries = list(records) or list(files)
    if entries:
        sampled = entries[:record_sample]
        missing = 0
        for rec in sampled:
            paths = _record_tree_paths(rec, declared_root)
            if paths and not any(index.file_exists(p) for p in paths):
                missing += 1
        res.records_checked = len(sampled)
        res.records_missing = missing

    # Reconcile the declared status against the measured physical state.
    if res.verdict in _PRESENT_VERDICTS and res.status == "not_downloaded":
        res.status_flag = "present_but_marked_not_downloaded"
    elif res.verdict in _NO_DATA_VERDICTS and res.status == "downloaded":
        res.status_flag = "claims_downloaded_but_no_data"
    return res


# --- full sweep ---------------------------------------------------------------------------
@dataclass
class AuditReport:
    results: list[AuditResult] = field(default_factory=list)
    uncovered: list[AuditResult] = field(default_factory=list)  # tree data dirs with no manifest

    @property
    def problems(self) -> list[AuditResult]:
        return [r for r in self.results + self.uncovered if r.is_problem]


def _read_files_field(manifest_path: Path) -> list:
    try:
        return json.loads(manifest_path.read_text()).get("files", []) or []
    except Exception:
        return []


def audit_manifests(manifests_root: Path, repo_root: Path, index: TreeIndex) -> AuditReport:
    """Audit every manifest under ``manifests_root`` against ``index``, then reverse-check the
    tree for data directories that no manifest covers."""
    from mriforge.data.eda.catalog import discover

    report = AuditReport()
    entries = discover(Path(manifests_root), Path(repo_root))
    for e in entries:
        declared = tree_relative(e.data_root)
        files = _read_files_field(e.manifest_path) if not e.records else []
        report.results.append(
            audit_entry(
                dataset_id=e.dataset_id,
                declared_root=declared,
                status=e.status,
                records=list(e.records),
                files=files,
                index=index,
                manifest_path=str(e.manifest_path),
            )
        )

    # Reverse check: which on-disk data directories does no manifest reach?
    covered: set[str] = set()
    for r in report.results:
        for p in (r.declared_root, r.found_at):
            if p:
                covered.add(p.strip("/"))
    report.uncovered = _find_uncovered(index, covered)
    return report


def _is_covered(reldir: str, covered: set[str]) -> bool:
    return any(
        c == reldir or c.startswith(reldir + "/") or reldir.startswith(c + "/") for c in covered
    )


def _find_uncovered(index: TreeIndex, covered: set[str]) -> list[AuditResult]:
    """Top-level and ``external/*`` directories that hold data but no manifest points into."""
    candidates: list[str] = [d for d in index.dirs if d and "/" not in d]  # top-level dirs
    candidates += [d for d in index.dirs if d.startswith("external/") and d.count("/") == 1]
    out: list[AuditResult] = []
    for d in sorted(set(candidates)):
        st = index.dirs[d]
        if st.data > 0 and not _is_covered(d, covered):
            out.append(
                AuditResult(
                    dataset_id=f"disk:{d}",
                    manifest_path="",
                    declared_root=d,
                    status="on_disk",
                    verdict="no_manifest",
                    found_at=st.data_dir,
                    data_files=st.data,
                    archive_files=st.archive,
                    other_files=st.other,
                    note="on-disk data with no covering manifest",
                )
            )
    return out


def proposed_fix(result: AuditResult, index: TreeIndex) -> dict | None:
    """Manifest-field corrections for an auto-fixable **external descriptor stub**, else ``None``.

    Auto-fixing is restricted to ``data/manifests/external/<x>.json`` whose data is physically
    present (``ok`` / ``data_relocated``). The caller additionally requires the manifest's
    ``records[]`` to be empty — a populated manifest carries structured pairing/contrast metadata
    that a flat re-enumerate would destroy, so those go to the authoritative cluster ``--populate``
    instead. The output mirrors what ``gen_external_dataset_manifests.populate_manifest`` writes:
    ``data_root`` rewritten to the real (``extracted/``) root, ``status='downloaded'``, and
    ``records[]`` of ``{relative_path, file_id}`` enumerated from the tree.
    """
    if "/manifests/external/" not in result.manifest_path.replace("\\", "/"):
        return None
    if result.verdict not in ("ok", "data_relocated") or not result.found_at:
        return None
    # Stay inside the dataset's OWN external dir. A stub that declares no external data_root, or
    # whose data relocates onto a *core* dataset (``external/fastmri_knee`` → ``knee/fastmri``,
    # ``external/m4raw_v16`` → ``m4raw/data``) or a single-subject leaf (inhouse_ulf_paired →
    # ``…/sub-0057/…/anat``), is a duplicate pointer — not a fillable external download. Skip it.
    declared = result.declared_root or ""
    if not declared.startswith("external/"):
        return None
    own_dir = "/".join(declared.split("/")[:2])  # external/<name>
    if result.verdict == "data_relocated" and "/extracted/" in result.found_at:
        root_rel = result.found_at.split("/extracted/", 1)[0] + "/extracted"  # the extract sibling
    elif result.verdict == "data_relocated":
        root_rel = result.found_at
    else:  # ok — data already at the declared root
        root_rel = declared
    if not (root_rel == own_dir or root_rel.startswith(own_dir + "/")):
        return None
    files = index.data_files_under(root_rel)
    if not files:
        return None
    records = [
        {"relative_path": f[len(root_rel) + 1 :], "file_id": f.rsplit("/", 1)[-1].rsplit(".", 1)[0]}
        for f in files
    ]
    return {
        "data_root": f"databases/{root_rel}",
        "status": "downloaded",
        "records": records,
        "total_records": len(records),
    }


def format_markdown(report: AuditReport) -> str:
    """Render the audit as a grouped Markdown report (problems first, grouped by verdict)."""
    by_verdict: dict[str, list[AuditResult]] = {}
    for r in report.results:
        by_verdict.setdefault(r.verdict, []).append(r)

    n = len(report.results)
    n_clean = sum(1 for r in report.results if not r.is_problem)
    lines = ["# Dataset layout audit", ""]
    lines.append(
        f"{n} manifests audited — **{n_clean} clean**, **{n - n_clean} flagged** "
        f"(verdict ≠ ok, or stale status / missing records), "
        f"{len(report.uncovered)} on-disk dirs with no manifest.\n"
    )

    order = [
        "absent",
        "archive_only",
        "metadata_only",
        "empty",
        "unresolvable",
        "data_relocated",
        "ok",
    ]
    titles = {
        "absent": "ABSENT — genuinely not downloaded",
        "archive_only": "ARCHIVE_ONLY — downloaded but never extracted",
        "metadata_only": "METADATA_ONLY — failed / placeholder download",
        "empty": "EMPTY — directory exists, no files",
        "unresolvable": "UNRESOLVABLE — manifest has no usable data_root",
        "data_relocated": "DATA_RELOCATED — present, but data_root is wrong",
        "ok": "OK",
    }
    for v in order:
        rows = by_verdict.get(v, [])
        if not rows:
            continue
        lines.append(f"## {titles[v]} ({len(rows)})\n")
        for r in sorted(rows, key=lambda x: x.dataset_id):
            extra = []
            if r.status_flag:
                extra.append(f"status={r.status}→**{r.status_flag}**")
            if r.found_at and r.found_at != r.declared_root:
                extra.append(f"data at `{r.found_at}`")
            if r.records_missing:
                extra.append(f"{r.records_missing}/{r.records_checked} sampled records missing")
            tail = ("; ".join(extra) + " — " if extra else "") + r.note
            lines.append(
                f"- **{r.dataset_id}** ({r.data_files}d/{r.archive_files}a/{r.other_files}o) — {tail}"
            )
        lines.append("")

    if report.uncovered:
        lines.append(
            f"## NO_MANIFEST — on-disk data with no covering manifest ({len(report.uncovered)})\n"
        )
        for r in sorted(report.uncovered, key=lambda x: x.dataset_id):
            lines.append(f"- **{r.declared_root}** ({r.data_files} data files) — {r.note}")
        lines.append("")
    return "\n".join(lines)
