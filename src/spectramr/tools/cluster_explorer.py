import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(no_args_is_help=True)


# Constants - Dynamic path resolution
def get_project_root() -> Path:
    """Auto-detect project root by finding .git or pyproject.toml"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to parent of parent (src/tools -> src -> root)
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
# Override via SPECTRAMR_CLUSTER_ROOT if you run on an HPC mount whose absolute
# prefix differs from the project root.
CLUSTER_ROOT = Path(os.environ.get("SPECTRAMR_CLUSTER_ROOT", str(PROJECT_ROOT)))
LOCAL_ROOT = PROJECT_ROOT  # Use discovered root instead of a hardcoded path.
DATABASES_DIR = PROJECT_ROOT / "databases"


def resolve_path(path_str: str, base_paths: list[Path]) -> Path | None:
    """Tries to resolve a path string against common bases or via remapping."""
    if not path_str:
        return None

    p = Path(path_str)

    # 1. Direct check
    if p.exists():
        return p

    # 2. Local to Cluster Remap
    path_str_abs = str(p.absolute())
    if str(LOCAL_ROOT) in path_str_abs:
        remapped = Path(path_str_abs.replace(str(LOCAL_ROOT), str(CLUSTER_ROOT), 1))
        if remapped.exists():
            return remapped

    # 2.a Be aggressive for common prefixes if not absolute
    if not p.is_absolute() and CLUSTER_ROOT.exists():
        parts = p.parts
        if parts and parts[0] in ("databases", "data"):
            remapped = CLUSTER_ROOT / p
            if remapped.exists():
                return remapped

    # 2.b Handle paths that contain LOCAL_ROOT but are not necessarily absolute
    if str(LOCAL_ROOT) in path_str and CLUSTER_ROOT.exists():
        remapped = Path(path_str.replace(str(LOCAL_ROOT), str(CLUSTER_ROOT), 1))
        if remapped.exists():
            return remapped

    # 3. Check relative to search bases
    for base in base_paths:
        if (base / path_str).exists():
            return base / path_str
        # Strip potential 'databases/' or 'data/' prefix if already in base
        parts = p.parts
        if parts and parts[0] in ("data", "databases"):
            sub_p = Path(*parts[1:])
            if (base / sub_p).exists():
                return base / sub_p

    return None


def check_experiment(config_path: Path) -> dict[str, Any]:
    """Audit a single experiment config for data readiness."""
    report = {
        "name": config_path.name,
        "path": str(config_path),
        "status": "OK",
        "errors": [],
        "warnings": [],
        "dataset_type": "unknown",
        "samples_found": 0,
        "total_samples": 0,
    }

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        report["status"] = "FAIL"
        report["errors"].append(f"YAML Parse Error: {e}")
        return report

    data_cfg = config.get("data", {})
    if not data_cfg:
        report["status"] = "WARNING"
        report["warnings"].append("No 'data' section found.")
        return report

    dataset_type = data_cfg.get("dataset_type", "unknown")
    report["dataset_type"] = dataset_type

    if dataset_type == "synthetic":
        report["status"] = "OK (Synthetic)"
        return report

    # Search bases for resolution
    search_bases = [Path("."), CLUSTER_ROOT, DATABASES_DIR]
    if CLUSTER_ROOT.exists():
        search_bases.append(CLUSTER_ROOT / "databases")

    # 1. Check Data Root
    data_root_str = data_cfg.get("data_root")
    resolved_root = resolve_path(data_root_str, search_bases) if data_root_str else None

    # Root Inference (mimic ConsolidatedDatasetFactory)
    if not resolved_root and data_cfg.get("datasets"):
        # Accessing nested list from config
        datasets_list = data_cfg.get("datasets", [])
        if datasets_list and isinstance(datasets_list, list):
            ds_path = datasets_list[0].get("path")
            resolved_root = resolve_path(ds_path, search_bases)

    if not resolved_root and dataset_type != "synthetic":
        missing_root_error = f"Data root not found: {data_root_str}"
    else:
        missing_root_error = None

    # 2. Check Index Path (for FastMRI/H5)
    index_path_str = data_cfg.get("index_path")
    if index_path_str:
        resolved_index = resolve_path(index_path_str, search_bases)
        if not resolved_index:
            report["status"] = "FAIL"
            report["errors"].append(f"Index file not found: {index_path_str}")
        else:
            # Audit the index content
            try:
                # Support both .json (v3) and .pkl (v1/v2) manifests
                if str(resolved_index).endswith(".json"):
                    with open(resolved_index, encoding="utf-8") as f:
                        manifest = json.load(f)
                    files = manifest.get("records", [])
                    # Normalize 'relative_path' → 'path' for uniform checking
                    for rec in files:
                        if "relative_path" in rec and "path" not in rec:
                            rec["path"] = rec["relative_path"]
                    use_relative = True
                else:
                    with open(resolved_index, "rb") as f:
                        manifest = pickle.load(f)
                    files = manifest.get("files", []) if isinstance(manifest, dict) else manifest
                    use_relative = isinstance(manifest, dict) and manifest.get("use_relative")

                report["total_samples"] = len(files)

                # Fallback to manifest data_root if resolved_root was missing
                manifest_data_root = (
                    manifest.get("data_root") if isinstance(manifest, dict) else None
                )
                if not resolved_root and manifest_data_root:
                    resolved_root = resolve_path(manifest_data_root, search_bases)

                if not resolved_root and missing_root_error:
                    report["status"] = "FAIL"
                    report["errors"].append(missing_root_error)
                    missing_root_error = None  # clear so we don't add it again at the end

                # Check first 5 samples
                checked = 0
                found = 0
                for item in files[:5]:
                    checked += 1
                    p_val = item.get("path") if isinstance(item, dict) else str(item)
                    # Resolve relative in manifest
                    if use_relative and resolved_root:
                        res_p = resolved_root / p_val
                    else:
                        res_p = resolve_path(
                            p_val, [resolved_root] if resolved_root else search_bases
                        )

                    if res_p and res_p.exists():
                        found += 1

                report["samples_found"] = found
                if found == 0 and len(files) > 0:
                    report["status"] = "FAIL"
                    report["errors"].append(
                        f"File paths in index {index_path_str} are unreachable."
                    )
                elif found < checked:
                    report["warnings"].append(
                        f"Some samples in index are missing ({found}/{checked} checked)."
                    )
            except Exception as e:
                report["errors"].append(f"Index read error: {e}")

    # 3. Check NIfTI directories (for Translation)
    if dataset_type in ("nifti_paired", "paired_nifti", "contrast_aware_paired"):
        lr_dir = data_cfg.get("input_lr_dir")
        hr_dir = data_cfg.get("input_hr_dir")

        res_lr = resolve_path(lr_dir, [resolved_root] if resolved_root else search_bases)
        res_hr = resolve_path(hr_dir, [resolved_root] if resolved_root else search_bases)

        if not res_lr:
            report["status"] = "FAIL"
            report["errors"].append(f"Source dir not found: {lr_dir}")
        if not res_hr:
            report["status"] = "FAIL"
            report["errors"].append(f"Target dir not found: {hr_dir}")

    # Add missing root error if not cleared
    if missing_root_error:
        report["status"] = "FAIL"
        report["errors"].append(missing_root_error)
        missing_root_error = None

    return report


@app.command()
def info():
    """Display environment information for cluster runs."""
    console.print("[bold blue]Cluster Explorer Environment Information[/bold blue]")
    console.print(f"Cluster Root: {CLUSTER_ROOT}")
    console.print(f"Local Root:   {LOCAL_ROOT}")
    console.print(f"Databases:    {DATABASES_DIR}")
    console.print(
        f"Status:       [green]{'ON CLUSTER' if CLUSTER_ROOT.exists() else 'LOCAL MODE'}[/green]"
    )


@app.command(name="scan")
def scan_cmd(
    path: str = typer.Argument("experiments", help="Path to audit (file or directory)"),
    recursive: bool = typer.Option(True, help="Scan directories recursively"),
):
    """Scan and verify cluster readiness for experiments."""
    root = Path(path)
    if not root.exists():
        console.print(f"[red]Error: {path} does not exist.[/red]")
        return

    configs = []
    if root.is_file():
        configs.append(root)
    else:
        pattern = "**/*.yaml" if recursive else "*.yaml"
        configs.extend(list(root.glob(pattern)))

    console.print(f"Found {len(configs)} configuration files. Auditing...")

    results = []
    for cfg in track(configs, description="Verifying..."):
        results.append(check_experiment(cfg))

    # Output Table
    table = Table(title="Cluster Readiness Report")
    table.add_column("Experiment", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Issues", style="yellow")

    fails = 0
    for r in sorted(results, key=lambda x: x["status"]):
        status_style = (
            "green"
            if r["status"].startswith("OK")
            else "red"
            if r["status"] == "FAIL"
            else "yellow"
        )
        if r["status"] == "FAIL":
            fails += 1

        issues = "\n".join(r["errors"] + r["warnings"])
        samples_info = f"{r['total_samples']}" if r["total_samples"] > 0 else "-"

        table.add_row(
            r["name"],
            r["dataset_type"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            samples_info,
            issues,
        )

    console.print(table)

    if fails > 0:
        console.print(f"\n[red]CRITICAL: {fails} experiments failed readiness check.[/red]")
    else:
        console.print("\n[green]All checked experiments are ready for cluster runs![/green]")


@app.command(name="preprocess-status")
def preprocess_status_cmd(
    dataset: list[str] | None = typer.Option(
        None, "--dataset", "-d", help="Specific dataset(s) to check"
    ),
    databases_dir: str = typer.Option("databases", help="Base directory containing raw datasets"),
    check_dims: bool = typer.Option(False, "--check-dims", help="Check file dimensions (slower)"),
    max_depth: int = typer.Option(4, "--max-depth", help="Max directory depth to search"),
):
    """Check preprocessing status by examining output directories following preprocessing.py structure.

    Output structure per preprocessing.py:
      {input_dir}_image/
        ├── nifti_reconstructed/
        ├── registered/
        ├── coil_sensitivity/
        ├── kspace/
        ├── nifti_reconstructed/
        └── manifests/
    """
    console.print("[bold blue]Preprocessing Status Audit[/bold blue]")

    # Resolve databases directory
    search_bases = [Path("."), CLUSTER_ROOT, DATABASES_DIR]
    res_databases = resolve_path(databases_dir, search_bases)

    if not res_databases or not res_databases.exists():
        console.print(f"[red]Error: Databases directory not found: {databases_dir}[/red]")
        console.print(f"[dim]Searched: {search_bases}[/dim]")
        return

    console.print(f"[dim]Databases dir: {res_databases}[/dim]")

    # Discover datasets: recursively look for *_image directories
    if dataset:
        # User specified datasets - try to find their _image directories
        output_dirs = []
        for key in dataset:
            # Search recursively for the _image directory
            matches = list(res_databases.rglob(f"{key}_image"))
            if matches:
                output_dirs.append((key, matches[0]))
            else:
                console.print(f"[yellow]Dataset '{key}' not found[/yellow]")
    else:
        # Auto-discover: find all *_image directories
        console.print(f"[dim]Searching for preprocessed datasets (max depth {max_depth})...[/dim]")

        output_dirs = []

        def search_dirs(path: Path, depth: int = 0):
            """search_dirs.

            Args:
                path (Path): Description.
                depth (int): Description.
            Returns:
                Any: Description.
            """
            if depth > max_depth:
                return
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        if item.name.endswith("_image"):
                            # Found a preprocessed output directory
                            base_name = item.name.replace("_image", "")
                            output_dirs.append((base_name, item))
                        elif not item.name.startswith("."):
                            search_dirs(item, depth + 1)
            except PermissionError as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        search_dirs(res_databases)
        output_dirs = sorted(output_dirs, key=lambda x: x[0])

    if not output_dirs:
        console.print("[yellow]No preprocessed datasets found.[/yellow]")
        console.print(f"[dim]Looking for directories ending with '_image' in {res_databases}[/dim]")
        return

    console.print(f"Found {len(output_dirs)} preprocessed dataset(s)\n")

    # Audit each dataset
    for key, output_dir in output_dirs:
        input_dir = output_dir.parent / key  # Sibling directory

        console.print(f"[bold cyan]{'─' * 60}[/bold cyan]")
        console.print(f"[bold cyan]Dataset: {key}[/bold cyan]")
        console.print(f"[dim]  Path: {output_dir}[/dim]")

        # Check each subdirectory
        table = Table(show_header=True, header_style="bold")
        table.add_column("Directory", style="cyan", width=20)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Files", justify="right", width=8)
        table.add_column("Details", width=40)

        total_files = 0

        # Dynamically find actual subdirectories
        actual_subdirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
        actual_subdirs.sort()

        if not actual_subdirs:
            table.add_row("-", "⚠️", "0", "[yellow]No preprocessed subdirectories found[/yellow]")

        for subdir in actual_subdirs:
            subdir_path = output_dir / subdir

            # Count files
            files = list(subdir_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_files += file_count

            if file_count == 0:
                table.add_row(subdir, "⚠️", "0", "[yellow]Empty directory[/yellow]")
                continue

            # Get file type breakdown
            extensions = {}
            for f in files:
                if f.is_file():
                    ext = f.suffix.lower()
                    if ext == ".gz" and f.stem.endswith(".nii"):
                        ext = ".nii.gz"
                    extensions[ext] = extensions.get(ext, 0) + 1

            ext_summary = ", ".join(f"{count} {ext}" for ext, count in sorted(extensions.items()))

            # Optionally check dimensions
            dims_info = ""
            if check_dims and subdir in [
                "nifti_reconstructed",
                "registered",
                "isotropic",
                "normalized",
                "nifti_reconstructed",
            ]:
                sample_files = [f for f in files if f.is_file() and f.suffix in [".gz", ".nii"]]
                if sample_files:
                    try:
                        import nibabel as nib

                        sample = nib.load(str(sample_files[0]))
                        dims_info = f" | Shape: {sample.shape}"
                    except Exception:
                        dims_info = " | [dim]Could not read dims[/dim]"

            table.add_row(subdir, "✅", str(file_count), f"{ext_summary}{dims_info}")

        console.print(table)
        console.print(f"[bold]Total files: {total_files}[/bold]\n")

    console.print(f"[bold cyan]{'─' * 60}[/bold cyan]")
    console.print("[green]Audit complete.[/green]")


@app.command(name="list")
def list_cmd(
    dataset: list[str] | None = typer.Option(
        None, "--dataset", "-d", help="Specific dataset(s) to check"
    ),
    databases_dir: str = typer.Option("databases", help="Base directory containing raw datasets"),
    check_dims: bool = typer.Option(False, "--check-dims", help="Check file dimensions (slower)"),
    max_depth: int = typer.Option(4, "--max-depth", help="Max directory depth to search"),
):
    """Alias for preprocess-status. Lists audited tracked datasets and their artifacts."""
    preprocess_status_cmd(dataset, databases_dir, check_dims, max_depth)


if __name__ == "__main__":
    app()
