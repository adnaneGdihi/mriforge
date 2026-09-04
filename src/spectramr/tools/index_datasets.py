"""
Dataset Indexing Tool.

CLI tool for scanning HDF5 datasets and ensuring a lightweight pickle index
for efficient data loading without traversing the filesystem at runtime.
"""

import pickle
from pathlib import Path

import h5py
import typer
from tqdm import tqdm

app = typer.Typer()


@app.command()
def create_index(
    data_root: str = typer.Option(..., help="Path to H5 files root"),
    output_path: str = typer.Option(
        "data/manifests/fastmri_knee.pkl", help="Path to save pickle index"
    ),
    pattern: str = typer.Option("**/*.h5", help="Glob pattern to match files"),
    use_relative_paths: bool = typer.Option(
        True, help="Store relative paths (recommended for portability)"
    ),
):
    """Scans H5 files and creates a lightweight pickle index.

    Stores paths relative to data_root for cross-machine portability.
    The data loader will resolve these relative to the configured data_root.
    """
    root = Path(data_root).resolve()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root}...")
    files = list(root.glob(pattern))

    index = []
    print("Reading metadata...")
    for p in tqdm(files):
        try:
            with h5py.File(p, "r") as f:
                if "kspace" in f:
                    kspace_shape = f["kspace"].shape

                    # Store relative path for portability
                    if use_relative_paths:
                        rel_path = str(p.relative_to(root))
                    else:
                        rel_path = str(p.absolute())

                    # [METADATA EXTRACTION]
                    # Extract site/scanner info for Leave-One-Site-Out validation
                    metadata = {}

                    # 1. Direct Attributes
                    for key in [
                        "scanner",
                        "site",
                        "institution",
                        "systemVendor",
                        "systemModel",
                        "patient_id",
                    ]:
                        if key in f.attrs:
                            metadata[key] = str(f.attrs[key])

                    # 2. ISMRMRD Header Parsing (if available).
                    # FastMRI usually stores the header in an 'xml' dataset or an
                    # 'ismrmrd_header' attribute. (The previous ``f.dataset``
                    # probe raised AttributeError on every file — h5py.File has
                    # no ``.dataset`` — and was silently skipped.)
                    header_xml = None
                    if "xml" in f:
                        # It's a dataset
                        try:
                            xml_data = f["xml"][()]
                            if hasattr(xml_data, "item"):  # numpy scalar
                                header_xml = xml_data.item()
                            else:
                                header_xml = xml_data[0]  # byte string usually
                        except Exception:
                            # A bare ``except:`` swallowed KeyboardInterrupt /
                            # SystemExit too; narrow it. A malformed xml dataset
                            # just yields no header.
                            header_xml = None
                    elif "ismrmrd_header" in f.attrs:
                        header_xml = f.attrs["ismrmrd_header"]

                    if header_xml:
                        try:
                            import xml.etree.ElementTree as ET

                            if isinstance(header_xml, bytes):
                                header_xml = header_xml.decode("utf-8", errors="ignore")

                            # Remove garbage chars if any
                            header_xml = header_xml.strip()
                            # Parse XML
                            # ISMRMRD uses namespace, but we'll try relaxed find
                            xml_root = ET.fromstring(header_xml)

                            # Helper to find text
                            def find_text(node, tag):
                                # Try with namespace and without
                                """find_text.

                                Args:
                                    node (Any): Description.
                                    tag (Any): Description.
                                Returns:
                                    Any: Description.
                                """
                                ns = {"n": "http://www.ismrm.org/ISMRMRD"}
                                res = node.find(f".//n:{tag}", ns)
                                if res is None:
                                    res = node.find(f".//{tag}")
                                return res.text if res is not None else None

                            for tag in [
                                "institutionName",
                                "systemVendor",
                                "systemModel",
                                "receiverChannels",
                            ]:
                                val = find_text(xml_root, tag)
                                if val:
                                    metadata[tag] = val
                        except Exception:
                            # print(f"XML parse error for {p}: {e}")
                            pass

                    index.append(
                        {
                            "path": rel_path,
                            "shape": kspace_shape,
                            "filename": p.name,
                            "file_id": p.stem,
                            "metadata": metadata,  # [NEW] Store extracted metadata
                        }
                    )
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    # Store metadata with index
    manifest = {
        "version": 2,
        "data_root": str(root),  # Original root for reference
        "use_relative": use_relative_paths,
        "files": index,
    }

    with open(output, "wb") as f:
        pickle.dump(manifest, f)

    print(f"Saved index of {len(index)} files to {output}")
    if use_relative_paths:
        print(f"Paths are relative to: {root}")


if __name__ == "__main__":
    app()
