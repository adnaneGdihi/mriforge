"""Manifest service implementation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from spectramr.domain.interfaces.service_interfaces import IManifestService


class ManifestService(IManifestService):
    """Service for managing dataset processing manifests.

    Provides functionality to create, save, load, and validate manifests
    that track preprocessing operations and artifacts.
    """

    def __init__(self, manifests_dir: str = "./manifests"):
        """Initialize the manifest service.

        Args:
            manifests_dir: Directory to store manifest files

        """
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}",
        )

    def create_manifest(
        self,
        dataset_name: str,
        processing_tasks: list[str],
        artifacts: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a manifest for a dataset processing operation.

        Args:
            dataset_name: Name of the dataset being processed
            processing_tasks: List of processing tasks performed
            artifacts: Dictionary of artifacts created/paths
            metadata: Optional additional metadata

        Returns:
            Complete manifest dictionary

        """
        timestamp = datetime.now().isoformat()

        manifest = {
            "dataset_name": dataset_name,
            "processing_tasks": processing_tasks,
            "artifacts": artifacts,
            "timestamp": timestamp,
            "version": "1.0",
            "metadata": metadata or {},
        }

        # Add command line arguments if available
        try:
            import sys

            manifest["command_args"] = sys.argv
        except (ImportError, AttributeError):
            manifest["command_args"] = []

        return manifest

    def save_manifest(
        self,
        manifest: dict[str, Any],
        output_path: str,
    ) -> str:
        """Save a manifest to file.

        Args:
            manifest: Manifest dictionary to save
            output_path: Path where to save the manifest

        Returns:
            Path to the saved manifest file

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Manifest saved to {output_path}")
        return str(output_path)

    def load_manifest(self, manifest_path: str) -> dict[str, Any]:
        """Load a manifest from file.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            Loaded manifest dictionary

        Raises:
            FileNotFoundError: If manifest file doesn't exist

        """
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        return manifest

    def list_manifests(self, dataset_name: str | None = None) -> list[str]:
        """List available manifests, optionally filtered by dataset.

        Args:
            dataset_name: Optional dataset name to filter by

        Returns:
            List of manifest file paths

        """
        if not self.manifests_dir.exists():
            return []

        manifests = []
        for manifest_file in self.manifests_dir.rglob("*.json"):
            if dataset_name:
                # Check if this manifest belongs to the requested dataset
                try:
                    manifest = self.load_manifest(str(manifest_file))
                    if manifest.get("dataset_name") == dataset_name:
                        manifests.append(str(manifest_file))
                except (json.JSONDecodeError, KeyError):
                    continue
            else:
                manifests.append(str(manifest_file))

        return sorted(manifests)

    def validate_manifest(self, manifest: dict[str, Any]) -> bool:
        """Validate manifest structure and content.

        Args:
            manifest: Manifest dictionary to validate

        Returns:
            True if manifest is valid, False otherwise

        """
        required_keys = ["dataset_name", "processing_tasks", "artifacts", "timestamp"]

        # Check required keys
        for key in required_keys:
            if key not in manifest:
                self.logger.warning(f"Manifest missing required key: {key}")
                return False

        # Validate data types
        if not isinstance(manifest["dataset_name"], str):
            self.logger.warning("dataset_name must be a string")
            return False

        if not isinstance(manifest["processing_tasks"], list):
            self.logger.warning("processing_tasks must be a list")
            return False

        if not isinstance(manifest["artifacts"], dict):
            self.logger.warning("artifacts must be a dictionary")
            return False

        # Validate timestamp format
        try:
            datetime.fromisoformat(manifest["timestamp"])
        except (ValueError, TypeError):
            self.logger.warning("Invalid timestamp format")
            return False

        return True
