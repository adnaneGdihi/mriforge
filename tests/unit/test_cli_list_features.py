"""Unit tests for CLI list-features command."""

import json
import sys
from io import StringIO
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

from spectramr.cli import list_features


class TestListFeaturesCommand:
    """Test the list-features CLI command."""

    def test_list_features_json_output(self, tmp_path):
        """Test list-features with JSON output."""
        args = argparse.Namespace(
            module="models",
            format="json",
            output=None,
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0, "Command should return 0 on success"
            assert isinstance(output, str)
            # Verify it's valid JSON
            data = json.loads(output)
            assert isinstance(data, list)
            assert len(data) > 0
        finally:
            sys.stdout = old_stdout

    def test_list_features_markdown_output(self, tmp_path):
        """Test list-features with Markdown output."""
        args = argparse.Namespace(
            module="losses",
            format="markdown",
            output=None,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0
            assert isinstance(output, str)
            assert "## Loss Features" in output or "features" in output.lower()
        finally:
            sys.stdout = old_stdout

    def test_list_features_csv_output(self):
        """Test list-features with CSV output."""
        args = argparse.Namespace(
            module="metrics",
            format="csv",
            output=None,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0
            # CSV should have headers
            lines = output.strip().split("\n")
            assert len(lines) > 1, "CSV should have header and data rows"
        finally:
            sys.stdout = old_stdout

    def test_list_features_file_output(self, tmp_path):
        """Test list-features saving to file."""
        output_file = tmp_path / "features.md"

        args = argparse.Namespace(
            module="all",
            format="markdown",
            output=output_file,
        )

        result = list_features(args)

        assert result == 0
        assert output_file.exists(), f"Output file should exist at {output_file}"
        content = output_file.read_text()
        assert len(content) > 100, "Generated file should have substantial content"

    def test_list_features_strategies(self):
        """Test listing strategy features."""
        args = argparse.Namespace(
            module="strategies",
            format="json",
            output=None,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0
            data = json.loads(output)
            assert len(data) > 0
            # Check that at least one known strategy is in the list
            names = [f["name"] for f in data]
            assert "gan" in names or "diffusion" in names
        finally:
            sys.stdout = old_stdout

    def test_list_features_physics(self):
        """Test listing physics features."""
        args = argparse.Namespace(
            module="physics",
            format="json",
            output=None,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0
            data = json.loads(output)
            assert len(data) > 0
        finally:
            sys.stdout = old_stdout

    def test_list_features_services(self):
        """Test listing service features."""
        args = argparse.Namespace(
            module="services",
            format="json",
            output=None,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = list_features(args)
            output = sys.stdout.getvalue()

            assert result == 0
            data = json.loads(output)
            assert len(data) > 0
        finally:
            sys.stdout = old_stdout
