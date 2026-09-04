"""Guard: container entry points reference ``spectramr.cli``, not stale ``src.cli``.

After the 2026-05 ``src → spectramr`` refactor the Docker ``ENTRYPOINT`` still read
``python -m src.cli`` — so every ``docker run spectramr:… <verb>`` failed with
``No module named src``. The launcher's ``DockerBackend`` appends ``<verb>
--config …`` to this entrypoint, so a regression here silently breaks
``spectramr launch --where docker``. This static check fails fast if the stale
path ever returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_CONTAINER_DIR = Path(__file__).resolve().parents[3] / "scripts" / "container"


@pytest.mark.parametrize("name", ["Dockerfile", "spectramr.def"])
def test_no_stale_src_cli_reference(name):
    path = _CONTAINER_DIR / name
    if not path.exists():  # pragma: no cover - defensive
        pytest.skip(f"{name} not present")
    text = path.read_text(encoding="utf-8")
    assert "src.cli" not in text, f"{name} still references the stale 'src.cli' module path"


def test_dockerfile_entrypoint_uses_spectramr_cli():
    text = (_CONTAINER_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert '"spectramr.cli"' in text, "Dockerfile ENTRYPOINT must invoke spectramr.cli"
