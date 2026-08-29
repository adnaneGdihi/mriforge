from pathlib import Path

TRUSTED_ROOTS = {"experiments", "checkpoints", "artifacts"}


def validate_file_path(path: str | Path) -> Path:
    """Validate that ``path`` resolves *inside* one of the project's trusted
    roots (``experiments/``, ``checkpoints/``, ``artifacts/``).

    The previous implementation tested ``root in resolved.parts`` — a
    *name-membership* check, not a containment check. That accepted any absolute
    path merely containing a trusted-named component (``/etc/experiments/passwd``)
    and never rejected ``..`` traversal. A security validator that silently
    accepts an untrusted path manufactures false confidence (pitfall #9), so this
    now (a) rejects ``..`` / ``\\`` traversal sequences pre-resolution, (b)
    requires the resolved path to live under the project root (cwd), and (c)
    requires its first project-relative component to be a trusted root.

    Args:
        path: The candidate file path (relative paths resolve against cwd).

    Returns:
        The resolved, validated absolute ``Path``.

    Raises:
        ValueError: If traversal is detected or the path is not contained under
            a trusted root within the project tree.
    """
    raw = str(path)
    if ".." in raw or "\\" in raw:
        raise ValueError(f"Path traversal detected: {raw}")

    base = Path.cwd().resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()

    if not resolved.is_relative_to(base):
        raise ValueError(f"Untrusted path (outside project root {base}): {resolved}")

    rel_parts = resolved.relative_to(base).parts
    if not rel_parts or rel_parts[0] not in TRUSTED_ROOTS:
        raise ValueError(f"Untrusted path (not under {sorted(TRUSTED_ROOTS)}): {resolved}")
    return resolved
