"""Small safe-output helpers for isolated study artifacts."""

from pathlib import Path


def create_output_directory(path: Path) -> Path:
    """Create an output path exclusively; fail rather than overwrite a run."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
