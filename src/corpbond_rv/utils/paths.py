from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    """Return the repository root by walking upward until pyproject.toml is found."""
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate project root from {cur}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
