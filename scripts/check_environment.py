#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    root = Path.cwd()
    result = {
        "ok": True,
        "python": sys.version.split()[0],
        "cwd": str(root),
        "package_checks": {
            "numpy": package_available("numpy"),
            "pandas": package_available("pandas"),
            "pyarrow": package_available("pyarrow"),
            "sklearn": package_available("sklearn"),
            "plotly": package_available("plotly"),
        },
        "public_safety_checks": {
            "data_raw_exists": (root / "data" / "raw").exists(),
            "data_processed_exists": (root / "data" / "processed").exists(),
            "pgpass_in_repo": any(p.name == ".pgpass" for p in root.rglob("*") if ".git" not in p.parts),
        },
        "env": {
            "github_actions": os.getenv("GITHUB_ACTIONS", "false"),
        },
    }

    # Public CI should not need proprietary local data directories.
    if result["public_safety_checks"]["pgpass_in_repo"]:
        result["ok"] = False

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
