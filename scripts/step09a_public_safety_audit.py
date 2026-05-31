#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_PUBLIC_DIR_PREFIXES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "run_logs",
    "logs",
    ".cache",
    ".wrds",
    "artifacts/model_cards",
]

FORBIDDEN_PUBLIC_SUFFIXES = [
    ".parquet",
    ".feather",
    ".arrow",
    ".duckdb",
    ".sqlite",
    ".db",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".gz",
    ".7z",
    ".pem",
    ".key",
    ".pgpass",
    ".env",
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".txt",
    ".sh",
    ".cfg",
    ".ini",
    ".gitignore",
}

SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    (
        "password_assignment",
        re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    ),
    (
        "api_secret_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-./+=]{16,}['\"]"
        ),
    ),
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def is_forbidden_public_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")

    for prefix in FORBIDDEN_PUBLIC_DIR_PREFIXES:
        if rel == prefix or rel.startswith(prefix + "/"):
            return True

    lower = rel.lower()
    for suffix in FORBIDDEN_PUBLIC_SUFFIXES:
        if lower.endswith(suffix):
            return True

    return False


def should_skip_repo_scan(rel: str) -> bool:
    rel = rel.replace("\\", "/")

    skip_prefixes = [
        ".git/",
        "data/raw/",
        "data/interim/",
        "data/processed/",
        "run_logs/",
        "logs/",
        "artifacts/model_cards/",
        ".venv/",
        "venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    ]

    return any(rel.startswith(prefix) for prefix in skip_prefixes)


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", "Makefile"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_text_file(path: Path, root: Path) -> list[dict[str, Any]]:
    rel = str(path.relative_to(root)).replace("\\", "/")
    findings: list[dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return [{"path": rel, "pattern": "read_error", "detail": repr(exc)}]

    for name, pattern in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            snippet = text[start:end].replace("\n", " ")
            findings.append(
                {
                    "path": rel,
                    "pattern": name,
                    "detail": snippet[:240],
                }
            )

    return findings


def git_ls_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    except Exception:
        return []


def collect_public_candidate_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []

    fixed = [
        "README.md",
        "DATA_ACCESS.md",
        "LICENSE",
        ".gitignore",
        "pyproject.toml",
        "environment.yml",
        "Makefile",
        "mkdocs.yml",
    ]

    for rel in fixed:
        p = root / rel
        if p.exists():
            candidates.append(p)

    for pattern in [
        "docs/**/*.md",
        "docs/assets/tables/*.csv",
        "docs/assets/tables/*.json",
        "docs/assets/figures/*.png",
        "docs/assets/figures/*.jpg",
        "docs/assets/figures/*.svg",
        "docs/assets/interactive/*.html",
        "artifacts/tables/step08e_model_leaderboard.csv",
        "artifacts/tables/step08f_signal_robustness_metrics.csv",
        "artifacts/tables/step08g_rating_amount_exposure_metrics.csv",
        "artifacts/tables/step08g_rating_amount_exposure_summary.csv",
        "artifacts/figures_static/step08e_*.png",
        "artifacts/figures_interactive/step08e_*.html",
    ]:
        candidates.extend(root.glob(pattern))

    out = []
    seen = set()
    for p in candidates:
        if p.exists() and p.is_file():
            rel = str(p.relative_to(root))
            if rel not in seen:
                seen.add(rel)
                out.append(p)

    return out


def inventory_risky_local_files(root: Path, max_sample: int = 100) -> tuple[int, list[str]]:
    risky = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel.startswith(".git/"):
            continue

        if is_forbidden_public_path(rel):
            risky.append(rel)

    return len(risky), risky[:max_sample]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 09A public safety audit. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    secret_findings: list[dict[str, Any]] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel = str(p.relative_to(root)).replace("\\", "/")

        if should_skip_repo_scan(rel):
            continue

        if p.name == ".pgpass" or rel.endswith(".pgpass"):
            secret_findings.append(
                {"path": rel, "pattern": "pgpass_file", "detail": "credential file name"}
            )
            continue

        if is_text_file(p):
            secret_findings.extend(scan_text_file(p, root))

    tracked = git_ls_files(root)

    tracked_forbidden = [
        rel for rel in tracked
        if is_forbidden_public_path(rel)
    ]

    public_candidates = collect_public_candidate_paths(root)

    public_forbidden = [
        str(p.relative_to(root)).replace("\\", "/")
        for p in public_candidates
        if is_forbidden_public_path(str(p.relative_to(root)).replace("\\", "/"))
    ]

    risky_count, risky_sample = inventory_risky_local_files(root)

    audit = {
        "ok": (
            len(secret_findings) == 0
            and len(tracked_forbidden) == 0
            and len(public_forbidden) == 0
        ),
        "run_id": run_id,
        "workspace": str(root),
        "secret_findings": secret_findings,
        "tracked_forbidden_files": tracked_forbidden,
        "public_candidate_forbidden_files": public_forbidden,
        "public_candidate_files_count": len(public_candidates),
        "public_candidate_files": [
            str(p.relative_to(root)).replace("\\", "/")
            for p in public_candidates
        ],
        "risky_local_files_count": risky_count,
        "risky_local_files_sample": risky_sample,
        "policy": {
            "allowed_public": [
                "code",
                "README and docs",
                "aggregate result tables",
                "aggregate figures",
                "interactive HTML built from aggregate summaries",
                "sanitized manifests",
                "tests and CI configuration",
            ],
            "forbidden_public": [
                "raw WRDS/vendor data",
                "interim or processed row-level Parquet panels",
                "credentials, actual .pgpass files, API keys, passwords, tokens",
                "model binary/state files",
                "cluster logs",
                "archives containing private artifacts",
            ],
        },
        "note": "Local risky files may exist during research; the audit fails only if secrets, tracked forbidden files, or public-candidate forbidden files are found.",
    }

    checks = pd.DataFrame(
        [
            {
                "check": "secret_scan",
                "ok": len(secret_findings) == 0,
                "n_findings": len(secret_findings),
            },
            {
                "check": "git_tracked_forbidden_files",
                "ok": len(tracked_forbidden) == 0,
                "n_findings": len(tracked_forbidden),
            },
            {
                "check": "public_candidate_forbidden_files",
                "ok": len(public_forbidden) == 0,
                "n_findings": len(public_forbidden),
            },
            {
                "check": "risky_local_file_inventory",
                "ok": True,
                "n_findings": risky_count,
            },
        ]
    )

    audit_json = table_dir / "step09a_public_safety_audit.json"
    audit_csv = table_dir / "step09a_public_safety_audit_checks.csv"

    write_json(audit_json, audit)
    checks.to_csv(audit_csv, index=False)

    bundle = log_dir / f"step09a_public_safety_audit_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(audit_json, arcname=str(audit_json.relative_to(root)))
        tar.add(audit_csv, arcname=str(audit_csv.relative_to(root)))
        if (root / ".gitignore").exists():
            tar.add(root / ".gitignore", arcname=".gitignore")

    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"AUDIT_JSON={audit_json}")
    print(f"AUDIT_CSV={audit_csv}")
    print(f"BUNDLE={bundle}")

    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
