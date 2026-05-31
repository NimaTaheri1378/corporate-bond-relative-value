#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_DIR_PREFIXES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "run_logs",
    "logs",
    ".cache",
    ".wrds",
    "artifacts/model_cards",
]

FORBIDDEN_SUFFIXES = [
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
    ".bib",
    ".qmd",
}

SCRIPT_ALLOWLIST = [
    # Public safety and docs.
    ("scripts/step09a_public_safety_audit.py", "release", "Public safety audit: checks secrets, forbidden public files, and tracked forbidden files."),
    ("scripts/step09b_generate_public_docs.py", "release", "Generates README, docs, and public aggregate assets."),
    ("scripts/step09c_prepare_public_push.py", "release", "Stages only public-safe reproducibility files by allowlist."),

    # TRACE cleaning / validation.
    ("scripts/step04a_trace_metadata_audit.py", "trace-clean", "Raw TRACE footer/metadata audit."),
    ("scripts/step04d_trace_code_policy_audit.py", "trace-clean", "TRACE code-policy audit."),
    ("scripts/step04e_trace_policy_retention_audit.py", "trace-clean", "Cleaning policy retention audit."),
    ("scripts/step04f_trace_cleaner.py", "trace-clean", "Builds clean TRACE universes."),
    ("scripts/step04g_validate_clean_outputs.py", "trace-clean", "Validates clean TRACE outputs."),
    ("scripts/step04h_build_clean_trace_manifests.py", "trace-clean", "Builds final clean TRACE manifests."),

    # Security master / joins.
    ("scripts/step05a_security_master_local_qa.py", "security-master", "Local non-TRACE/FISD/security-master QA."),
    ("scripts/step05b_identifier_coverage_audit.py", "security-master", "TRACE/FISD/return/link identifier coverage audit."),
    ("scripts/step05c_build_fisd_security_master.py", "security-master", "Builds local FISD issue/rating/amount dimensions."),
    ("scripts/step05d_trace_fisd_join_audit.py", "security-master", "Row-level TRACE-to-FISD join audit."),
    ("scripts/step05e_trace_fisd_join_panel_smoke.py", "security-master", "Smoke joined TRACE-FISD panel."),
    ("scripts/step05f_build_trace_fisd_panel.py", "security-master", "Builds stable TRACE-FISD processed panel."),
    ("scripts/step05g_validate_trace_fisd_panel.py", "security-master", "Validates stable TRACE-FISD panel."),
    ("scripts/step05h_curve_ready_universe_audit.py", "security-master", "Curve-ready issuer-date support audit."),

    # Curves and residual features.
    ("scripts/step06a_curve_fit_smoke.py", "curves", "Nelson-Siegel curve-fit smoke."),
    ("scripts/step06b_curve_model_comparison_smoke.py", "curves", "Guarded curve-family model comparison smoke."),
    ("scripts/step06c_build_curve_inputs.py", "curves", "Builds issue-date curve input aggregates."),
    ("scripts/step06d_validate_curve_inputs.py", "curves", "Validates curve-input outputs."),
    ("scripts/step06e_curve_fit_residuals.py", "curves", "Fits guarded curves and residuals."),
    ("scripts/step06f_validate_curve_fit_outputs.py", "curves", "Validates curve-fit outputs."),
    ("scripts/step06g_residual_feature_smoke.py", "features", "Builds residual feature layer."),
    ("scripts/step06h_validate_residual_features.py", "features", "Validates residual feature outputs."),

    # Labels, models, backtests, robustness.
    ("scripts/step07a_label_source_audit.py", "labels", "Audits return-label sources."),
    ("scripts/step07b_monthly_label_matrix_smoke.py", "labels", "Builds monthly label matrix."),
    ("scripts/step07c_validate_monthly_label_matrix.py", "labels", "Validates/promotes monthly label matrix."),
    ("scripts/step08a_monthly_baseline_ridge.py", "models", "Transparent ridge/signal baseline."),
    ("scripts/step08b_monthly_signal_backtest.py", "backtest", "Cost-aware residual-signal decile backtest."),
    ("scripts/step08c_gpu_mlp_monthly.py", "models", "GPU PyTorch MLP model."),
    ("scripts/step08d_lightgbm_monthly.py", "models", "CPU LightGBM tabular model."),
    ("scripts/step08e_model_leaderboard_and_figures.py", "results", "Builds model leaderboard and figures."),
    ("scripts/step08f_signal_robustness_exposure.py", "robustness", "Maturity/liquidity robustness and exposure audit."),
    ("scripts/step08g_rating_amount_exposure_audit.py", "robustness", "As-of rating and amount-outstanding exposure audit."),
]


FIXED_PUBLIC_FILES = [
    "README.md",
    "DATA_ACCESS.md",
    "LICENSE",
    ".gitignore",
    "pyproject.toml",
    "environment.yml",
    "Makefile",
    "mkdocs.yml",
]


PUBLIC_DIRS = [
    "docs",
    "configs",
    "src",
    "tests",
    ".github/workflows",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def run_cmd(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc


def is_forbidden_path(relpath: str) -> bool:
    relpath = relpath.replace("\\", "/")
    lower = relpath.lower()

    for prefix in FORBIDDEN_DIR_PREFIXES:
        if relpath == prefix or relpath.startswith(prefix + "/"):
            return True

    for suffix in FORBIDDEN_SUFFIXES:
        if lower.endswith(suffix):
            return True

    return False


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", "Makefile"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_file_for_secrets(path: Path, root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if not is_text_file(path):
        return findings

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return [{"path": rel(path, root), "pattern": "read_error", "detail": repr(exc)}]

    for name, pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            snippet = text[start:end].replace("\n", " ")
            findings.append(
                {
                    "path": rel(path, root),
                    "pattern": name,
                    "detail": snippet[:260],
                }
            )

    return findings


def iter_public_dir_files(root: Path, dirname: str) -> list[Path]:
    base = root / dirname
    if not base.exists():
        return []

    out: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        r = rel(p, root)
        if "__pycache__" in r or ".pytest_cache" in r or ".ipynb_checkpoints" in r:
            continue
        if is_forbidden_path(r):
            continue
        out.append(p)
    return out


def build_script_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for script, stage, description in SCRIPT_ALLOWLIST:
        p = root / script
        rows.append(
            {
                "script": script,
                "stage": stage,
                "description": description,
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else 0,
                "public_safe_reason": "Source code only; no raw WRDS data or local Parquet outputs.",
            }
        )
    return pd.DataFrame(rows)


def compile_scripts(root: Path, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for _, row in manifest.iterrows():
        script = str(row["script"])
        if not bool(row["exists"]):
            continue
        proc = run_cmd(
            [sys.executable, "-m", "py_compile", script],
            cwd=root,
            check=False,
        )
        results.append(
            {
                "script": script,
                "compile_ok": proc.returncode == 0,
                "stderr": proc.stderr.strip()[-500:],
            }
        )
    return results


def collect_public_files(root: Path) -> list[Path]:
    files: list[Path] = []

    for f in FIXED_PUBLIC_FILES:
        p = root / f
        if p.exists() and p.is_file():
            files.append(p)

    for d in PUBLIC_DIRS:
        files.extend(iter_public_dir_files(root, d))

    for script, _, _ in SCRIPT_ALLOWLIST:
        p = root / script
        if p.exists() and p.is_file():
            files.append(p)

    # Explicit public aggregate outputs copied by Step 09B.
    for pattern in [
        "docs/assets/tables/*.csv",
        "docs/assets/tables/*.json",
        "docs/assets/figures/*.png",
        "docs/assets/figures/*.jpg",
        "docs/assets/figures/*.svg",
        "docs/assets/interactive/*.html",
    ]:
        files.extend(root.glob(pattern))

    # De-duplicate and sort.
    unique = {}
    for p in files:
        if p.exists() and p.is_file():
            unique[rel(p, root)] = p

    return [unique[k] for k in sorted(unique)]


def write_reproducibility_docs(root: Path, manifest: pd.DataFrame) -> list[Path]:
    docs_dir = root / "docs"
    table_dir = root / "docs" / "assets" / "tables"
    docs_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest_public = manifest.copy()
    manifest_public.to_csv(table_dir / "public_reproducibility_scripts.csv", index=False)

    existing = manifest_public.loc[manifest_public["exists"]].copy()

    md_lines = [
        "# Reproducibility",
        "",
        "This public repository includes the code path needed to reproduce the research pipeline with the user's own licensed WRDS access.",
        "",
        "The public repository intentionally excludes raw WRDS/vendor data, row-level Parquet panels, credentials, model binaries, and cluster logs.",
        "",
        "## Public script manifest",
        "",
        "| Stage | Script | Purpose |",
        "|---|---|---|",
    ]

    for _, row in existing.iterrows():
        md_lines.append(
            f"| {row['stage']} | `{row['script']}` | {row['description']} |"
        )

    md_lines.extend(
        [
            "",
            "## Rebuild outline",
            "",
            "```bash",
            "# 1. Configure licensed WRDS credentials locally; do not commit credentials.",
            "# 2. Run extraction and local QA scripts on your own WRDS account.",
            "# 3. Build clean TRACE, FISD joins, curve inputs, curves, residual features, labels, models, and robustness outputs.",
            "# 4. Run the public safety audit before committing.",
            "python scripts/step09a_public_safety_audit.py --workspace .",
            "```",
            "",
            "## Public-safe artifacts",
            "",
            "Aggregate result tables and figures live under `docs/assets/`. They are public-safe summaries, not raw vendor data.",
            "",
        ]
    )

    repro_path = docs_dir / "reproducibility.md"
    repro_path.write_text("\n".join(md_lines), encoding="utf-8")

    return [repro_path, table_dir / "public_reproducibility_scripts.csv"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 09C prepare safe reproducible public push.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--stage", action="store_true", help="Stage public-safe allowlisted files.")
    parser.add_argument("--commit", action="store_true", help="Commit staged public-safe files.")
    parser.add_argument("--push", action="store_true", help="Push commit to the configured remote.")
    parser.add_argument("--message", default="Public-safe research release: corporate bond relative value")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_script_manifest(root)
    repro_paths = write_reproducibility_docs(root, manifest)

    compile_results = compile_scripts(root, manifest)
    compile_bad = [r for r in compile_results if not r["compile_ok"]]

    public_files = collect_public_files(root)

    forbidden_candidates = [
        rel(p, root) for p in public_files
        if is_forbidden_path(rel(p, root))
    ]

    secret_findings: list[dict[str, Any]] = []
    for p in public_files:
        secret_findings.extend(scan_file_for_secrets(p, root))

    if compile_bad:
        print("COMPILE FAILURES:")
        print(json.dumps(compile_bad, indent=2))
        raise SystemExit(1)

    if forbidden_candidates:
        print("FORBIDDEN PUBLIC CANDIDATES:")
        print(json.dumps(forbidden_candidates, indent=2))
        raise SystemExit(1)

    if secret_findings:
        print("SECRET FINDINGS:")
        print(json.dumps(secret_findings, indent=2))
        raise SystemExit(1)

    manifest_path = table_dir / "step09c_public_push_file_manifest.csv"
    compile_path = table_dir / "step09c_public_push_compile_checks.json"
    summary_path = table_dir / "step09c_public_push_summary.json"

    push_files_df = pd.DataFrame(
        [{"path": rel(p, root), "size_bytes": p.stat().st_size} for p in public_files]
    ).sort_values("path")

    push_files_df.to_csv(manifest_path, index=False)
    compile_path.write_text(json.dumps(compile_results, indent=2, sort_keys=True), encoding="utf-8")

    staged_files: list[str] = []
    staged_forbidden: list[str] = []

    if args.stage or args.commit or args.push:
        add_paths = [rel(p, root) for p in public_files]
        run_cmd(["git", "add", "--", *add_paths], cwd=root, check=True)

        proc = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=root, check=True)
        staged_files = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        staged_forbidden = [x for x in staged_files if is_forbidden_path(x)]

        if staged_forbidden:
            print("STAGED FORBIDDEN FILES:")
            print(json.dumps(staged_forbidden, indent=2))
            raise SystemExit(1)

    summary = {
        "ok": True,
        "run_id": run_id,
        "workspace": str(root),
        "public_files_count": len(public_files),
        "script_allowlist_count": int(len(manifest)),
        "scripts_existing_count": int(manifest["exists"].sum()),
        "scripts_missing_optional": manifest.loc[~manifest["exists"], "script"].tolist(),
        "compile_checks": {
            "n_checked": len(compile_results),
            "n_failed": len(compile_bad),
        },
        "forbidden_public_candidates": forbidden_candidates,
        "secret_findings": secret_findings,
        "staged": bool(args.stage or args.commit or args.push),
        "staged_files_count": len(staged_files),
        "staged_forbidden_files": staged_forbidden,
        "commit_requested": bool(args.commit),
        "push_requested": bool(args.push),
        "manifest_path": str(manifest_path),
        "reproducibility_docs": [str(p) for p in repro_paths],
        "policy": "Allowlisted code/docs/configs/tests/public aggregate assets only. No data/, run_logs/, Parquet, credentials, model binaries, or archives.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.commit:
        proc = run_cmd(["git", "status", "--porcelain"], cwd=root, check=True)
        print(proc.stdout)

        run_cmd(["git", "commit", "-m", args.message], cwd=root, check=True)

    if args.push:
        run_cmd(["git", "push"], cwd=root, check=True)

    bundle = log_dir / f"step09c_public_push_manifest_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, manifest_path, compile_path, *repro_paths]:
            if p.exists():
                tar.add(p, arcname=rel(p, root))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

