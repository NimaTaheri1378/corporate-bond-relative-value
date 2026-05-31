#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_EXPECTED_TRACE_ROWS = 310_324_418


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Required CSV is empty: {path}")
    return df


def numeric_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def normalize_manifest(df: pd.DataFrame, source_stage: str, source_manifest: str) -> pd.DataFrame:
    out = df.copy()
    out["source_stage"] = source_stage
    out["source_manifest"] = source_manifest

    if "ok" not in out.columns:
        out["ok"] = True

    for col in ["n_rows", "expected_rows", "file_size_bytes", "rows_per_sec", "elapsed_sec"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["start_date", "end_date", "output_path"]:
        if col not in out.columns:
            out[col] = ""

    return out


def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.loc[truthy_series(df["ok"])].copy()


def load_step03_manifests(root: Path, scale_run_id: str, resume_run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_dir = root / "data" / "manifests" / "extractions"

    scale_path = manifest_dir / f"step03c_trace_{scale_run_id}.csv"
    scale_validation_path = manifest_dir / f"step03c_trace_{scale_run_id}_validation.json"

    resume_path = manifest_dir / f"step03f_trace_{resume_run_id}.csv"
    resume_validation_path = manifest_dir / f"step03f_trace_{resume_run_id}_validation.json"

    step03h_paths = sorted(
        p
        for p in manifest_dir.glob("step03c_trace_step03h_20101015_retry_*.csv")
        if not p.name.endswith("_plan.csv")
    )

    if not step03h_paths:
        raise FileNotFoundError(
            f"No Step 03h manifest found under {manifest_dir}/step03c_trace_step03h_20101015_retry_*.csv"
        )

    scale = normalize_manifest(read_csv_required(scale_path), "step03e_scale_ok", str(scale_path))
    resume = normalize_manifest(read_csv_required(resume_path), "step03f_resume_ok", str(resume_path))

    h_frames = []
    for path in step03h_paths:
        h_frames.append(normalize_manifest(read_csv_required(path), "step03h_final_retry_ok", str(path)))
    step03h = pd.concat(h_frames, ignore_index=True)

    combined = pd.concat([ok_rows(scale), ok_rows(resume), ok_rows(step03h)], ignore_index=True)

    if "output_path" not in combined.columns:
        raise ValueError("Combined manifests do not contain output_path.")

    before_dedup = len(combined)
    combined = combined.drop_duplicates("output_path", keep="last").reset_index(drop=True)

    metadata = {
        "scale_manifest": str(scale_path),
        "scale_validation": str(scale_validation_path),
        "resume_manifest": str(resume_path),
        "resume_validation": str(resume_validation_path),
        "step03h_manifests": [str(p) for p in step03h_paths],
        "rows_before_dedup": before_dedup,
        "duplicates_dropped_by_output_path": before_dedup - len(combined),
        "scale_total_rows": numeric_sum(scale, "n_rows"),
        "scale_ok_rows": numeric_sum(ok_rows(scale), "n_rows"),
        "resume_total_rows": numeric_sum(resume, "n_rows"),
        "resume_ok_rows": numeric_sum(ok_rows(resume), "n_rows"),
        "step03h_total_rows": numeric_sum(step03h, "n_rows"),
        "step03h_ok_rows": numeric_sum(ok_rows(step03h), "n_rows"),
        "scale_total_tasks": int(len(scale)),
        "scale_ok_tasks": int(len(ok_rows(scale))),
        "resume_total_tasks": int(len(resume)),
        "resume_ok_tasks": int(len(ok_rows(resume))),
        "step03h_total_tasks": int(len(step03h)),
        "step03h_ok_tasks": int(len(ok_rows(step03h))),
    }

    return combined, metadata


def parquet_footer(path_str: str, expected_rows: int | None = None) -> dict[str, Any]:
    path = Path(path_str)
    rec: dict[str, Any] = {
        "output_path": path_str,
        "exists": path.exists(),
        "file_size_bytes_actual": None,
        "footer_rows": None,
        "expected_rows_manifest": expected_rows,
        "row_count_match": False,
        "schema_sha256": None,
        "schema_text": None,
        "metadata_error": "",
    }

    if not path.exists():
        rec["metadata_error"] = "missing_file"
        return rec

    try:
        import pyarrow.parquet as pq

        rec["file_size_bytes_actual"] = int(path.stat().st_size)
        pf = pq.ParquetFile(path)
        rec["footer_rows"] = int(pf.metadata.num_rows)

        schema_text = str(pf.schema_arrow)
        rec["schema_text"] = schema_text
        rec["schema_sha256"] = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()

        if expected_rows is not None:
            rec["row_count_match"] = int(expected_rows) == rec["footer_rows"]
        else:
            rec["row_count_match"] = True

    except Exception as exc:
        rec["metadata_error"] = repr(exc)

    return rec


def check_parquet_metadata(final_manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    tasks: list[tuple[str, int | None]] = []

    for _, row in final_manifest.iterrows():
        output_path = str(row.get("output_path", ""))
        if not output_path:
            continue
        n_rows_raw = row.get("n_rows")
        expected: int | None
        try:
            expected = int(n_rows_raw)
        except Exception:
            expected = None
        tasks.append((output_path, expected))

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(parquet_footer, p, e) for p, e in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())

    return pd.DataFrame(rows)


def duplicate_window_count(df: pd.DataFrame) -> int:
    cols = [c for c in ["start_date", "end_date"] if c in df.columns]
    if len(cols) != 2:
        return 0
    return int(df.duplicated(cols, keep=False).sum())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def package_outputs(root: Path, output_paths: list[Path], run_id: str) -> Path:
    bundle = root / "run_logs" / f"step04a_trace_metadata_audit_bundle_{run_id}.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle, "w:gz") as tar:
        for path in output_paths:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(root)))

    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 04A local-only TRACE metadata audit. No WRDS. No row scan."
    )
    parser.add_argument(
        "--workspace",
        default="/home/nt612/github/Corporate Bond Relative-Value",
        help="Project workspace path.",
    )
    parser.add_argument(
        "--scale-run-id",
        default="scale_20260528T222052Z",
        help="Step 03e scale run id.",
    )
    parser.add_argument(
        "--resume-run-id",
        default="resume_20260529T161449Z",
        help="Step 03f resume run id.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=PROJECT_EXPECTED_TRACE_ROWS,
        help="Final expected TRACE row count.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Threads for Parquet footer checks.",
    )
    parser.add_argument(
        "--skip-parquet-metadata",
        action="store_true",
        help="Skip Parquet footer checks.",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Create a tar.gz review bundle.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    if not root.exists():
        raise FileNotFoundError(f"Workspace does not exist: {root}")

    artifacts_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "extractions"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    final_manifest, source_meta = load_step03_manifests(
        root=root,
        scale_run_id=args.scale_run_id,
        resume_run_id=args.resume_run_id,
    )

    final_manifest_path = manifest_dir / "step04a_trace_final_manifest.csv"
    final_manifest.to_csv(final_manifest_path, index=False)

    final_rows = numeric_sum(final_manifest, "n_rows")
    final_expected_rows = numeric_sum(final_manifest, "expected_rows")

    output_paths = final_manifest["output_path"].dropna().astype(str).tolist()
    missing_files = [p for p in output_paths if not Path(p).exists()]

    metadata_df = pd.DataFrame()
    metadata_summary: dict[str, Any] = {
        "metadata_checked": False,
        "metadata_ok": None,
        "metadata_files_checked": 0,
        "metadata_missing_files": len(missing_files),
        "metadata_row_mismatches": None,
        "metadata_errors": None,
        "schema_fingerprints": {},
    }

    metadata_inventory_path = artifacts_dir / "step04a_trace_parquet_metadata_inventory.csv"

    if not args.skip_parquet_metadata:
        metadata_df = check_parquet_metadata(final_manifest, workers=args.workers)
        metadata_df.to_csv(metadata_inventory_path, index=False)

        if not metadata_df.empty:
            metadata_missing = int((metadata_df["exists"] == False).sum())  # noqa: E712
            metadata_errors = int(metadata_df["metadata_error"].fillna("").astype(str).ne("").sum())
            metadata_row_mismatches = int((metadata_df["row_count_match"] == False).sum())  # noqa: E712
            schema_counts = (
                metadata_df["schema_sha256"]
                .fillna("MISSING_OR_ERROR")
                .value_counts()
                .head(20)
                .to_dict()
            )
        else:
            metadata_missing = 0
            metadata_errors = 0
            metadata_row_mismatches = 0
            schema_counts = {}

        metadata_summary = {
            "metadata_checked": True,
            "metadata_ok": metadata_missing == 0 and metadata_errors == 0 and metadata_row_mismatches == 0,
            "metadata_files_checked": int(len(metadata_df)),
            "metadata_missing_files": metadata_missing,
            "metadata_row_mismatches": metadata_row_mismatches,
            "metadata_errors": metadata_errors,
            "schema_fingerprints": schema_counts,
        }

    summary = {
        "run_id": run_id,
        "ok": (
            final_rows == int(args.expected_rows)
            and len(missing_files) == 0
            and (
                args.skip_parquet_metadata
                or metadata_summary.get("metadata_ok") is True
            )
        ),
        "workspace": str(root),
        "workspace_realpath": str(root.resolve()),
        "scale_run_id": args.scale_run_id,
        "resume_run_id": args.resume_run_id,
        "expected_rows_project": int(args.expected_rows),
        "final_rows": int(final_rows),
        "final_expected_rows_from_manifests": int(final_expected_rows),
        "final_rows_minus_project_expected": int(final_rows - int(args.expected_rows)),
        "final_rows_minus_manifest_expected": int(final_rows - final_expected_rows),
        "combined_partitions": int(len(final_manifest)),
        "duplicate_date_window_rows": duplicate_window_count(final_manifest),
        "missing_files": int(len(missing_files)),
        "missing_files_sample": missing_files[:10],
        **source_meta,
        **metadata_summary,
    }

    summary_path = artifacts_dir / "step04a_trace_metadata_audit.json"
    write_json(summary_path, summary)

    row_counts_path = artifacts_dir / "step04a_trace_partition_row_counts.csv"
    cols = [
        c
        for c in [
            "source_stage",
            "start_date",
            "end_date",
            "n_rows",
            "expected_rows",
            "file_size_bytes",
            "output_path",
            "source_manifest",
        ]
        if c in final_manifest.columns
    ]
    final_manifest.loc[:, cols].to_csv(row_counts_path, index=False)

    output_files = [
        summary_path,
        row_counts_path,
        final_manifest_path,
    ]

    if metadata_inventory_path.exists():
        output_files.append(metadata_inventory_path)

    bundle_path: Path | None = None
    if args.package:
        bundle_path = package_outputs(root, output_files, run_id)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"FINAL_MANIFEST={final_manifest_path}")
    print(f"ROW_COUNTS={row_counts_path}")
    if metadata_inventory_path.exists():
        print(f"METADATA_INVENTORY={metadata_inventory_path}")
    if bundle_path is not None:
        print(f"BUNDLE={bundle_path}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
