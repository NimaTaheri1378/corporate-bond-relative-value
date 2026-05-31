#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def footer_rows(path: Path) -> tuple[bool, int | None, int | None, str | None, str]:
    if not path.exists():
        return False, None, None, None, "missing_file"

    try:
        pf = pq.ParquetFile(path)
        schema_text = str(pf.schema_arrow)
        import hashlib

        schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
        return True, int(pf.metadata.num_rows), int(path.stat().st_size), schema_hash, ""
    except Exception as exc:
        return True, None, None, None, repr(exc)


def check_one(row: dict[str, Any]) -> dict[str, Any]:
    ext_path = Path(str(row["extended_output_path"]))
    pub_path = Path(str(row["core_public_output_path"]))

    ext_exists, ext_footer, ext_size, ext_schema, ext_error = footer_rows(ext_path)
    pub_exists, pub_footer, pub_size, pub_schema, pub_error = footer_rows(pub_path)

    expected_ext = int(float(row.get("extended_rows", 0) or 0))
    expected_pub = int(float(row.get("core_public_rows", 0) or 0))

    return {
        "start_date": row.get("start_date", ""),
        "end_date": row.get("end_date", ""),
        "source_stage": row.get("source_stage", ""),
        "raw_rows": int(float(row.get("raw_rows", 0) or 0)),
        "extended_output_path": str(ext_path),
        "core_public_output_path": str(pub_path),
        "extended_expected_rows": expected_ext,
        "core_public_expected_rows": expected_pub,
        "extended_exists": ext_exists,
        "core_public_exists": pub_exists,
        "extended_footer_rows": ext_footer,
        "core_public_footer_rows": pub_footer,
        "extended_file_size_bytes": ext_size,
        "core_public_file_size_bytes": pub_size,
        "extended_schema_sha256": ext_schema,
        "core_public_schema_sha256": pub_schema,
        "extended_error": ext_error,
        "core_public_error": pub_error,
        "extended_row_match": ext_footer == expected_ext,
        "core_public_row_match": pub_footer == expected_pub,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 04G validate cleaned TRACE outputs. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    summary_path = root / "artifacts/tables/step04f_trace_clean_summary.json"
    part_path = root / "artifacts/tables/step04f_trace_clean_partition_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Step 04F summary: {summary_path}")
    if not part_path.exists():
        raise FileNotFoundError(f"Missing Step 04F partition summary: {part_path}")

    step04f = json.loads(summary_path.read_text())
    parts = pd.read_csv(part_path)

    rows = parts.to_dict("records")
    results = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"partitions_to_validate={len(rows)}")
    print(f"workers={args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(check_one, row) for row in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)
            if i == 1 or i % 100 == 0 or i == len(futures):
                ok_ext = sum(1 for r in results if r["extended_row_match"])
                ok_pub = sum(1 for r in results if r["core_public_row_match"])
                print(f"progress {i}/{len(futures)} extended_ok={ok_ext} core_public_ok={ok_pub}", flush=True)

    out = pd.DataFrame(results).sort_values(["start_date", "end_date", "extended_output_path"])

    ext_expected = int(out["extended_expected_rows"].sum())
    pub_expected = int(out["core_public_expected_rows"].sum())
    ext_footer = int(pd.to_numeric(out["extended_footer_rows"], errors="coerce").fillna(0).sum())
    pub_footer = int(pd.to_numeric(out["core_public_footer_rows"], errors="coerce").fillna(0).sum())

    ext_missing = int((~out["extended_exists"].astype(bool)).sum())
    pub_missing = int((~out["core_public_exists"].astype(bool)).sum())
    ext_mismatch = int((~out["extended_row_match"].astype(bool)).sum())
    pub_mismatch = int((~out["core_public_row_match"].astype(bool)).sum())

    ext_errors = int(out["extended_error"].fillna("").astype(str).ne("").sum())
    pub_errors = int(out["core_public_error"].fillna("").astype(str).ne("").sum())

    ext_schema_counts = out["extended_schema_sha256"].fillna("MISSING").value_counts().to_dict()
    pub_schema_counts = out["core_public_schema_sha256"].fillna("MISSING").value_counts().to_dict()

    summary = {
        "ok": (
            ext_missing == 0
            and pub_missing == 0
            and ext_mismatch == 0
            and pub_mismatch == 0
            and ext_errors == 0
            and pub_errors == 0
            and ext_footer == int(step04f["extended_regular_rows"])
            and pub_footer == int(step04f["core_public_rows"])
        ),
        "run_id": run_id,
        "workspace": str(root),
        "partitions_validated": int(len(out)),
        "extended_expected_rows_from_partition_summary": ext_expected,
        "extended_footer_rows": ext_footer,
        "extended_footer_minus_expected": int(ext_footer - ext_expected),
        "extended_footer_minus_step04f_summary": int(ext_footer - int(step04f["extended_regular_rows"])),
        "extended_missing_files": ext_missing,
        "extended_row_mismatches": ext_mismatch,
        "extended_read_errors": ext_errors,
        "extended_schema_fingerprints": ext_schema_counts,
        "core_public_expected_rows_from_partition_summary": pub_expected,
        "core_public_footer_rows": pub_footer,
        "core_public_footer_minus_expected": int(pub_footer - pub_expected),
        "core_public_footer_minus_step04f_summary": int(pub_footer - int(step04f["core_public_rows"])),
        "core_public_missing_files": pub_missing,
        "core_public_row_mismatches": pub_mismatch,
        "core_public_read_errors": pub_errors,
        "core_public_schema_fingerprints": pub_schema_counts,
    }

    table_dir = root / "artifacts/tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    detail_path = table_dir / "step04g_clean_output_footer_validation_detail.csv"
    validation_path = table_dir / "step04g_clean_output_footer_validation.json"

    out.to_csv(detail_path, index=False)
    validation_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bundle = log_dir / f"step04g_clean_output_footer_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [validation_path, detail_path, summary_path, part_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION={validation_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
