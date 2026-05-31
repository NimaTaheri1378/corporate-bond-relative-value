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
import pyarrow.parquet as pq


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def footer_one(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["output_path"]))
    expected_rows = int(float(row.get("issue_date_rows", 0) or 0))

    rec: dict[str, Any] = {
        "universe": str(row.get("universe", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "output_path": str(path),
        "source_panel_path": str(row.get("source_panel_path", "")),
        "manifest_issue_date_rows": expected_rows,
        "manifest_curve_ready_rows": int(float(row.get("curve_ready_rows", 0) or 0)),
        "manifest_usable_rows": int(float(row.get("usable_rows", 0) or 0)),
        "exists": path.exists(),
        "footer_rows": None,
        "file_size_bytes": None,
        "schema_sha256": None,
        "row_count_match": False,
        "error": "",
        "ok": False,
    }

    if not path.exists():
        rec["error"] = "missing_file"
        return rec

    try:
        pf = pq.ParquetFile(path)
        schema_text = str(pf.schema_arrow)
        rec["footer_rows"] = int(pf.metadata.num_rows)
        rec["file_size_bytes"] = int(path.stat().st_size)
        rec["schema_sha256"] = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
        rec["row_count_match"] = rec["footer_rows"] == expected_rows
        rec["ok"] = bool(rec["row_count_match"])
        return rec
    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 06D validate curve input outputs. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    curve_manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_inputs_v1_{args.universe}_manifest.csv"
    )
    curve_nonempty_manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_inputs_v1_{args.universe}_nonempty_manifest.csv"
    )
    input_curve_ready_manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"trace_fisd_panel_v1_{args.universe}_curve_ready_manifest.csv"
    )
    step06c_summary_path = root / "artifacts" / "tables" / "step06c_curve_inputs_summary.json"
    step06c_detail_path = root / "artifacts" / "tables" / "step06c_curve_inputs_partition_summary.csv"

    for p in [curve_manifest_path, input_curve_ready_manifest_path, step06c_summary_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    curve_manifest = pd.read_csv(curve_manifest_path)
    input_manifest = pd.read_csv(input_curve_ready_manifest_path)
    step06c_summary = json.loads(step06c_summary_path.read_text())

    # Attach original source curve-ready counts to the curve-input manifest.
    src = input_manifest.loc[:, ["output_path", "curve_ready_rows"]].copy()
    src["curve_ready_rows"] = pd.to_numeric(src["curve_ready_rows"], errors="coerce").fillna(0).astype("int64")
    src = src.rename(
        columns={
            "output_path": "source_panel_path",
            "curve_ready_rows": "source_curve_ready_rows",
        }
    )

    corrected = curve_manifest.merge(src, on="source_panel_path", how="left")
    corrected["source_curve_ready_rows"] = pd.to_numeric(
        corrected["source_curve_ready_rows"], errors="coerce"
    ).fillna(0).astype("int64")

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = corrected.to_dict("records")
    results = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions_to_validate={len(rows)}")
    print(f"workers={args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(footer_one, row) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % 100 == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                footer_rows = sum(int(r.get("footer_rows") or 0) for r in results)
                print(f"progress {i}/{len(futures)} ok={ok} footer_rows={footer_rows:,}", flush=True)

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)

    corrected_validated_manifest = corrected.copy()
    footer_counts = detail.loc[:, ["output_path", "footer_rows", "file_size_bytes", "schema_sha256", "row_count_match"]]
    corrected_validated_manifest = corrected_validated_manifest.merge(footer_counts, on="output_path", how="left")

    corrected_manifest_path = manifest_dir / f"curve_inputs_v1_{args.universe}_validated_manifest.csv"
    corrected_nonempty_path = manifest_dir / f"curve_inputs_v1_{args.universe}_validated_nonempty_manifest.csv"

    corrected_validated_manifest.to_csv(corrected_manifest_path, index=False)
    corrected_validated_manifest.loc[
        pd.to_numeric(corrected_validated_manifest["footer_rows"], errors="coerce").fillna(0) > 0
    ].to_csv(corrected_nonempty_path, index=False)

    expected_issue_date_rows = int(pd.to_numeric(corrected["issue_date_rows"], errors="coerce").fillna(0).sum())
    footer_rows = int(pd.to_numeric(detail["footer_rows"], errors="coerce").fillna(0).sum())

    original_curve_ready_expected = int(pd.to_numeric(input_manifest["curve_ready_rows"], errors="coerce").fillna(0).sum())
    corrected_source_curve_ready = int(pd.to_numeric(corrected["source_curve_ready_rows"], errors="coerce").fillna(0).sum())

    missing_files = int((~detail["exists"].astype(bool)).sum())
    row_mismatches = int((~detail["row_count_match"].astype(bool)).sum())
    errors = int(detail["error"].fillna("").astype(str).ne("").sum())

    schema_counts = detail["schema_sha256"].fillna("MISSING").value_counts().to_dict()

    summary = {
        "ok": (
            missing_files == 0
            and row_mismatches == 0
            and errors == 0
            and footer_rows == expected_issue_date_rows
            and corrected_source_curve_ready == original_curve_ready_expected
        ),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "partitions_validated": int(len(detail)),
        "missing_files": missing_files,
        "read_errors": errors,
        "row_count_mismatches": row_mismatches,
        "issue_date_rows_from_manifest": expected_issue_date_rows,
        "footer_rows": footer_rows,
        "footer_minus_manifest_issue_date_rows": int(footer_rows - expected_issue_date_rows),
        "source_curve_ready_rows_corrected": corrected_source_curve_ready,
        "source_curve_ready_rows_original_manifest": original_curve_ready_expected,
        "source_curve_ready_minus_original": int(corrected_source_curve_ready - original_curve_ready_expected),
        "step06c_reported_curve_ready_rows_read": int(step06c_summary.get("curve_ready_rows_read", 0)),
        "step06c_skipped_existing": int(step06c_summary.get("skipped_existing", 0)),
        "issue_date_rows": int(step06c_summary.get("issue_date_rows", expected_issue_date_rows)),
        "usable_rows_reported_step06c": int(step06c_summary.get("usable_rows", 0)),
        "schema_fingerprints": {str(k): int(v) for k, v in schema_counts.items()},
        "validated_manifest": str(corrected_manifest_path),
        "validated_nonempty_manifest": str(corrected_nonempty_path),
        "detail_path": str(table_dir / "step06d_curve_inputs_validation_detail.csv"),
        "source_step06c_summary": str(step06c_summary_path),
        "source_step06c_detail": str(step06c_detail_path),
        "note": "Local-only validation/fix of curve-input manifest counters. No WRDS. No parquet upload.",
    }

    detail_path = table_dir / "step06d_curve_inputs_validation_detail.csv"
    validation_path = table_dir / "step06d_curve_inputs_validation.json"

    detail.to_csv(detail_path, index=False)
    write_json(validation_path, summary)

    bundle = log_dir / f"step06d_curve_inputs_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            validation_path,
            detail_path,
            corrected_manifest_path,
            corrected_nonempty_path,
            step06c_summary_path,
        ]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION={validation_path}")
    print(f"DETAIL={detail_path}")
    print(f"VALIDATED_MANIFEST={corrected_manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
