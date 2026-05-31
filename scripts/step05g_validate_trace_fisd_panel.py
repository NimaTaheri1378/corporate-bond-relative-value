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


def read_bool_true_count(path: Path, column: str) -> tuple[int | None, str]:
    try:
        pf = pq.ParquetFile(path)
        if column not in set(pf.schema_arrow.names):
            return None, f"missing_column:{column}"
        arr = pf.read(columns=[column]).to_pandas()[column]
        return int(arr.astype(bool).sum()), ""
    except Exception as exc:
        return None, repr(exc)


def validate_one(row: dict[str, Any], verify_curve_ready: bool) -> dict[str, Any]:
    path = Path(str(row["output_path"]))

    expected_rows = int(float(row.get("rows", 0) or 0))
    expected_matched = int(float(row.get("matched_rows", 0) or 0))
    expected_curve_ready = int(float(row.get("curve_ready_rows", 0) or 0))

    rec: dict[str, Any] = {
        "universe": str(row.get("universe", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "output_path": str(path),
        "source_clean_path": str(row.get("source_clean_path", "")),
        "expected_rows": expected_rows,
        "expected_matched_rows": expected_matched,
        "expected_curve_ready_rows": expected_curve_ready,
        "exists": path.exists(),
        "footer_rows": None,
        "file_size_bytes": None,
        "schema_sha256": None,
        "row_count_match": False,
        "curve_ready_true_count": None,
        "curve_ready_count_match": None,
        "error": "",
        "curve_ready_error": "",
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

        if verify_curve_ready:
            val, err = read_bool_true_count(path, "is_curve_ready")
            rec["curve_ready_true_count"] = val
            rec["curve_ready_error"] = err
            rec["curve_ready_count_match"] = val == expected_curve_ready
        else:
            rec["curve_ready_count_match"] = True

        rec["ok"] = bool(rec["row_count_match"] and rec["curve_ready_count_match"] and not rec["error"])
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 05G validate stable TRACE-FISD panel. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-curve-ready", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest_path = root / "data" / "manifests" / "processed" / f"trace_fisd_panel_v1_{args.universe}_manifest.csv"
    nonempty_manifest_path = root / "data" / "manifests" / "processed" / f"trace_fisd_panel_v1_{args.universe}_nonempty_manifest.csv"
    step05f_summary_path = root / "artifacts" / "tables" / "step05f_trace_fisd_panel_summary.json"
    step05f_detail_path = root / "artifacts" / "tables" / "step05f_trace_fisd_panel_detail.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 05F manifest: {manifest_path}")
    if not step05f_summary_path.exists():
        raise FileNotFoundError(f"Missing Step 05F summary: {step05f_summary_path}")

    manifest = pd.read_csv(manifest_path)
    step05f_summary = json.loads(step05f_summary_path.read_text())

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions_to_validate={len(rows)}")
    print(f"verify_curve_ready={args.verify_curve_ready}")
    print(f"workers={args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(validate_one, row, args.verify_curve_ready) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % 100 == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                footer_rows = sum(int(r.get("footer_rows") or 0) for r in results)
                curve_ready = sum(int(r.get("curve_ready_true_count") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} footer_rows={footer_rows:,} "
                    f"curve_ready_true={curve_ready:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)

    expected_rows = int(pd.to_numeric(manifest["rows"], errors="coerce").fillna(0).sum())
    expected_matched = int(pd.to_numeric(manifest["matched_rows"], errors="coerce").fillna(0).sum())
    expected_curve_ready = int(pd.to_numeric(manifest["curve_ready_rows"], errors="coerce").fillna(0).sum())

    footer_rows = int(pd.to_numeric(detail["footer_rows"], errors="coerce").fillna(0).sum())
    curve_ready_true = int(pd.to_numeric(detail["curve_ready_true_count"], errors="coerce").fillna(0).sum())

    row_mismatches = int((~detail["row_count_match"].astype(bool)).sum())
    curve_ready_mismatches = int((~detail["curve_ready_count_match"].astype(bool)).sum())
    missing_files = int((~detail["exists"].astype(bool)).sum())
    errors = int(detail["error"].fillna("").astype(str).ne("").sum())
    curve_ready_errors = int(detail["curve_ready_error"].fillna("").astype(str).ne("").sum())

    schema_counts = detail["schema_sha256"].fillna("MISSING").value_counts().to_dict()

    # Build a curve-ready manifest from Step 05F per-partition counts.
    curve_ready_manifest = manifest.loc[pd.to_numeric(manifest["curve_ready_rows"], errors="coerce").fillna(0) > 0].copy()
    curve_ready_manifest_path = root / "data" / "manifests" / "processed" / f"trace_fisd_panel_v1_{args.universe}_curve_ready_manifest.csv"
    curve_ready_manifest.to_csv(curve_ready_manifest_path, index=False)

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    detail_path = table_dir / f"step05g_trace_fisd_panel_{args.universe}_validation_detail.csv"
    validation_path = table_dir / f"step05g_trace_fisd_panel_{args.universe}_validation.json"

    detail.to_csv(detail_path, index=False)

    expected_from_step05f = step05f_summary["universe_summaries"][args.universe]

    summary = {
        "ok": (
            missing_files == 0
            and errors == 0
            and curve_ready_errors == 0
            and row_mismatches == 0
            and (not args.verify_curve_ready or curve_ready_mismatches == 0)
            and footer_rows == expected_rows
            and footer_rows == int(expected_from_step05f["rows_written"])
            and expected_curve_ready == int(expected_from_step05f["curve_ready_rows"])
        ),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "verify_curve_ready": bool(args.verify_curve_ready),
        "partitions_validated": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()),
        "missing_files": missing_files,
        "read_errors": errors,
        "curve_ready_read_errors": curve_ready_errors,
        "row_count_mismatches": row_mismatches,
        "curve_ready_count_mismatches": curve_ready_mismatches,
        "expected_rows_from_manifest": expected_rows,
        "footer_rows": footer_rows,
        "footer_minus_manifest": int(footer_rows - expected_rows),
        "expected_matched_rows_from_manifest": expected_matched,
        "expected_curve_ready_rows_from_manifest": expected_curve_ready,
        "curve_ready_true_count": curve_ready_true if args.verify_curve_ready else None,
        "curve_ready_true_minus_manifest": int(curve_ready_true - expected_curve_ready) if args.verify_curve_ready else None,
        "step05f_rows_written": int(expected_from_step05f["rows_written"]),
        "step05f_matched_rows": int(expected_from_step05f["matched_rows"]),
        "step05f_curve_ready_rows": int(expected_from_step05f["curve_ready_rows"]),
        "schema_fingerprints": {str(k): int(v) for k, v in schema_counts.items()},
        "manifest": str(manifest_path),
        "nonempty_manifest": str(nonempty_manifest_path),
        "curve_ready_manifest": str(curve_ready_manifest_path),
        "detail_path": str(detail_path),
        "source_summary": str(step05f_summary_path),
        "source_detail": str(step05f_detail_path),
        "note": "Local-only processed panel validation. Upload bundle only, not parquet.",
    }

    write_json(validation_path, summary)

    bundle = log_dir / f"step05g_trace_fisd_panel_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            validation_path,
            detail_path,
            manifest_path,
            nonempty_manifest_path,
            curve_ready_manifest_path,
            step05f_summary_path,
        ]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION={validation_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

