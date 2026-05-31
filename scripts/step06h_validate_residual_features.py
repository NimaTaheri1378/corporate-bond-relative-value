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


CHECK_COLUMNS = [
    "issuer_date_residual_z",
    "issuer_date_bucket_residual_z",
    "is_cheap_vs_curve",
    "is_rich_vs_curve",
    "curve_unstable",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_one(row: dict[str, Any], read_feature_counts: bool) -> dict[str, Any]:
    path = Path(str(row["output_path"]))
    expected_rows = int(float(row.get("rows", 0) or 0))

    rec: dict[str, Any] = {
        "universe": str(row.get("universe", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "output_path": str(path),
        "residuals_input_path": str(row.get("residuals_input_path", "")),
        "expected_rows": expected_rows,
        "exists": path.exists(),
        "footer_rows": None,
        "file_size_bytes": None,
        "schema_sha256": None,
        "row_count_match": False,
        "non_null_issuer_date_z": None,
        "non_null_bucket_z": None,
        "cheap_rows": None,
        "rich_rows": None,
        "unstable_curve_rows": None,
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

        if read_feature_counts and rec["footer_rows"] > 0:
            available = set(pf.schema_arrow.names)
            selected = [c for c in CHECK_COLUMNS if c in available]

            if selected:
                df = pf.read(columns=selected).to_pandas()

                if "issuer_date_residual_z" in df:
                    rec["non_null_issuer_date_z"] = int(df["issuer_date_residual_z"].notna().sum())
                if "issuer_date_bucket_residual_z" in df:
                    rec["non_null_bucket_z"] = int(df["issuer_date_bucket_residual_z"].notna().sum())
                if "is_cheap_vs_curve" in df:
                    rec["cheap_rows"] = int(df["is_cheap_vs_curve"].fillna(False).astype(bool).sum())
                if "is_rich_vs_curve" in df:
                    rec["rich_rows"] = int(df["is_rich_vs_curve"].fillna(False).astype(bool).sum())
                if "curve_unstable" in df:
                    rec["unstable_curve_rows"] = int(df["curve_unstable"].fillna(False).astype(bool).sum())

        rec["ok"] = bool(rec["row_count_match"] and rec["error"] == "")
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def int_sum(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 06H validate residual-feature outputs. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-feature-counts", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_residual_features_v1_{args.universe}_manifest.csv"
    )
    nonempty_manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_residual_features_v1_{args.universe}_nonempty_manifest.csv"
    )
    step06g_summary_path = root / "artifacts" / "tables" / "step06g_residual_feature_summary.json"
    step06g_detail_path = root / "artifacts" / "tables" / "step06g_residual_feature_partition_summary.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 06G manifest: {manifest_path}")
    if not step06g_summary_path.exists():
        raise FileNotFoundError(f"Missing Step 06G summary: {step06g_summary_path}")

    manifest = pd.read_csv(manifest_path)
    step06g_summary = load_json(step06g_summary_path)

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions_to_validate={len(rows)}")
    print(f"workers={args.workers}")
    print(f"read_feature_counts={not args.skip_feature_counts}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(validate_one, row, not args.skip_feature_counts)
            for row in rows
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % 100 == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                footer_rows = sum(int(r.get("footer_rows") or 0) for r in results)
                z_rows = sum(int(r.get("non_null_issuer_date_z") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} footer_rows={footer_rows:,} z_rows={z_rows:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)

    expected_rows = int(pd.to_numeric(manifest["rows"], errors="coerce").fillna(0).sum())
    footer_rows = int_sum(detail, "footer_rows")

    missing_files = int((~detail["exists"].astype(bool)).sum())
    row_mismatches = int((~detail["row_count_match"].astype(bool)).sum())
    read_errors = int(detail["error"].fillna("").astype(str).ne("").sum())

    feature_counts_checked = not args.skip_feature_counts

    computed_non_null_z = int_sum(detail, "non_null_issuer_date_z")
    computed_bucket_z = int_sum(detail, "non_null_bucket_z")
    computed_cheap = int_sum(detail, "cheap_rows")
    computed_rich = int_sum(detail, "rich_rows")
    computed_unstable = int_sum(detail, "unstable_curve_rows")

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    validated_manifest = manifest.merge(
        detail.loc[:, ["output_path", "footer_rows", "file_size_bytes", "schema_sha256", "row_count_match"]],
        on="output_path",
        how="left",
    )

    validated_manifest_path = manifest_dir / f"curve_residual_features_v1_{args.universe}_validated_manifest.csv"
    validated_nonempty_manifest_path = manifest_dir / f"curve_residual_features_v1_{args.universe}_validated_nonempty_manifest.csv"

    validated_manifest.to_csv(validated_manifest_path, index=False)
    validated_manifest.loc[
        pd.to_numeric(validated_manifest["footer_rows"], errors="coerce").fillna(0) > 0
    ].to_csv(validated_nonempty_manifest_path, index=False)

    detail_path = table_dir / f"step06h_residual_features_{args.universe}_validation_detail.csv"
    validation_path = table_dir / f"step06h_residual_features_{args.universe}_validation.json"

    detail.to_csv(detail_path, index=False)

    summary = {
        "ok": (
            missing_files == 0
            and row_mismatches == 0
            and read_errors == 0
            and footer_rows == expected_rows
            and footer_rows == int(step06g_summary["rows_written"])
            and (
                not feature_counts_checked
                or (
                    computed_non_null_z == int(step06g_summary["non_null_issuer_date_z"])
                    and computed_bucket_z == int(step06g_summary["non_null_bucket_z"])
                    and computed_cheap == int(step06g_summary["cheap_rows"])
                    and computed_rich == int(step06g_summary["rich_rows"])
                    and computed_unstable == int(step06g_summary["unstable_curve_rows"])
                )
            )
        ),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "partitions_validated": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()),
        "missing_files": missing_files,
        "read_errors": read_errors,
        "row_count_mismatches": row_mismatches,
        "expected_rows_from_manifest": expected_rows,
        "footer_rows": footer_rows,
        "footer_minus_manifest": int(footer_rows - expected_rows),
        "step06g_rows_written": int(step06g_summary["rows_written"]),
        "feature_counts_checked": feature_counts_checked,
        "non_null_issuer_date_z": computed_non_null_z if feature_counts_checked else None,
        "non_null_issuer_date_z_minus_step06g": (
            computed_non_null_z - int(step06g_summary["non_null_issuer_date_z"])
            if feature_counts_checked else None
        ),
        "non_null_bucket_z": computed_bucket_z if feature_counts_checked else None,
        "non_null_bucket_z_minus_step06g": (
            computed_bucket_z - int(step06g_summary["non_null_bucket_z"])
            if feature_counts_checked else None
        ),
        "cheap_rows": computed_cheap if feature_counts_checked else None,
        "cheap_rows_minus_step06g": (
            computed_cheap - int(step06g_summary["cheap_rows"])
            if feature_counts_checked else None
        ),
        "rich_rows": computed_rich if feature_counts_checked else None,
        "rich_rows_minus_step06g": (
            computed_rich - int(step06g_summary["rich_rows"])
            if feature_counts_checked else None
        ),
        "unstable_curve_rows": computed_unstable if feature_counts_checked else None,
        "unstable_curve_rows_minus_step06g": (
            computed_unstable - int(step06g_summary["unstable_curve_rows"])
            if feature_counts_checked else None
        ),
        "schema_fingerprints": {
            str(k): int(v)
            for k, v in detail["schema_sha256"].fillna("MISSING").value_counts().items()
        },
        "validated_manifest": str(validated_manifest_path),
        "validated_nonempty_manifest": str(validated_nonempty_manifest_path),
        "detail_path": str(detail_path),
        "source_step06g_summary": str(step06g_summary_path),
        "source_step06g_detail": str(step06g_detail_path),
        "note": "Local-only residual-feature validation. Upload bundle only, not feature parquet.",
    }

    write_json(validation_path, summary)

    bundle = log_dir / f"step06h_residual_feature_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            validation_path,
            detail_path,
            validated_manifest_path,
            validated_nonempty_manifest_path,
            step06g_summary_path,
        ]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION={validation_path}")
    print(f"DETAIL={detail_path}")
    print(f"VALIDATED_MANIFEST={validated_manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
