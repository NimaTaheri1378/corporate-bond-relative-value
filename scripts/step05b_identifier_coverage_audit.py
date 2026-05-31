#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


TRACE_ID_COLUMNS = ["cusip_id", "bond_sym_id", "company_symbol"]

RAW_TABLE_SPECS = {
    "fisd_issue_issuer": ["complete_cusip", "issue_cusip", "issuer_cusip", "issue_id", "issuer_id"],
    "bondret_monthly": ["cusip", "bond_sym_id", "bsym", "company_symbol", "issue_id"],
    "contrib_bond_returns": ["cusip", "permno"],
    "crsp_bond_link": ["cusip", "permno", "permco"],
    "fang_bond_firm_link": ["issuer_cusip", "permno", "permco", "gvkey"],
    "fisd_coupon_info": ["issue_id"],
    "fisd_rating_hist": ["issue_id"],
    "fisd_amount_outstanding": ["issue_id"],
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def norm_generic_value(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"<NA>", "NA", "NAN", "NONE", "NULL"}:
        return None
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def norm_cusip_value(x: Any) -> str | None:
    s = norm_generic_value(x)
    if s is None:
        return None
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or None


def norm_series_values(s: pd.Series, kind: str) -> set[str]:
    if kind == "cusip":
        vals = s.map(norm_cusip_value)
    else:
        vals = s.map(norm_generic_value)
    return {v for v in vals.dropna().astype(str).tolist() if v}


def parquet_files_for_table(raw_root: Path, table: str) -> list[Path]:
    d = raw_root / table
    if not d.exists():
        return []
    return sorted(d.rglob("*.parquet"))


def read_unique_values_from_file(path: Path, requested_columns: list[str]) -> dict[str, set[str]]:
    out = {c: set() for c in requested_columns}
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in requested_columns if c in available]

    if not selected:
        return out

    # Use ParquetFile.read, not pq.read_table, to avoid directory-partition field merging.
    table = pf.read(columns=selected)
    df = table.to_pandas()

    for col in selected:
        kind = "cusip" if "cusip" in col.lower() else "generic"
        out[col].update(norm_series_values(df[col], kind=kind))

    return out


def merge_set_dicts(dicts: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for d in dicts:
        for k, values in d.items():
            merged.setdefault(k, set()).update(values)
    return merged


def scan_raw_table(raw_root: Path, table: str, columns: list[str], workers: int) -> dict[str, Any]:
    files = parquet_files_for_table(raw_root, table)

    rec: dict[str, Any] = {
        "table": table,
        "files": len(files),
        "ok": bool(files),
        "error": "" if files else "missing_table_directory_or_no_parquet",
        "columns_requested": columns,
        "distinct_counts": {},
        "sets": {},
    }

    if not files:
        return rec

    partials: list[dict[str, set[str]]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(read_unique_values_from_file, p, columns) for p in files]
        for fut in as_completed(futs):
            try:
                partials.append(fut.result())
            except Exception as exc:
                errors.append(repr(exc))

    merged = merge_set_dicts(partials)

    rec["ok"] = len(errors) == 0
    rec["error"] = "; ".join(errors[:5])
    rec["distinct_counts"] = {k: len(v) for k, v in merged.items()}
    rec["sets"] = merged
    return rec


def load_clean_manifest(root: Path, universe: str) -> pd.DataFrame:
    p = root / "data" / "manifests" / "processed" / f"trace_clean_v1_{universe}_nonempty_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing cleaned TRACE manifest: {p}")

    df = pd.read_csv(p)
    df["clean_rows"] = pd.to_numeric(df["clean_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["clean_rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def scan_trace_partition(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["output_path"]))

    rec: dict[str, Any] = {
        "output_path": str(path),
        "clean_rows_manifest": int(row.get("clean_rows", 0) or 0),
        "footer_rows": 0,
        "ok": False,
        "error": "",
        "sets": {c: set() for c in TRACE_ID_COLUMNS},
    }

    try:
        pf = pq.ParquetFile(path)
        rec["footer_rows"] = int(pf.metadata.num_rows)

        available = set(pf.schema_arrow.names)
        selected = [c for c in TRACE_ID_COLUMNS if c in available]

        if selected:
            table = pf.read(columns=selected)
            df = table.to_pandas()

            for col in selected:
                kind = "cusip" if col == "cusip_id" else "generic"
                rec["sets"][col] = norm_series_values(df[col], kind=kind)

        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def scan_trace_universe(root: Path, universe: str, workers: int, limit_partitions: int) -> dict[str, Any]:
    manifest = choose_partitions(load_clean_manifest(root, universe), limit_partitions)
    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"trace_universe={universe} partitions={len(rows)} workers={workers}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(scan_trace_partition, row) for row in rows]

        for i, fut in enumerate(as_completed(futs), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % 100 == 0 or i == len(futs):
                ok = sum(1 for r in results if r.get("ok"))
                footer_rows = sum(int(r.get("footer_rows") or 0) for r in results if r.get("ok"))
                print(f"progress {universe} {i}/{len(futs)} ok={ok} rows={footer_rows:,}", flush=True)

    merged = merge_set_dicts([r["sets"] for r in results if r.get("ok")])

    return {
        "universe": universe,
        "partitions_scanned": len(results),
        "partitions_ok": sum(1 for r in results if r.get("ok")),
        "partitions_failed": sum(1 for r in results if not r.get("ok")),
        "footer_rows_scanned": sum(int(r.get("footer_rows") or 0) for r in results if r.get("ok")),
        "manifest_rows_scanned": int(manifest["clean_rows"].sum()),
        "sets": merged,
        "distinct_counts": {k: len(v) for k, v in merged.items()},
        "failed_examples": [r for r in results if not r.get("ok")][:5],
    }


def coverage(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def make_coverage_rows(
    trace_scans: dict[str, dict[str, Any]],
    raw_scans: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for universe, scan in trace_scans.items():
        trace_sets = scan["sets"]

        trace_cusip = trace_sets.get("cusip_id", set())
        trace_bsym = trace_sets.get("bond_sym_id", set())
        trace_company = trace_sets.get("company_symbol", set())

        targets = [
            ("fisd_issue_issuer", "complete_cusip", trace_cusip, "trace_cusip_id"),
            ("fisd_issue_issuer", "issue_cusip", trace_cusip, "trace_cusip_id"),
            ("bondret_monthly", "cusip", trace_cusip, "trace_cusip_id"),
            ("contrib_bond_returns", "cusip", trace_cusip, "trace_cusip_id"),
            ("crsp_bond_link", "cusip", trace_cusip, "trace_cusip_id"),
            ("bondret_monthly", "bond_sym_id", trace_bsym, "trace_bond_sym_id"),
            ("bondret_monthly", "bsym", trace_bsym, "trace_bond_sym_id"),
            ("bondret_monthly", "company_symbol", trace_company, "trace_company_symbol"),
        ]

        for table, col, left, left_name in targets:
            right = raw_scans.get(table, {}).get("sets", {}).get(col, set())
            matched = len(left & right)

            rows.append(
                {
                    "universe": universe,
                    "left_key": left_name,
                    "right_table": table,
                    "right_key": col,
                    "left_distinct": len(left),
                    "right_distinct": len(right),
                    "matched_distinct": matched,
                    "left_coverage_pct": coverage(matched, len(left)),
                    "right_overlap_pct": coverage(matched, len(right)),
                    "note": "Counts only; no raw identifiers written.",
                }
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 05B identifier coverage audit. Local only. No WRDS. No raw IDs written."
    )
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-partitions", type=int, default=0)
    parser.add_argument("--universes", nargs="+", default=["core_public", "extended_regular"])
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    raw_root = root / "data" / "raw" / "wrds" / "v1"
    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = utc_stamp()

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"raw_root={raw_root}")
    print(f"limit_partitions={args.limit_partitions}")

    raw_scans: dict[str, dict[str, Any]] = {}

    for table, columns in RAW_TABLE_SPECS.items():
        print(f"scanning_raw_table={table}", flush=True)
        raw_scans[table] = scan_raw_table(raw_root, table, columns, workers=args.workers)

    trace_scans: dict[str, dict[str, Any]] = {}

    for universe in args.universes:
        trace_scans[universe] = scan_trace_universe(
            root=root,
            universe=universe,
            workers=args.workers,
            limit_partitions=args.limit_partitions,
        )

    coverage_rows = make_coverage_rows(trace_scans, raw_scans)
    coverage_df = pd.DataFrame(coverage_rows)

    trace_summary_rows = []
    for universe, scan in trace_scans.items():
        row = {
            "universe": universe,
            "partitions_scanned": scan["partitions_scanned"],
            "partitions_ok": scan["partitions_ok"],
            "partitions_failed": scan["partitions_failed"],
            "footer_rows_scanned": scan["footer_rows_scanned"],
            "manifest_rows_scanned": scan["manifest_rows_scanned"],
        }
        for k, v in scan["distinct_counts"].items():
            row[f"distinct_{k}"] = v
        trace_summary_rows.append(row)

    trace_summary_df = pd.DataFrame(trace_summary_rows)

    raw_summary_rows = []
    for table, scan in raw_scans.items():
        row = {
            "table": table,
            "ok": scan.get("ok"),
            "files": scan.get("files"),
            "error": scan.get("error", ""),
        }
        for k, v in scan.get("distinct_counts", {}).items():
            row[f"distinct_{k}"] = v
        raw_summary_rows.append(row)

    raw_summary_df = pd.DataFrame(raw_summary_rows)

    coverage_path = table_dir / "step05b_identifier_coverage.csv"
    trace_summary_path = table_dir / "step05b_trace_identifier_summary.csv"
    raw_summary_path = table_dir / "step05b_raw_identifier_summary.csv"
    summary_path = table_dir / "step05b_identifier_coverage_summary.json"

    coverage_df.to_csv(coverage_path, index=False)
    trace_summary_df.to_csv(trace_summary_path, index=False)
    raw_summary_df.to_csv(raw_summary_path, index=False)

    ok = (
        all(scan["partitions_failed"] == 0 for scan in trace_scans.values())
        and all(bool(scan.get("ok")) for scan in raw_scans.values() if scan.get("files", 0) > 0)
        and len(coverage_df) > 0
    )

    summary = {
        "ok": bool(ok),
        "run_id": run_id,
        "workspace": str(root),
        "limit_partitions": int(args.limit_partitions),
        "universes": args.universes,
        "trace": {
            u: {k: v for k, v in scan.items() if k not in {"sets"}}
            for u, scan in trace_scans.items()
        },
        "raw_tables": {
            t: {k: v for k, v in scan.items() if k != "sets"}
            for t, scan in raw_scans.items()
        },
        "coverage_path": str(coverage_path),
        "trace_summary_path": str(trace_summary_path),
        "raw_summary_path": str(raw_summary_path),
        "note": "Local-only identifier coverage audit. No WRDS. No raw identifier values written to outputs.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step05b_identifier_coverage_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, coverage_path, trace_summary_path, raw_summary_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"COVERAGE={coverage_path}")
    print(f"TRACE_SUMMARY={trace_summary_path}")
    print(f"RAW_SUMMARY={raw_summary_path}")
    print(f"BUNDLE={bundle}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
