#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RETURN_TABLE_SPECS = {
    "bondret_monthly": {
        "date_cols": ["date", "t_date"],
        "key_cols": ["issue_id", "cusip", "bond_sym_id", "bsym", "company_symbol", "isin"],
        "return_cols": ["ret_eom", "ret_l5m", "ret_ldm"],
        "price_cols": ["price_eom", "price_l5m", "price_ldm", "yield", "t_yld_pt", "duration", "tmt"],
    },
    "contrib_bond_returns": {
        "date_cols": ["date"],
        "key_cols": ["cusip", "permno"],
        "return_cols": ["ret_eom", "ret_exc", "ret_texc", "ret_eq"],
        "price_cols": ["price_eom", "yield", "yield_spread", "duration", "tmt"],
    },
}

FEATURE_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "residual_yld_pt",
    "issuer_date_residual_z",
    "issuer_date_bucket_residual_z",
    "years_to_maturity_wavg",
    "curve_rmse",
    "n_issues",
    "gross_volume",
    "trade_count",
]


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


def norm_set_from_series(s: pd.Series, kind: str) -> set[str]:
    if kind == "cusip":
        vals = s.map(norm_cusip_value)
    else:
        vals = s.map(norm_generic_value)
    return {v for v in vals.dropna().astype(str).tolist() if v}


def parquet_files_for_table(root: Path, table: str) -> list[Path]:
    d = root / "data" / "raw" / "wrds" / "v1" / table
    if not d.exists():
        return []
    return sorted(d.rglob("*.parquet"))


def read_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in columns if c in available]

    if not selected:
        return pd.DataFrame(columns=columns)

    df = pf.read(columns=selected).to_pandas()

    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA

    return df.loc[:, columns].copy()


def summarize_numeric(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if x.empty:
        return {
            "non_null": 0,
            "finite": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
        }

    return {
        "non_null": int(s.notna().sum()),
        "finite": int(len(x)),
        "mean": round(float(x.mean()), 10),
        "std": round(float(x.std(ddof=1)), 10) if len(x) > 1 else 0.0,
        "min": round(float(x.min()), 10),
        "p01": round(float(x.quantile(0.01)), 10),
        "p50": round(float(x.quantile(0.50)), 10),
        "p99": round(float(x.quantile(0.99)), 10),
        "max": round(float(x.max()), 10),
    }


def scan_return_file(path: Path, table: str, spec: dict[str, list[str]]) -> dict[str, Any]:
    cols = list(dict.fromkeys(spec["date_cols"] + spec["key_cols"] + spec["return_cols"] + spec["price_cols"]))
    rec: dict[str, Any] = {
        "table": table,
        "path": str(path),
        "rows": 0,
        "date_min": None,
        "date_max": None,
        "key_sets": {k: set() for k in spec["key_cols"]},
        "return_values": {k: [] for k in spec["return_cols"]},
        "price_values": {k: [] for k in spec["price_cols"]},
        "ok": False,
        "error": "",
    }

    try:
        df = read_selected(path, cols)
        rec["rows"] = int(len(df))

        # Prefer `date`; fall back to t_date.
        date_col = None
        for c in spec["date_cols"]:
            if c in df.columns and df[c].notna().any():
                date_col = c
                break

        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                rec["date_min"] = str(dates.min().date())
                rec["date_max"] = str(dates.max().date())

        for key in spec["key_cols"]:
            if key not in df.columns:
                continue
            kind = "cusip" if "cusip" in key.lower() else "generic"
            rec["key_sets"][key] = norm_set_from_series(df[key], kind=kind)

        for col in spec["return_cols"]:
            if col in df.columns:
                x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not x.empty:
                    rec["return_values"][col] = [x.astype("float64")]

        for col in spec["price_cols"]:
            if col in df.columns:
                x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not x.empty:
                    rec["price_values"][col] = [x.astype("float64")]

        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def merge_return_scan(records: list[dict[str, Any]], spec: dict[str, list[str]]) -> dict[str, Any]:
    rows = int(sum(int(r.get("rows") or 0) for r in records if r.get("ok")))
    ok_files = int(sum(1 for r in records if r.get("ok")))
    failed_files = int(sum(1 for r in records if not r.get("ok")))

    date_mins = [r["date_min"] for r in records if r.get("date_min")]
    date_maxs = [r["date_max"] for r in records if r.get("date_max")]

    key_sets: dict[str, set[str]] = {k: set() for k in spec["key_cols"]}
    for r in records:
        if not r.get("ok"):
            continue
        for k, vals in r.get("key_sets", {}).items():
            key_sets.setdefault(k, set()).update(vals)

    return_stats: dict[str, Any] = {}
    for col in spec["return_cols"]:
        chunks = []
        for r in records:
            chunks.extend(r.get("return_values", {}).get(col, []))
        series = pd.concat(chunks, ignore_index=True) if chunks else pd.Series(dtype="float64")
        return_stats[col] = summarize_numeric(series)

    price_stats: dict[str, Any] = {}
    for col in spec["price_cols"]:
        chunks = []
        for r in records:
            chunks.extend(r.get("price_values", {}).get(col, []))
        series = pd.concat(chunks, ignore_index=True) if chunks else pd.Series(dtype="float64")
        price_stats[col] = summarize_numeric(series)

    return {
        "rows": rows,
        "ok_files": ok_files,
        "failed_files": failed_files,
        "date_min": min(date_mins) if date_mins else None,
        "date_max": max(date_maxs) if date_maxs else None,
        "distinct_keys": {k: len(v) for k, v in key_sets.items()},
        "key_sets": key_sets,
        "return_stats": return_stats,
        "price_stats": price_stats,
        "failed_examples": [r for r in records if not r.get("ok")][:5],
    }


def scan_return_table(root: Path, table: str, spec: dict[str, list[str]], workers: int) -> dict[str, Any]:
    files = parquet_files_for_table(root, table)

    out: dict[str, Any] = {
        "table": table,
        "files": len(files),
        "ok": False,
        "error": "" if files else "missing_or_empty_table_dir",
    }

    if not files:
        return out

    records: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(scan_return_file, p, table, spec) for p in files]

        for fut in as_completed(futs):
            records.append(fut.result())

    merged = merge_return_scan(records, spec)
    out.update({k: v for k, v in merged.items() if k != "key_sets"})
    out["ok"] = merged["failed_files"] == 0
    out["key_sets"] = merged["key_sets"]
    return out


def load_fisd_issue_map(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "security_master_v1" / "fisd_issue_master.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["issue_id", "complete_cusip"])

    pf = pq.ParquetFile(path)
    selected = [c for c in ["issue_id", "complete_cusip"] if c in set(pf.schema_arrow.names)]
    if not selected:
        return pd.DataFrame(columns=["issue_id", "complete_cusip"])

    df = pf.read(columns=selected).to_pandas()

    if "issue_id" not in df.columns:
        df["issue_id"] = pd.NA
    if "complete_cusip" not in df.columns:
        df["complete_cusip"] = pd.NA

    df["issue_id_norm"] = df["issue_id"].map(norm_generic_value)
    df["complete_cusip_norm"] = df["complete_cusip"].map(norm_cusip_value)

    return df.loc[:, ["issue_id_norm", "complete_cusip_norm"]].dropna().drop_duplicates()


def load_feature_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_residual_features_v1_{universe}_validated_nonempty_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing residual-feature manifest: {path}")

    df = pd.read_csv(path)
    df["rows"] = pd.to_numeric(df["rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()
    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def scan_feature_partition(path: Path) -> dict[str, Any]:
    rec = {
        "path": str(path),
        "rows": 0,
        "date_min": None,
        "date_max": None,
        "issue_id_set": set(),
        "issuer_id_set": set(),
        "ok": False,
        "error": "",
    }

    try:
        df = read_selected(path, FEATURE_COLUMNS)
        rec["rows"] = int(len(df))

        if "trd_exctn_dt" in df.columns:
            dates = pd.to_datetime(df["trd_exctn_dt"], errors="coerce").dropna()
            if not dates.empty:
                rec["date_min"] = str(dates.min().date())
                rec["date_max"] = str(dates.max().date())

        if "issue_id" in df.columns:
            rec["issue_id_set"] = norm_set_from_series(df["issue_id"], kind="generic")
        if "issuer_id" in df.columns:
            rec["issuer_id_set"] = norm_set_from_series(df["issuer_id"], kind="generic")

        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def scan_feature_sample(root: Path, universe: str, limit_partitions: int, workers: int) -> dict[str, Any]:
    manifest_all = load_feature_manifest(root, universe)
    manifest = choose_partitions(manifest_all, limit_partitions)

    records: list[dict[str, Any]] = []

    print(f"feature_universe={universe} partitions={len(manifest)}")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(scan_feature_partition, Path(str(p))) for p in manifest["output_path"]]

        for i, fut in enumerate(as_completed(futs), start=1):
            rec = fut.result()
            records.append(rec)

            if i == 1 or i % 25 == 0 or i == len(futs):
                ok = sum(1 for r in records if r.get("ok"))
                rows = sum(int(r.get("rows") or 0) for r in records if r.get("ok"))
                print(f"progress_feature {i}/{len(futs)} ok={ok} rows={rows:,}", flush=True)

    ok_records = [r for r in records if r.get("ok")]
    issue_ids: set[str] = set()
    issuer_ids: set[str] = set()

    for r in ok_records:
        issue_ids.update(r["issue_id_set"])
        issuer_ids.update(r["issuer_id_set"])

    date_mins = [r["date_min"] for r in ok_records if r.get("date_min")]
    date_maxs = [r["date_max"] for r in ok_records if r.get("date_max")]

    return {
        "universe": universe,
        "manifest_partitions_total": int(len(manifest_all)),
        "sample_partitions": int(len(manifest)),
        "sample_ok_partitions": int(len(ok_records)),
        "sample_failed_partitions": int(len(records) - len(ok_records)),
        "sample_rows": int(sum(int(r.get("rows") or 0) for r in ok_records)),
        "sample_date_min": min(date_mins) if date_mins else None,
        "sample_date_max": max(date_maxs) if date_maxs else None,
        "sample_distinct_issue_id": int(len(issue_ids)),
        "sample_distinct_issuer_id": int(len(issuer_ids)),
        "issue_id_set": issue_ids,
        "issuer_id_set": issuer_ids,
        "failed_examples": [r for r in records if not r.get("ok")][:5],
    }


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def make_coverage_rows(feature_scan: dict[str, Any], return_scans: dict[str, dict[str, Any]], issue_map: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    feature_issue_ids = feature_scan["issue_id_set"]

    issue_to_cusip = dict(zip(issue_map["issue_id_norm"], issue_map["complete_cusip_norm"], strict=False))
    feature_cusips = {issue_to_cusip[i] for i in feature_issue_ids if i in issue_to_cusip and issue_to_cusip[i]}

    targets = [
        ("bondret_monthly", "issue_id", feature_issue_ids, "feature_issue_id"),
        ("bondret_monthly", "cusip", feature_cusips, "feature_issue_id_mapped_complete_cusip"),
        ("contrib_bond_returns", "cusip", feature_cusips, "feature_issue_id_mapped_complete_cusip"),
    ]

    for table, key, left, left_name in targets:
        right = return_scans.get(table, {}).get("key_sets", {}).get(key, set())
        matched = len(left & right)

        rows.append(
            {
                "feature_universe": feature_scan["universe"],
                "left_key": left_name,
                "right_table": table,
                "right_key": key,
                "left_distinct": int(len(left)),
                "right_distinct": int(len(right)),
                "matched_distinct": int(matched),
                "left_coverage_pct": pct(matched, len(left)),
                "right_overlap_pct": pct(matched, len(right)),
                "note": "Counts only; no raw identifiers written.",
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 07A label-source audit. Local only. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--feature-sample-partitions", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"feature_sample_partitions={args.feature_sample_partitions}")

    return_scans: dict[str, dict[str, Any]] = {}

    for table, spec in RETURN_TABLE_SPECS.items():
        print(f"scanning_return_table={table}", flush=True)
        return_scans[table] = scan_return_table(root, table, spec, workers=args.workers)

    feature_scan = scan_feature_sample(
        root=root,
        universe=args.universe,
        limit_partitions=args.feature_sample_partitions,
        workers=args.workers,
    )

    issue_map = load_fisd_issue_map(root)
    coverage_rows = make_coverage_rows(feature_scan, return_scans, issue_map)

    return_summary_rows = []
    for table, scan in return_scans.items():
        row = {
            "table": table,
            "ok": bool(scan.get("ok")),
            "files": int(scan.get("files", 0)),
            "rows": int(scan.get("rows", 0)),
            "ok_files": int(scan.get("ok_files", 0)),
            "failed_files": int(scan.get("failed_files", 0)),
            "date_min": scan.get("date_min"),
            "date_max": scan.get("date_max"),
            "distinct_keys": json.dumps(scan.get("distinct_keys", {}), sort_keys=True),
            "return_stats": json.dumps(scan.get("return_stats", {}), sort_keys=True),
            "price_stats": json.dumps(scan.get("price_stats", {}), sort_keys=True),
            "error": scan.get("error", ""),
        }
        return_summary_rows.append(row)

    return_summary = pd.DataFrame(return_summary_rows)
    coverage = pd.DataFrame(coverage_rows)

    feature_summary_public = {
        k: v
        for k, v in feature_scan.items()
        if k not in {"issue_id_set", "issuer_id_set"}
    }

    # Decide what we can safely build next.
    bondret_issue_cov = None
    if not coverage.empty:
        m = (
            (coverage["right_table"] == "bondret_monthly")
            & (coverage["right_key"] == "issue_id")
        )
        if m.any():
            bondret_issue_cov = float(coverage.loc[m, "left_coverage_pct"].iloc[0])

    contrib_cusip_cov = None
    if not coverage.empty:
        m = (
            (coverage["right_table"] == "contrib_bond_returns")
            & (coverage["right_key"] == "cusip")
        )
        if m.any():
            contrib_cusip_cov = float(coverage.loc[m, "left_coverage_pct"].iloc[0])

    recommendation = {
        "primary_near_term_label_source": "bondret_monthly via issue_id" if (bondret_issue_cov or 0) >= 50 else "needs_review",
        "secondary_label_source": "contrib_bond_returns via FISD complete_cusip mapping" if (contrib_cusip_cov or 0) >= 30 else "needs_review",
        "daily_5_20_60_plan": "Audit TRACE-derived forward price/yield labels after monthly-return source coverage is validated.",
        "leakage_note": "Do not train models until label dates, signal dates, and horizon embargo rules are explicitly encoded.",
    }

    summary = {
        "ok": bool(all(scan.get("ok") for scan in return_scans.values()) and feature_scan["sample_failed_partitions"] == 0 and len(coverage_rows) > 0),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "feature_scan": feature_summary_public,
        "return_tables": {
            table: {k: v for k, v in scan.items() if k not in {"key_sets"}}
            for table, scan in return_scans.items()
        },
        "issue_map_rows": int(len(issue_map)),
        "recommendation": recommendation,
        "note": "Local-only label-source audit. No WRDS. No raw identifiers written.",
    }

    summary_path = table_dir / "step07a_label_source_audit_summary.json"
    return_summary_path = table_dir / "step07a_return_table_summary.csv"
    coverage_path = table_dir / "step07a_label_source_identifier_coverage.csv"

    write_json(summary_path, summary)
    return_summary.to_csv(return_summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    bundle = log_dir / f"step07a_label_source_audit_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, return_summary_path, coverage_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"RETURN_TABLE_SUMMARY={return_summary_path}")
    print(f"COVERAGE={coverage_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
