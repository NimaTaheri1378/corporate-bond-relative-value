#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


READ_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "years_to_maturity",
    "rptd_pr",
    "yld_pt",
    "entrd_vol_qt",
    "is_curve_ready",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_curve_ready_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"trace_fisd_panel_v1_{universe}_curve_ready_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing curve-ready manifest: {path}")

    df = pd.read_csv(path)
    df["curve_ready_rows"] = pd.to_numeric(df["curve_ready_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["curve_ready_rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()
    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def maturity_bucket_counts(s: pd.Series) -> dict[str, int]:
    y = pd.to_numeric(s, errors="coerce")
    buckets = pd.cut(
        y,
        bins=[0, 1, 3, 5, 7, 10, 20, 30, 40],
        labels=["0_1y", "1_3y", "3_5y", "5_7y", "7_10y", "10_20y", "20_30y", "30_40y"],
        include_lowest=False,
    ).astype("string").fillna("missing_or_outside")
    return {str(k): int(v) for k, v in buckets.value_counts(dropna=False).items()}


def issue_count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    if 6 <= n <= 9:
        return "6_9"
    if 10 <= n <= 19:
        return "10_19"
    return "20_plus"


def scan_partition(row: dict[str, Any], max_rows: int) -> dict[str, Any]:
    path = Path(str(row["output_path"]))

    rec: dict[str, Any] = {
        "output_path": str(path),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "expected_curve_ready_rows": int(row.get("curve_ready_rows", 0) or 0),
        "rows_read": 0,
        "curve_ready_rows": 0,
        "distinct_trade_dates": 0,
        "distinct_issuers": 0,
        "distinct_issues": 0,
        "issuer_date_pairs": 0,
        "issuer_date_pairs_ge3_issues": 0,
        "issuer_date_pairs_ge4_issues": 0,
        "issuer_date_pairs_ge5_issues": 0,
        "trade_rows_in_ge3_issue_pairs": 0,
        "trade_rows_in_ge4_issue_pairs": 0,
        "trade_rows_in_ge5_issue_pairs": 0,
        "gross_volume": 0.0,
        "median_years_to_maturity": None,
        "maturity_bucket_counts": {},
        "issuer_date_issue_count_buckets": {},
        "ok": False,
        "error": "",
    }

    try:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        selected = [c for c in READ_COLUMNS if c in available]
        if not selected:
            rec["error"] = "no_selected_columns"
            return rec

        table = pf.read(columns=selected)
        df = table.to_pandas()
        rec["rows_read"] = int(len(df))

        for c in READ_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA

        is_ready = df["is_curve_ready"].astype(bool)
        x = df.loc[is_ready].copy()
        rec["curve_ready_rows"] = int(len(x))

        if x.empty:
            rec["ok"] = True
            return rec

        x["trd_exctn_dt"] = pd.to_datetime(x["trd_exctn_dt"], errors="coerce").dt.date
        x["issuer_id"] = pd.to_numeric(x["issuer_id"], errors="coerce")
        x["issue_id"] = pd.to_numeric(x["issue_id"], errors="coerce")
        x["years_to_maturity"] = pd.to_numeric(x["years_to_maturity"], errors="coerce")
        x["entrd_vol_qt"] = pd.to_numeric(x["entrd_vol_qt"], errors="coerce")

        x = x.dropna(subset=["trd_exctn_dt", "issuer_id", "issue_id", "years_to_maturity"]).copy()

        rec["distinct_trade_dates"] = int(x["trd_exctn_dt"].nunique())
        rec["distinct_issuers"] = int(x["issuer_id"].nunique())
        rec["distinct_issues"] = int(x["issue_id"].nunique())
        rec["gross_volume"] = float(x["entrd_vol_qt"].fillna(0).sum())
        rec["median_years_to_maturity"] = None if x["years_to_maturity"].dropna().empty else round(float(x["years_to_maturity"].median()), 6)
        rec["maturity_bucket_counts"] = maturity_bucket_counts(x["years_to_maturity"])

        issuer_date = (
            x.groupby(["trd_exctn_dt", "issuer_id"], dropna=True)
            .agg(
                n_trades=("issue_id", "size"),
                n_issues=("issue_id", "nunique"),
                gross_volume=("entrd_vol_qt", "sum"),
            )
            .reset_index()
        )

        rec["issuer_date_pairs"] = int(len(issuer_date))
        rec["issuer_date_pairs_ge3_issues"] = int((issuer_date["n_issues"] >= 3).sum())
        rec["issuer_date_pairs_ge4_issues"] = int((issuer_date["n_issues"] >= 4).sum())
        rec["issuer_date_pairs_ge5_issues"] = int((issuer_date["n_issues"] >= 5).sum())

        # Map pair eligibility back to trade-level counts without writing identifiers.
        pair_flags = issuer_date.loc[:, ["trd_exctn_dt", "issuer_id", "n_issues"]]
        xx = x.merge(pair_flags, on=["trd_exctn_dt", "issuer_id"], how="left")
        rec["trade_rows_in_ge3_issue_pairs"] = int((xx["n_issues"] >= 3).sum())
        rec["trade_rows_in_ge4_issue_pairs"] = int((xx["n_issues"] >= 4).sum())
        rec["trade_rows_in_ge5_issue_pairs"] = int((xx["n_issues"] >= 5).sum())

        buckets = Counter(issue_count_bucket(int(v)) for v in issuer_date["n_issues"].fillna(0).astype(int))
        rec["issuer_date_issue_count_buckets"] = dict(sorted(buckets.items()))

        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def merge_nested_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    c = Counter()
    for r in records:
        for k, v in (r.get(key) or {}).items():
            c[str(k)] += int(v)
    return dict(sorted(c.items()))


def int_sum(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def float_sum(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest = choose_partitions(load_curve_ready_manifest(root, args.universe), args.limit_partitions)
    rows = manifest.to_dict("records")

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions={len(rows)}")
    print(f"workers={args.workers}")

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(scan_partition, row, args.max_rows_per_partition) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                ready = sum(int(r.get("curve_ready_rows") or 0) for r in results if r.get("ok"))
                ge3 = sum(int(r.get("issuer_date_pairs_ge3_issues") or 0) for r in results if r.get("ok"))
                print(f"progress {i}/{len(futures)} ok={ok} curve_ready_rows={ready:,} issuer_date_ge3={ge3:,}", flush=True)

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)
    ok_detail = detail.loc[detail["ok"] == True].copy()

    summary = {
        "ok": bool(len(detail) > 0 and detail["ok"].astype(bool).all()),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "limit_partitions": int(args.limit_partitions),
        "partitions_scanned": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()) if not detail.empty else 0,
        "partitions_failed": int((~detail["ok"].astype(bool)).sum()) if not detail.empty else 0,
        "rows_read": int_sum(ok_detail, "rows_read"),
        "curve_ready_rows": int_sum(ok_detail, "curve_ready_rows"),
        "expected_curve_ready_rows_from_manifest": int(pd.to_numeric(manifest["curve_ready_rows"], errors="coerce").fillna(0).sum()),
        "curve_ready_minus_manifest": int_sum(ok_detail, "curve_ready_rows") - int(pd.to_numeric(manifest["curve_ready_rows"], errors="coerce").fillna(0).sum()),
        "distinct_trade_dates_sum_by_partition": int_sum(ok_detail, "distinct_trade_dates"),
        "distinct_issuers_sum_by_partition": int_sum(ok_detail, "distinct_issuers"),
        "distinct_issues_sum_by_partition": int_sum(ok_detail, "distinct_issues"),
        "issuer_date_pairs": int_sum(ok_detail, "issuer_date_pairs"),
        "issuer_date_pairs_ge3_issues": int_sum(ok_detail, "issuer_date_pairs_ge3_issues"),
        "issuer_date_pairs_ge4_issues": int_sum(ok_detail, "issuer_date_pairs_ge4_issues"),
        "issuer_date_pairs_ge5_issues": int_sum(ok_detail, "issuer_date_pairs_ge5_issues"),
        "trade_rows_in_ge3_issue_pairs": int_sum(ok_detail, "trade_rows_in_ge3_issue_pairs"),
        "trade_rows_in_ge4_issue_pairs": int_sum(ok_detail, "trade_rows_in_ge4_issue_pairs"),
        "trade_rows_in_ge5_issue_pairs": int_sum(ok_detail, "trade_rows_in_ge5_issue_pairs"),
        "gross_volume": float_sum(ok_detail, "gross_volume"),
        "maturity_bucket_counts": merge_nested_counts(results, "maturity_bucket_counts"),
        "issuer_date_issue_count_buckets": merge_nested_counts(results, "issuer_date_issue_count_buckets"),
        "failed_examples": detail.loc[detail["ok"] != True].head(5).to_dict("records") if not detail.empty else [],
        "note": "Local-only curve-ready support audit. No WRDS. No raw issuer/bond IDs written in bundle.",
    }

    # Retention-style ratios.
    if summary["curve_ready_rows"] > 0:
        for key in ["trade_rows_in_ge3_issue_pairs", "trade_rows_in_ge4_issue_pairs", "trade_rows_in_ge5_issue_pairs"]:
            summary[f"{key}_pct_of_curve_ready"] = round(100.0 * summary[key] / summary["curve_ready_rows"], 6)

    detail_path = table_dir / f"step05h_curve_ready_{args.universe}_support_detail.csv"
    summary_path = table_dir / f"step05h_curve_ready_{args.universe}_support_summary.json"

    detail.to_csv(detail_path, index=False)
    write_json(summary_path, summary)

    bundle = log_dir / f"step05h_curve_ready_support_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, detail_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 05H curve-ready issuer-date support audit. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--max-rows-per-partition", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--progress-every", type=int, default=5)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
