#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


READ_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "years_to_maturity",
    "yld_pt",
    "rptd_pr",
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


def output_path(root: Path, universe: str, row: dict[str, Any]) -> Path:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")
    return (
        root
        / "data"
        / "processed"
        / "curve_inputs_v1"
        / f"universe={universe}"
        / f"trd_exctn_dt={start}_to_{end}"
        / "part.parquet"
    )


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def read_partition(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in READ_COLUMNS if c in available]
    if not selected:
        return pd.DataFrame(columns=READ_COLUMNS)

    table = pf.read(columns=selected)
    df = table.to_pandas()

    for c in READ_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df.loc[:, READ_COLUMNS].copy()


def build_issue_date_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "rows_read": int(len(df)),
        "curve_ready_rows": 0,
        "usable_rows": 0,
        "issue_date_rows": 0,
        "distinct_trade_dates": 0,
        "distinct_issuers": 0,
        "distinct_issues": 0,
    }

    if df.empty:
        return pd.DataFrame(), stats

    ready = df["is_curve_ready"].astype(bool)
    x = df.loc[ready].copy()
    stats["curve_ready_rows"] = int(len(x))

    if x.empty:
        return pd.DataFrame(), stats

    x["trd_exctn_dt"] = pd.to_datetime(x["trd_exctn_dt"], errors="coerce").dt.date
    x["issuer_id"] = pd.to_numeric(x["issuer_id"], errors="coerce")
    x["issue_id"] = pd.to_numeric(x["issue_id"], errors="coerce")
    x["years_to_maturity"] = pd.to_numeric(x["years_to_maturity"], errors="coerce")
    x["yld_pt"] = pd.to_numeric(x["yld_pt"], errors="coerce")
    x["rptd_pr"] = pd.to_numeric(x["rptd_pr"], errors="coerce")
    x["entrd_vol_qt"] = pd.to_numeric(x["entrd_vol_qt"], errors="coerce")

    usable = (
        x["trd_exctn_dt"].notna()
        & x["issuer_id"].notna()
        & x["issue_id"].notna()
        & x["years_to_maturity"].between(0.05, 40.0)
        & x["yld_pt"].notna()
        & x["yld_pt"].between(-20.0, 100.0)
        & x["rptd_pr"].notna()
        & x["rptd_pr"].between(1.0, 500.0)
        & x["entrd_vol_qt"].notna()
        & (x["entrd_vol_qt"] > 0)
    )

    x = x.loc[
        usable,
        ["trd_exctn_dt", "issuer_id", "issue_id", "years_to_maturity", "yld_pt", "rptd_pr", "entrd_vol_qt"],
    ].copy()

    stats["usable_rows"] = int(len(x))

    if x.empty:
        return pd.DataFrame(), stats

    # Log-volume weighting is deliberately conservative: it uses trade size information
    # without letting a single block trade dominate an issuer curve.
    x["weight"] = np.log1p(x["entrd_vol_qt"].clip(lower=0))
    x["wx_yld"] = x["yld_pt"] * x["weight"]
    x["wx_mat"] = x["years_to_maturity"] * x["weight"]
    x["wx_prc"] = x["rptd_pr"] * x["weight"]

    grouped = (
        x.groupby(["trd_exctn_dt", "issuer_id", "issue_id"], dropna=True, sort=False)
        .agg(
            trade_count=("issue_id", "size"),
            gross_volume=("entrd_vol_qt", "sum"),
            weight_sum=("weight", "sum"),
            wx_yld=("wx_yld", "sum"),
            wx_mat=("wx_mat", "sum"),
            wx_prc=("wx_prc", "sum"),
            min_years_to_maturity=("years_to_maturity", "min"),
            max_years_to_maturity=("years_to_maturity", "max"),
            min_yld_pt=("yld_pt", "min"),
            max_yld_pt=("yld_pt", "max"),
        )
        .reset_index()
    )

    grouped["yld_pt_wavg"] = grouped["wx_yld"] / grouped["weight_sum"]
    grouped["years_to_maturity_wavg"] = grouped["wx_mat"] / grouped["weight_sum"]
    grouped["rptd_pr_wavg"] = grouped["wx_prc"] / grouped["weight_sum"]

    grouped = grouped.loc[
        grouped["weight_sum"].gt(0)
        & grouped["years_to_maturity_wavg"].between(0.05, 40.0)
        & grouped["yld_pt_wavg"].between(-20.0, 100.0)
    ].copy()

    grouped = grouped.drop(columns=["wx_yld", "wx_mat", "wx_prc"])
    grouped = grouped.sort_values(["trd_exctn_dt", "issuer_id", "issue_id"], kind="mergesort").reset_index(drop=True)

    stats["issue_date_rows"] = int(len(grouped))
    stats["distinct_trade_dates"] = int(grouped["trd_exctn_dt"].nunique()) if not grouped.empty else 0
    stats["distinct_issuers"] = int(grouped["issuer_id"].nunique()) if not grouped.empty else 0
    stats["distinct_issues"] = int(grouped["issue_id"].nunique()) if not grouped.empty else 0

    return grouped, stats


def write_parquet_atomic(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return

    tmp = path.with_suffix(f".tmp.{os.getpid()}.parquet")
    if tmp.exists():
        tmp.unlink()

    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def process_one(root_str: str, universe: str, row: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    root = Path(root_str)
    in_path = Path(str(row["output_path"]))
    out_path = output_path(root, universe, row)

    rec: dict[str, Any] = {
        "universe": universe,
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "input_path": str(in_path),
        "output_path": str(out_path),
        "expected_curve_ready_rows": int(row.get("curve_ready_rows", 0) or 0),
        "rows_read": 0,
        "curve_ready_rows": 0,
        "usable_rows": 0,
        "issue_date_rows": 0,
        "distinct_trade_dates": 0,
        "distinct_issuers": 0,
        "distinct_issues": 0,
        "skipped_existing": False,
        "ok": False,
        "error": "",
    }

    try:
        if out_path.exists() and not overwrite:
            rec["issue_date_rows"] = parquet_rows(out_path)
            rec["skipped_existing"] = True
            rec["ok"] = True
            return rec

        raw = read_partition(in_path)
        agg, stats = build_issue_date_agg(raw)

        write_parquet_atomic(agg, out_path, overwrite=overwrite)

        rec.update(stats)
        rec["output_file_size_bytes"] = int(out_path.stat().st_size) if out_path.exists() else 0
        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def int_sum(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    full_manifest = load_curve_ready_manifest(root, args.universe)
    manifest = choose_partitions(full_manifest, args.limit_partitions)

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions={len(rows)}")
    print(f"workers={args.workers}")
    print(f"overwrite={args.overwrite}")

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(process_one, str(root), args.universe, row, args.overwrite)
            for row in rows
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                curve_ready = sum(int(r.get("curve_ready_rows") or 0) for r in results)
                usable = sum(int(r.get("usable_rows") or 0) for r in results)
                issue_rows = sum(int(r.get("issue_date_rows") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} failed={failed} "
                    f"curve_ready={curve_ready:,} usable={usable:,} issue_date_rows={issue_rows:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)
    ok_detail = detail.loc[detail["ok"] == True].copy()

    detail_path = table_dir / "step06c_curve_inputs_partition_summary.csv"
    summary_path = table_dir / "step06c_curve_inputs_summary.json"

    detail.to_csv(detail_path, index=False)

    output_manifest = pd.DataFrame(
        {
            "universe": ok_detail["universe"] if not ok_detail.empty else pd.Series(dtype=str),
            "start_date": ok_detail["start_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "end_date": ok_detail["end_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "issue_date_rows": pd.to_numeric(ok_detail["issue_date_rows"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "curve_ready_rows": pd.to_numeric(ok_detail["curve_ready_rows"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "usable_rows": pd.to_numeric(ok_detail["usable_rows"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "output_path": ok_detail["output_path"] if not ok_detail.empty else pd.Series(dtype=str),
            "source_panel_path": ok_detail["input_path"] if not ok_detail.empty else pd.Series(dtype=str),
        }
    )

    output_manifest = output_manifest.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)

    if args.limit_partitions > 0:
        manifest_path = manifest_dir / f"curve_inputs_v1_{args.universe}_smoke_manifest.csv"
    else:
        manifest_path = manifest_dir / f"curve_inputs_v1_{args.universe}_manifest.csv"

    nonempty_manifest_path = manifest_path.with_name(manifest_path.stem.replace("_manifest", "_nonempty_manifest") + ".csv")

    output_manifest.to_csv(manifest_path, index=False)
    output_manifest.loc[output_manifest["issue_date_rows"] > 0].to_csv(nonempty_manifest_path, index=False)

    summary = {
        "ok": bool(len(detail) > 0 and detail["ok"].astype(bool).all()),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "limit_partitions": int(args.limit_partitions),
        "partitions_requested": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()) if not detail.empty else 0,
        "partitions_failed": int((~detail["ok"].astype(bool)).sum()) if not detail.empty else 0,
        "expected_curve_ready_rows_from_input_manifest": int(pd.to_numeric(manifest["curve_ready_rows"], errors="coerce").fillna(0).sum()),
        "curve_ready_rows_read": int_sum(ok_detail, "curve_ready_rows"),
        "usable_rows": int_sum(ok_detail, "usable_rows"),
        "issue_date_rows": int_sum(ok_detail, "issue_date_rows"),
        "usable_pct_of_curve_ready": None if int_sum(ok_detail, "curve_ready_rows") == 0 else round(100.0 * int_sum(ok_detail, "usable_rows") / int_sum(ok_detail, "curve_ready_rows"), 6),
        "issue_date_rows_pct_of_usable": None if int_sum(ok_detail, "usable_rows") == 0 else round(100.0 * int_sum(ok_detail, "issue_date_rows") / int_sum(ok_detail, "usable_rows"), 6),
        "distinct_trade_dates_sum_by_partition": int_sum(ok_detail, "distinct_trade_dates"),
        "distinct_issuers_sum_by_partition": int_sum(ok_detail, "distinct_issuers"),
        "distinct_issues_sum_by_partition": int_sum(ok_detail, "distinct_issues"),
        "skipped_existing": int(ok_detail["skipped_existing"].fillna(False).astype(bool).sum()) if "skipped_existing" in ok_detail else 0,
        "detail_path": str(detail_path),
        "manifest": str(manifest_path),
        "nonempty_manifest": str(nonempty_manifest_path),
        "output_root_do_not_upload": str(root / "data" / "processed" / "curve_inputs_v1" / f"universe={args.universe}"),
        "note": "Local-only issue-date aggregation for curve fitting. Upload bundle only, not parquet.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step06c_curve_inputs_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, detail_path, manifest_path, nonempty_manifest_path]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 06C build curve input issue-date aggregates. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
