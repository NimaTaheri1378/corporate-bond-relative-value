#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
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


def load_step05e_module(root: Path):
    path = root / "scripts" / "step05e_trace_fisd_join_panel_smoke.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 05E script: {path}")

    spec = importlib.util.spec_from_file_location("step05e_trace_fisd_join_panel_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stable_output_path(root: Path, _run_id: str, universe: str, row: dict[str, Any]) -> Path:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")
    return (
        root
        / "data"
        / "processed"
        / "trace_fisd_panel_v1"
        / f"universe={universe}"
        / f"trd_exctn_dt={start}_to_{end}"
        / "part.parquet"
    )


def summarize_existing_join_output(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "rows_read": 0,
        "rows_written": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "match_rate_pct": None,
        "curve_ready_rows": 0,
        "missing_maturity_rows": 0,
        "missing_coupon_rows": 0,
        "matured_before_trade_rows": 0,
        "negative_years_to_maturity_rows": 0,
        "median_years_to_maturity": None,
        "distinct_issue_id": 0,
        "distinct_issuer_id": 0,
    }

    if not path.exists():
        return rec

    pf = pq.ParquetFile(path)
    rec["rows_written"] = int(pf.metadata.num_rows)
    rec["rows_read"] = int(pf.metadata.num_rows)

    if pf.metadata.num_rows == 0:
        return rec

    cols = [
        "is_fisd_matched",
        "is_curve_ready",
        "maturity",
        "coupon",
        "years_to_maturity",
        "issue_id",
        "issuer_id",
    ]
    available = set(pf.schema_arrow.names)
    selected = [c for c in cols if c in available]
    if not selected:
        return rec

    df = pf.read(columns=selected).to_pandas()

    if "is_fisd_matched" in df:
        matched = df["is_fisd_matched"].astype(bool)
    else:
        matched = df["issue_id"].notna() if "issue_id" in df else pd.Series(False, index=df.index)

    rec["matched_rows"] = int(matched.sum())
    rec["unmatched_rows"] = int((~matched).sum())
    rec["match_rate_pct"] = round(100.0 * rec["matched_rows"] / len(df), 6) if len(df) else None

    if "is_curve_ready" in df:
        rec["curve_ready_rows"] = int(df["is_curve_ready"].astype(bool).sum())

    if "maturity" in df:
        rec["missing_maturity_rows"] = int((matched & df["maturity"].isna()).sum())

    if "coupon" in df:
        rec["missing_coupon_rows"] = int((matched & df["coupon"].isna()).sum())

    if "years_to_maturity" in df:
        y = pd.to_numeric(df["years_to_maturity"], errors="coerce")
        rec["matured_before_trade_rows"] = int((matched & (y < -1e-9)).sum())
        rec["negative_years_to_maturity_rows"] = int((matched & (y < 0)).sum())
        yy = y.loc[matched].dropna()
        rec["median_years_to_maturity"] = None if yy.empty else round(float(yy.median()), 6)

    if "issue_id" in df:
        rec["distinct_issue_id"] = int(df["issue_id"].dropna().astype(str).nunique())

    if "issuer_id" in df:
        rec["distinct_issuer_id"] = int(df["issuer_id"].dropna().astype(str).nunique())

    return rec


def join_one_with_resume(
    mod,
    root: Path,
    universe: str,
    row: dict[str, Any],
    fisd_one: pd.DataFrame,
    max_rows: int,
    overwrite: bool,
) -> dict[str, Any]:
    out_path = stable_output_path(root, "stable", universe, row)

    rec = mod.join_one_partition(
        str(root),
        "stable",
        universe,
        row,
        fisd_one,
        max_rows,
        overwrite,
    )

    # Step 05E's skip path only returns rows_written. Fill metrics from the existing footer/data.
    if rec.get("ok") and out_path.exists() and int(rec.get("rows_read") or 0) == 0:
        rec.update(summarize_existing_join_output(out_path))
        rec["skipped_existing"] = True
    else:
        rec["skipped_existing"] = False

    # Make output path stable in the result record.
    rec["output_path_local_do_not_upload"] = str(out_path)
    return rec


def int_sum(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    mod = load_step05e_module(root)

    # Monkeypatch Step 05E output path so reused join code writes to stable processed panel.
    mod.output_path = stable_output_path

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fisd_one, fisd_meta = mod.load_fisd_master(root)

    all_results = []
    universe_summaries = {}

    for universe in args.universes:
        manifest = mod.choose_partitions(mod.load_manifest(root, universe), args.limit_partitions)
        rows = manifest.to_dict("records")

        print(f"run_id={run_id}")
        print(f"universe={universe}")
        print(f"partitions={len(rows)}")
        print(f"max_rows_per_partition={args.max_rows_per_partition}")
        print(f"workers={args.workers}")
        print(f"overwrite={args.overwrite}")

        results = []

        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [
                pool.submit(
                    join_one_with_resume,
                    mod,
                    root,
                    universe,
                    row,
                    fisd_one,
                    args.max_rows_per_partition,
                    args.overwrite,
                )
                for row in rows
            ]

            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                results.append(rec)

                if i == 1 or i % args.progress_every == 0 or i == len(futures):
                    ok = sum(1 for r in results if r.get("ok"))
                    failed = len(results) - ok
                    rows_read = sum(int(r.get("rows_read") or 0) for r in results)
                    matched = sum(int(r.get("matched_rows") or 0) for r in results)
                    curve_ready = sum(int(r.get("curve_ready_rows") or 0) for r in results)
                    print(
                        f"progress {universe} {i}/{len(futures)} ok={ok} failed={failed} "
                        f"rows_read={rows_read:,} matched={matched:,} curve_ready={curve_ready:,}",
                        flush=True,
                    )

        df = pd.DataFrame(results)
        all_results.append(df)

        rows_read = int_sum(df, "rows_read")
        matched = int_sum(df, "matched_rows")
        curve_ready = int_sum(df, "curve_ready_rows")

        universe_summaries[universe] = {
            "partitions_requested": int(len(df)),
            "partitions_ok": int(df["ok"].astype(bool).sum()) if not df.empty else 0,
            "partitions_failed": int((~df["ok"].astype(bool)).sum()) if not df.empty else 0,
            "rows_read": rows_read,
            "rows_written": int_sum(df, "rows_written"),
            "matched_rows": matched,
            "unmatched_rows": int_sum(df, "unmatched_rows"),
            "match_rate_pct": None if rows_read == 0 else round(100.0 * matched / rows_read, 6),
            "curve_ready_rows": curve_ready,
            "curve_ready_pct_of_rows_read": None if rows_read == 0 else round(100.0 * curve_ready / rows_read, 6),
            "missing_maturity_rows": int_sum(df, "missing_maturity_rows"),
            "missing_coupon_rows": int_sum(df, "missing_coupon_rows"),
            "matured_before_trade_rows": int_sum(df, "matured_before_trade_rows"),
            "negative_years_to_maturity_rows": int_sum(df, "negative_years_to_maturity_rows"),
            "skipped_existing": int(df["skipped_existing"].fillna(False).astype(bool).sum()) if "skipped_existing" in df else 0,
        }

    detail = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    detail_path = table_dir / "step05f_trace_fisd_panel_detail.csv"
    summary_path = table_dir / "step05f_trace_fisd_panel_summary.json"

    detail.to_csv(detail_path, index=False)

    manifest_paths = {}
    for universe in args.universes:
        u = detail.loc[(detail["universe"] == universe) & (detail["ok"] == True)].copy()
        if u.empty:
            continue

        out = pd.DataFrame(
            {
                "universe": universe,
                "start_date": u["start_date"],
                "end_date": u["end_date"],
                "rows": pd.to_numeric(u["rows_written"], errors="coerce").fillna(0).astype("int64"),
                "matched_rows": pd.to_numeric(u["matched_rows"], errors="coerce").fillna(0).astype("int64"),
                "curve_ready_rows": pd.to_numeric(u["curve_ready_rows"], errors="coerce").fillna(0).astype("int64"),
                "output_path": u["output_path_local_do_not_upload"],
                "source_clean_path": u["input_path"],
            }
        ).sort_values(["start_date", "end_date", "output_path"])

        full_manifest = manifest_dir / f"trace_fisd_panel_v1_{universe}_manifest.csv"
        nonempty_manifest = manifest_dir / f"trace_fisd_panel_v1_{universe}_nonempty_manifest.csv"

        out.to_csv(full_manifest, index=False)
        out.loc[out["rows"] > 0].to_csv(nonempty_manifest, index=False)

        manifest_paths[universe] = {
            "manifest": str(full_manifest),
            "nonempty_manifest": str(nonempty_manifest),
        }

    ok = bool(len(detail) > 0 and detail["ok"].astype(bool).all())

    summary = {
        "ok": ok,
        "run_id": run_id,
        "workspace": str(root),
        "universes": args.universes,
        "limit_partitions": int(args.limit_partitions),
        "max_rows_per_partition": int(args.max_rows_per_partition),
        "fisd_master": fisd_meta,
        "universe_summaries": universe_summaries,
        "detail_path": str(detail_path),
        "manifests": manifest_paths,
        "output_root_do_not_upload": str(root / "data" / "processed" / "trace_fisd_panel_v1"),
        "note": "Stable TRACE-to-FISD processed panel. Local-only. Upload bundle only, not parquet.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bundle = log_dir / f"step05f_trace_fisd_panel_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, detail_path]:
            tar.add(path, arcname=str(path.relative_to(root)))
        for item in manifest_paths.values():
            for p in item.values():
                pp = Path(p)
                if pp.exists():
                    tar.add(pp, arcname=str(pp.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 05F stable TRACE-to-FISD processed panel. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universes", nargs="+", default=["core_public"])
    p.add_argument("--limit-partitions", type=int, default=0)
    p.add_argument("--max-rows-per-partition", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
