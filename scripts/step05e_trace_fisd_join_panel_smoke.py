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


TRACE_COLUMNS = [
    "cusip_id",
    "bond_sym_id",
    "company_symbol",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "trd_rpt_dt",
    "trd_rpt_tm",
    "msg_seq_nb",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_pt",
    "rpt_side_cd",
    "trdg_mkt_cd",
    "dissem_fl",
    "source_stage",
    "source_start_date",
    "source_end_date",
]

FISD_COLUMNS = [
    "complete_cusip",
    "issue_id",
    "issuer_id",
    "issuer_cusip",
    "maturity",
    "coupon",
    "coupon_type",
    "bond_type",
    "security_level",
    "offering_date",
    "offering_amt",
    "principal_amt",
    "foreign_currency",
    "rule_144a",
    "private_placement",
    "asset_backed",
    "convertible",
    "exchangeable",
    "putable",
    "perpetual",
    "redeemable",
    "active_issue",
    "defaulted",
    "in_bankruptcy",
    "mtn",
    "yankee",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def normalize_cusip_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace({"": pd.NA, "<NA>": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )


def load_fisd_master(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / "data" / "processed" / "security_master_v1" / "fisd_issue_master.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing FISD issue master: {path}")

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in FISD_COLUMNS if c in available]

    df = pf.read(columns=selected).to_pandas()

    for col in FISD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["complete_cusip_norm"] = normalize_cusip_series(df["complete_cusip"])
    df["issue_id"] = pd.to_numeric(df["issue_id"], errors="coerce")
    df["issuer_id"] = pd.to_numeric(df["issuer_id"], errors="coerce")
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["offering_date"] = pd.to_datetime(df["offering_date"], errors="coerce")
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    df["principal_amt"] = pd.to_numeric(df["principal_amt"], errors="coerce")

    before = len(df)
    duplicate_cusip_rows = int(df["complete_cusip_norm"].duplicated(keep=False).sum())

    # Deterministic one-row-per-complete-CUSIP mapping.
    # Prefer records with maturity, coupon, and issue_id.
    df["_has_maturity"] = df["maturity"].notna().astype(int)
    df["_has_coupon"] = df["coupon"].notna().astype(int)
    df["_has_issue_id"] = df["issue_id"].notna().astype(int)

    df = df.sort_values(
        ["complete_cusip_norm", "_has_maturity", "_has_coupon", "_has_issue_id", "issue_id"],
        ascending=[True, True, True, True, True],
        na_position="first",
        kind="mergesort",
    )

    one = (
        df.dropna(subset=["complete_cusip_norm"])
        .drop_duplicates("complete_cusip_norm", keep="last")
        .reset_index(drop=True)
    )

    one = one.drop(columns=[c for c in ["_has_maturity", "_has_coupon", "_has_issue_id"] if c in one.columns])

    meta = {
        "fisd_master_path": str(path),
        "fisd_raw_rows": int(before),
        "fisd_rows_after_complete_cusip_dedup": int(len(one)),
        "distinct_complete_cusip_norm": int(one["complete_cusip_norm"].nunique()),
        "duplicate_complete_cusip_rows_before_dedup": duplicate_cusip_rows,
        "missing_maturity_after_dedup": int(one["maturity"].isna().sum()),
        "missing_coupon_after_dedup": int(one["coupon"].isna().sum()),
    }

    return one, meta


def load_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = root / "data" / "manifests" / "processed" / f"trace_clean_v1_{universe}_nonempty_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing TRACE clean manifest for {universe}: {path}")

    df = pd.read_csv(path)
    df["clean_rows"] = pd.to_numeric(df["clean_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["clean_rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def read_trace_partition(path: Path, max_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in TRACE_COLUMNS if c in available]

    if not selected:
        return pd.DataFrame(columns=TRACE_COLUMNS)

    frames = []
    rows_left = max_rows

    for batch in pf.iter_batches(batch_size=min(100_000, rows_left), columns=selected):
        part = batch.to_pandas()
        frames.append(part)
        rows_left -= len(part)
        if rows_left <= 0:
            break

    if not frames:
        return pd.DataFrame(columns=TRACE_COLUMNS)

    df = pd.concat(frames, ignore_index=True).head(max_rows)

    for col in TRACE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df.loc[:, TRACE_COLUMNS].copy()


def output_path(root: Path, run_id: str, universe: str, row: dict[str, Any]) -> Path:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")
    return (
        root
        / "data"
        / "interim"
        / "trace_fisd_join_smoke_v1"
        / f"run_id={run_id}"
        / f"universe={universe}"
        / f"trd_exctn_dt={start}_to_{end}"
        / "part.parquet"
    )


def join_one_partition(
    root_str: str,
    run_id: str,
    universe: str,
    row: dict[str, Any],
    fisd_one: pd.DataFrame,
    max_rows: int,
    overwrite: bool,
) -> dict[str, Any]:
    root = Path(root_str)
    in_path = Path(str(row["output_path"]))
    out_path = output_path(root, run_id, universe, row)

    rec: dict[str, Any] = {
        "universe": universe,
        "input_path": str(in_path),
        "output_path_local_do_not_upload": str(out_path),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "manifest_clean_rows": int(row.get("clean_rows", 0) or 0),
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
        "ok": False,
        "error": "",
    }

    try:
        if out_path.exists() and not overwrite:
            pf = pq.ParquetFile(out_path)
            rec["rows_written"] = int(pf.metadata.num_rows)
            rec["ok"] = True
            return rec

        trace = read_trace_partition(in_path, max_rows=max_rows)
        rec["rows_read"] = int(len(trace))

        if trace.empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trace.to_parquet(out_path, index=False, compression="zstd")
            rec["ok"] = True
            return rec

        trace["cusip_id_norm"] = normalize_cusip_series(trace["cusip_id"])
        trace["trd_exctn_dt"] = pd.to_datetime(trace["trd_exctn_dt"], errors="coerce")
        trace["rptd_pr"] = pd.to_numeric(trace["rptd_pr"], errors="coerce")
        trace["entrd_vol_qt"] = pd.to_numeric(trace["entrd_vol_qt"], errors="coerce")
        trace["yld_pt"] = pd.to_numeric(trace["yld_pt"], errors="coerce")

        joined = trace.merge(
            fisd_one,
            left_on="cusip_id_norm",
            right_on="complete_cusip_norm",
            how="left",
            copy=False,
            suffixes=("", "_fisd"),
        )

        joined["is_fisd_matched"] = joined["issue_id"].notna()
        joined["years_to_maturity"] = (joined["maturity"] - joined["trd_exctn_dt"]).dt.days / 365.25
        joined["maturity_bucket"] = pd.cut(
            joined["years_to_maturity"],
            bins=[-float("inf"), 0, 1, 3, 5, 7, 10, 20, 30, float("inf")],
            labels=["expired", "0_1y", "1_3y", "3_5y", "5_7y", "7_10y", "10_20y", "20_30y", "30y_plus"],
        ).astype("string")

        joined["is_curve_ready"] = (
            joined["is_fisd_matched"]
            & joined["trd_exctn_dt"].notna()
            & joined["maturity"].notna()
            & joined["years_to_maturity"].notna()
            & (joined["years_to_maturity"] > 0)
            & (joined["years_to_maturity"] <= 40)
        )

        matched = joined["is_fisd_matched"]
        rec["matched_rows"] = int(matched.sum())
        rec["unmatched_rows"] = int((~matched).sum())
        rec["match_rate_pct"] = round(100.0 * rec["matched_rows"] / len(joined), 6) if len(joined) else None
        rec["rows_written"] = int(len(joined))
        rec["curve_ready_rows"] = int(joined["is_curve_ready"].sum())
        rec["missing_maturity_rows"] = int((matched & joined["maturity"].isna()).sum())
        rec["missing_coupon_rows"] = int((matched & joined["coupon"].isna()).sum())
        rec["matured_before_trade_rows"] = int((matched & (joined["years_to_maturity"] < -1e-9)).sum())
        rec["negative_years_to_maturity_rows"] = int((matched & (joined["years_to_maturity"] < 0)).sum())

        years = joined.loc[matched, "years_to_maturity"].dropna()
        rec["median_years_to_maturity"] = None if years.empty else round(float(years.median()), 6)
        rec["distinct_issue_id"] = int(joined["issue_id"].dropna().astype(str).nunique())
        rec["distinct_issuer_id"] = int(joined["issuer_id"].dropna().astype(str).nunique())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(f".tmp.parquet")
        if tmp.exists():
            tmp.unlink()
        joined.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out_path)

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

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fisd_one, fisd_meta = load_fisd_master(root)

    all_results = []
    universe_summaries = {}

    for universe in args.universes:
        manifest = choose_partitions(load_manifest(root, universe), args.limit_partitions)
        rows = manifest.to_dict("records")

        print(f"run_id={run_id}")
        print(f"universe={universe}")
        print(f"partitions={len(rows)}")
        print(f"max_rows_per_partition={args.max_rows_per_partition}")
        print(f"workers={args.workers}")

        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [
                pool.submit(
                    join_one_partition,
                    str(root),
                    run_id,
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
        }

    detail = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    detail_path = table_dir / "step05e_trace_fisd_join_panel_smoke_detail.csv"
    summary_path = table_dir / "step05e_trace_fisd_join_panel_smoke_summary.json"

    detail.to_csv(detail_path, index=False)

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
        "local_join_output_root_do_not_upload": str(root / "data" / "interim" / "trace_fisd_join_smoke_v1" / f"run_id={run_id}"),
        "note": "Smoke joined TRACE-to-FISD panel. Local-only. Upload bundle only, not joined parquet.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step05e_trace_fisd_join_panel_smoke_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, detail_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 05E TRACE-to-FISD joined-panel smoke. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universes", nargs="+", default=["core_public", "extended_regular"])
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--max-rows-per-partition", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
