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
    "rptd_pr",
    "entrd_vol_qt",
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
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["offering_date"] = pd.to_datetime(df["offering_date"], errors="coerce")
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    df["principal_amt"] = pd.to_numeric(df["principal_amt"], errors="coerce")

    before = len(df)
    dup_cusip_rows = int(df["complete_cusip_norm"].duplicated(keep=False).sum())

    # Deterministic one-record-per-complete-CUSIP map for row-level join.
    # Prefer rows with maturity/coupon and non-null issue_id.
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
        "raw_rows": int(before),
        "distinct_complete_cusip_norm": int(one["complete_cusip_norm"].nunique()),
        "duplicate_complete_cusip_rows_before_dedup": dup_cusip_rows,
        "rows_after_complete_cusip_dedup": int(len(one)),
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

    for batch in pf.iter_batches(batch_size=min(50_000, rows_left), columns=selected):
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


def audit_partition(row: dict[str, Any], fisd_one: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    path = Path(str(row["output_path"]))

    rec: dict[str, Any] = {
        "output_path": str(path),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "manifest_clean_rows": int(row.get("clean_rows", 0) or 0),
        "rows_sampled": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "match_rate_pct": None,
        "matched_missing_maturity_rows": 0,
        "matched_missing_coupon_rows": 0,
        "matched_matured_before_trade_rows": 0,
        "matched_negative_years_to_maturity_rows": 0,
        "matched_median_years_to_maturity": None,
        "matched_distinct_issue_id": 0,
        "matched_distinct_issuer_id": 0,
        "ok": False,
        "error": "",
    }

    try:
        trace = read_trace_partition(path, max_rows=max_rows)
        rec["rows_sampled"] = int(len(trace))

        if trace.empty:
            rec["ok"] = True
            return rec

        trace["cusip_id_norm"] = normalize_cusip_series(trace["cusip_id"])
        trace["trd_exctn_dt"] = pd.to_datetime(trace["trd_exctn_dt"], errors="coerce")

        joined = trace.merge(
            fisd_one,
            left_on="cusip_id_norm",
            right_on="complete_cusip_norm",
            how="left",
            copy=False,
            suffixes=("", "_fisd"),
        )

        matched = joined["issue_id"].notna()
        rec["matched_rows"] = int(matched.sum())
        rec["unmatched_rows"] = int((~matched).sum())
        rec["match_rate_pct"] = None if len(joined) == 0 else round(100.0 * rec["matched_rows"] / len(joined), 6)

        matched_df = joined.loc[matched].copy()

        if not matched_df.empty:
            rec["matched_missing_maturity_rows"] = int(matched_df["maturity"].isna().sum())
            rec["matched_missing_coupon_rows"] = int(matched_df["coupon"].isna().sum())

            years = (matched_df["maturity"] - matched_df["trd_exctn_dt"]).dt.days / 365.25
            rec["matched_matured_before_trade_rows"] = int((years < -1e-9).sum())
            rec["matched_negative_years_to_maturity_rows"] = int((years < 0).sum())
            rec["matched_median_years_to_maturity"] = None if years.dropna().empty else round(float(years.dropna().median()), 6)
            rec["matched_distinct_issue_id"] = int(matched_df["issue_id"].dropna().astype(str).nunique())
            rec["matched_distinct_issuer_id"] = int(matched_df["issuer_id"].dropna().astype(str).nunique())

        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


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
        results = []

        print(f"run_id={run_id}")
        print(f"universe={universe}")
        print(f"partitions={len(rows)}")
        print(f"max_rows_per_partition={args.max_rows_per_partition}")

        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [
                pool.submit(audit_partition, row, fisd_one, args.max_rows_per_partition)
                for row in rows
            ]

            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                rec["universe"] = universe
                results.append(rec)

                if i == 1 or i % args.progress_every == 0 or i == len(futures):
                    done = len(results)
                    ok = sum(1 for r in results if r.get("ok"))
                    sampled = sum(int(r.get("rows_sampled") or 0) for r in results)
                    matched = sum(int(r.get("matched_rows") or 0) for r in results)
                    print(
                        f"progress {universe} {done}/{len(futures)} ok={ok} "
                        f"sampled={sampled:,} matched={matched:,}",
                        flush=True,
                    )

        df = pd.DataFrame(results)
        all_results.append(df)

        sampled = int(pd.to_numeric(df["rows_sampled"], errors="coerce").fillna(0).sum())
        matched = int(pd.to_numeric(df["matched_rows"], errors="coerce").fillna(0).sum())

        universe_summaries[universe] = {
            "partitions_scanned": int(len(df)),
            "partitions_ok": int(df["ok"].astype(bool).sum()),
            "partitions_failed": int((~df["ok"].astype(bool)).sum()),
            "rows_sampled": sampled,
            "matched_rows": matched,
            "unmatched_rows": int(pd.to_numeric(df["unmatched_rows"], errors="coerce").fillna(0).sum()),
            "row_match_rate_pct": None if sampled == 0 else round(100.0 * matched / sampled, 6),
            "matched_missing_maturity_rows": int(pd.to_numeric(df["matched_missing_maturity_rows"], errors="coerce").fillna(0).sum()),
            "matched_missing_coupon_rows": int(pd.to_numeric(df["matched_missing_coupon_rows"], errors="coerce").fillna(0).sum()),
            "matched_matured_before_trade_rows": int(pd.to_numeric(df["matched_matured_before_trade_rows"], errors="coerce").fillna(0).sum()),
            "matched_negative_years_to_maturity_rows": int(pd.to_numeric(df["matched_negative_years_to_maturity_rows"], errors="coerce").fillna(0).sum()),
        }

    detail = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    detail_path = table_dir / "step05d_trace_fisd_join_audit_detail.csv"
    summary_path = table_dir / "step05d_trace_fisd_join_audit_summary.json"

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
        "note": "Row-level TRACE-to-FISD audit only. No joined panel written and no raw identifiers written to outputs.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step05d_trace_fisd_join_audit_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, detail_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 05D row-level TRACE-to-FISD join audit. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universes", nargs="+", default=["core_public", "extended_regular"])
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--max-rows-per-partition", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--progress-every", type=int, default=5)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
