#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ISSUE_COLUMNS = [
    "issue_id",
    "issuer_id",
    "complete_cusip",
    "issue_cusip",
    "issuer_cusip",
    "isin",
    "cusip_name",
    "issue_name",
    "prospectus_issuer_name",
    "bond_type",
    "coupon_type",
    "maturity",
    "offering_date",
    "offering_amt",
    "offering_price",
    "offering_yield",
    "principal_amt",
    "security_level",
    "industry_code",
    "industry_group",
    "naics_code",
    "country_domicile",
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

COUPON_COLUMNS = [
    "issue_id",
    "coupon",
    "coupon_change_indicator",
    "dated_date",
    "day_count_basis",
    "first_interest_date",
    "last_interest_date",
    "next_interest_date",
    "interest_frequency",
    "pay_in_kind",
    "pay_in_kind_exp_date",
]

RATING_COLUMNS = ["issue_id", "rating", "rating_date", "rating_status", "rating_type", "reason"]
AMOUNT_COLUMNS = ["issue_id", "effective_date", "amount_outstanding", "action_amount", "action_price", "action_type"]

DATE_COLUMNS = [
    "maturity",
    "offering_date",
    "dated_date",
    "first_interest_date",
    "last_interest_date",
    "next_interest_date",
    "pay_in_kind_exp_date",
    "rating_date",
    "effective_date",
]

NUMERIC_COLUMNS = [
    "issue_id",
    "issuer_id",
    "offering_amt",
    "offering_price",
    "offering_yield",
    "principal_amt",
    "coupon",
    "interest_frequency",
    "amount_outstanding",
    "action_amount",
    "action_price",
]

STRING_COLUMNS = [
    "complete_cusip",
    "issue_cusip",
    "issuer_cusip",
    "isin",
    "cusip_name",
    "issue_name",
    "prospectus_issuer_name",
    "bond_type",
    "coupon_type",
    "security_level",
    "industry_code",
    "industry_group",
    "naics_code",
    "country_domicile",
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
    "coupon_change_indicator",
    "day_count_basis",
    "pay_in_kind",
    "rating",
    "rating_status",
    "rating_type",
    "reason",
    "action_type",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def table_files(root: Path, table: str) -> list[Path]:
    table_dir = root / "data" / "raw" / "wrds" / "v1" / table
    if not table_dir.exists():
        raise FileNotFoundError(f"Missing raw table directory: {table_dir}")
    files = sorted(table_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {table_dir}")
    return files


def read_raw_table(root: Path, table: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in table_files(root, table):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        selected = [c for c in columns if c in available]
        if not selected:
            continue

        part = pf.read(columns=selected).to_pandas()
        for col in columns:
            if col not in part.columns:
                part[col] = pd.NA
        frames.append(part.loc[:, columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def normalize_cusip(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace({"": pd.NA, "<NA>": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    for col in ["complete_cusip", "issue_cusip", "issuer_cusip"]:
        if col in out.columns:
            out[col] = normalize_cusip(out[col])

    return out


def dedupe_issue_master(issue: pd.DataFrame) -> pd.DataFrame:
    issue = issue.copy()
    issue = issue.sort_values(["issue_id", "complete_cusip"], na_position="last", kind="mergesort")
    return issue.drop_duplicates("issue_id", keep="last").reset_index(drop=True)


def dedupe_coupon(coupon: pd.DataFrame) -> pd.DataFrame:
    coupon = coupon.copy()
    sort_cols = [c for c in ["issue_id", "dated_date", "next_interest_date"] if c in coupon.columns]
    if sort_cols:
        coupon = coupon.sort_values(sort_cols, na_position="last", kind="mergesort")
    return coupon.drop_duplicates("issue_id", keep="last").reset_index(drop=True)


def clean_history(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.dropna(subset=["issue_id"]).copy()
    if date_col in out.columns:
        out = out.sort_values(["issue_id", date_col], na_position="last", kind="mergesort")
    return out.reset_index(drop=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def count_distinct(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].dropna().astype(str).nunique())


def load_step05b_coverage(root: Path) -> pd.DataFrame:
    path = root / "artifacts" / "tables" / "step05b_identifier_coverage.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 05C build local FISD security-master dimensions. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    out_root = root / "data" / "processed" / "security_master_v1"
    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"

    out_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print("reading FISD raw tables")

    issue_raw = coerce_types(read_raw_table(root, "fisd_issue_issuer", ISSUE_COLUMNS))
    coupon_raw = coerce_types(read_raw_table(root, "fisd_coupon_info", COUPON_COLUMNS))
    rating_hist = coerce_types(read_raw_table(root, "fisd_rating_hist", RATING_COLUMNS))
    amount_hist = coerce_types(read_raw_table(root, "fisd_amount_outstanding", AMOUNT_COLUMNS))

    issue_master = dedupe_issue_master(issue_raw)
    coupon_one = dedupe_coupon(coupon_raw)

    master = issue_master.merge(coupon_one, on="issue_id", how="left", suffixes=("", "_coupon"))
    master = master.sort_values(["complete_cusip", "issue_id"], na_position="last", kind="mergesort").reset_index(drop=True)

    rating_hist = clean_history(rating_hist, "rating_date")
    amount_hist = clean_history(amount_hist, "effective_date")

    master_path = out_root / "fisd_issue_master.parquet"
    rating_path = out_root / "fisd_rating_history.parquet"
    amount_path = out_root / "fisd_amount_outstanding_history.parquet"

    write_parquet(master, master_path)
    write_parquet(rating_hist, rating_path)
    write_parquet(amount_hist, amount_path)

    step05b_cov = load_step05b_coverage(root)
    step05b_fisd_rows = []
    if not step05b_cov.empty:
        mask = (step05b_cov["right_table"].astype(str) == "fisd_issue_issuer") & (
            step05b_cov["right_key"].astype(str) == "complete_cusip"
        )
        step05b_fisd_rows = step05b_cov.loc[mask].to_dict("records")

    summary = {
        "ok": True,
        "run_id": run_id,
        "workspace": str(root),
        "issue_raw_rows": int(len(issue_raw)),
        "issue_master_rows": int(len(issue_master)),
        "security_master_rows": int(len(master)),
        "coupon_raw_rows": int(len(coupon_raw)),
        "coupon_rows_after_issue_id_dedup": int(len(coupon_one)),
        "rating_history_rows": int(len(rating_hist)),
        "amount_outstanding_history_rows": int(len(amount_hist)),
        "distinct_complete_cusip": count_distinct(master, "complete_cusip"),
        "distinct_issue_id": count_distinct(master, "issue_id"),
        "distinct_issuer_id": count_distinct(master, "issuer_id"),
        "distinct_issuer_cusip": count_distinct(master, "issuer_cusip"),
        "missing_complete_cusip_rows": int(master["complete_cusip"].isna().sum()) if "complete_cusip" in master else None,
        "missing_maturity_rows": int(master["maturity"].isna().sum()) if "maturity" in master else None,
        "missing_coupon_rows": int(master["coupon"].isna().sum()) if "coupon" in master else None,
        "duplicate_issue_id_rows_removed": int(len(issue_raw) - len(issue_master)),
        "duplicate_coupon_issue_id_rows_removed": int(len(coupon_raw) - len(coupon_one)),
        "step05b_trace_to_fisd_complete_cusip_coverage_rows": step05b_fisd_rows,
        "outputs_do_not_upload": {
            "fisd_issue_master": str(master_path),
            "fisd_rating_history": str(rating_path),
            "fisd_amount_outstanding_history": str(amount_path),
        },
        "note": "Local-only FISD security-master dimensions. Rating and amount histories remain time-varying and must be joined as-of later; do not use future ratings as static features.",
    }

    if len(master) == 0 or count_distinct(master, "complete_cusip") == 0:
        summary["ok"] = False
    if len(rating_hist) == 0 or len(amount_hist) == 0:
        summary["ok"] = False

    summary_path = table_dir / "step05c_fisd_security_master_summary.json"
    coverage_path = table_dir / "step05c_trace_to_fisd_coverage_from_step05b.csv"
    manifest_path = root / "data" / "manifests" / "processed" / "security_master_v1_manifest.json"

    write_json(summary_path, summary)
    write_json(manifest_path, summary)
    pd.DataFrame(step05b_fisd_rows).to_csv(coverage_path, index=False)

    bundle = log_dir / f"step05c_fisd_security_master_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, coverage_path, manifest_path]:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"COVERAGE={coverage_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
