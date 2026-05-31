#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "trc_st",
    "scrty_type_cd",
    "wis_fl",
    "cmsn_trd",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_sign_cd",
    "yld_pt",
    "asof_cd",
    "days_to_sttl_ct",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "rpt_side_cd",
    "buy_cmsn_rt",
    "buy_cpcty_cd",
    "sell_cmsn_rt",
    "sell_cpcty_cd",
    "spcl_trd_fl",
    "trdg_mkt_cd",
    "dissem_fl",
    "orig_msg_seq_nb",
    "stlmnt_dt",
    "trd_mod_3",
    "trd_mod_4",
    "rptg_party_type",
    "ats_indicator",
]

STRING_COLUMNS = [
    "cusip_id",
    "bond_sym_id",
    "company_symbol",
    "trd_exctn_tm",
    "trd_rpt_tm",
    "trc_st",
    "scrty_type_cd",
    "wis_fl",
    "cmsn_trd",
    "yld_sign_cd",
    "asof_cd",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "rpt_side_cd",
    "buy_cpcty_cd",
    "sell_cpcty_cd",
    "spcl_trd_fl",
    "trdg_mkt_cd",
    "dissem_fl",
    "trd_mod_3",
    "trd_mod_4",
    "rptg_party_type",
    "ats_indicator",
]

FLOAT_COLUMNS = [
    "msg_seq_nb",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_pt",
    "days_to_sttl_ct",
    "buy_cmsn_rt",
    "sell_cmsn_rt",
    "orig_msg_seq_nb",
]

DATE_COLUMNS = ["trd_exctn_dt", "trd_rpt_dt", "stlmnt_dt"]

OUTPUT_COLUMNS = TRACE_COLUMNS + [
    "is_public_disseminated",
    "is_extended_regular",
    "is_core_public",
    "source_output_path",
    "source_stage",
    "source_start_date",
    "source_end_date",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def is_missing_code(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip()
    return s.isna() | text.isna() | text.isin(["", "<NA>", "NA", "N/A", "None", "nan", "NaN"])


def code_eq(s: pd.Series, value: str) -> pd.Series:
    return s.astype("string").str.strip().fillna("").eq(value)


def load_final_manifest(root: Path) -> pd.DataFrame:
    path = root / "data" / "manifests" / "extractions" / "step04a_trace_final_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 04A final manifest: {path}")

    df = pd.read_csv(path)
    if "ok" in df.columns:
        df = df.loc[truthy_series(df["ok"])].copy()

    df = df.drop_duplicates("output_path", keep="last").reset_index(drop=True)

    required = {"output_path", "n_rows", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Final manifest missing required columns: {sorted(missing)}")

    return df


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def safe_partition_name(row: dict[str, Any]) -> str:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")
    return f"trd_exctn_dt={start}_to_{end}"


def output_paths(root: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    part = safe_partition_name(row)
    base = root / "data" / "processed" / "trace_clean_v1"
    extended = base / "universe=extended_regular" / part / "part.parquet"
    public = base / "universe=core_public" / part / "part.parquet"
    return extended, public


def read_raw_partition(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in TRACE_COLUMNS if c in available]

    # Use ParquetFile.read, not pq.read_table, to avoid directory-partition field merging.
    table = pf.read(columns=selected)
    df = table.to_pandas()

    for col in TRACE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df.loc[:, TRACE_COLUMNS].copy()


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in STRING_COLUMNS:
        out[col] = out[col].astype("string")

    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    for col in DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def clean_partition_df(df: pd.DataFrame, row: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    x = canonicalize(df)

    trade_date_ok = x["trd_exctn_dt"].notna()
    price = pd.to_numeric(x["rptd_pr"], errors="coerce")
    volume = pd.to_numeric(x["entrd_vol_qt"], errors="coerce")

    basic = trade_date_ok & price.notna() & volume.notna() & (price > 0) & (price <= 500) & (volume > 0)
    executed_t = code_eq(x["trc_st"], "T")
    no_asof = is_missing_code(x["asof_cd"])
    regular_sale_1 = is_missing_code(x["sale_cndtn_cd"]) | code_eq(x["sale_cndtn_cd"], "@")
    regular_sale_2 = is_missing_code(x["sale_cndtn2_cd"])
    not_when_issued = ~code_eq(x["wis_fl"], "Y")
    not_special = ~code_eq(x["spcl_trd_fl"], "Y")
    no_commission = ~code_eq(x["cmsn_trd"], "Y")
    public = code_eq(x["dissem_fl"], "Y")

    extended_mask = (
        basic
        & executed_t
        & no_asof
        & regular_sale_1
        & regular_sale_2
        & not_when_issued
        & not_special
        & no_commission
    )
    public_mask = extended_mask & public

    x["is_public_disseminated"] = public
    x["is_extended_regular"] = extended_mask
    x["is_core_public"] = public_mask
    x["source_output_path"] = str(row.get("output_path", ""))
    x["source_stage"] = str(row.get("source_stage", ""))
    x["source_start_date"] = str(row.get("start_date", ""))
    x["source_end_date"] = str(row.get("end_date", ""))

    extended = x.loc[extended_mask, OUTPUT_COLUMNS].copy()
    core_public = x.loc[public_mask, OUTPUT_COLUMNS].copy()

    dedup_key = [
        "cusip_id",
        "trd_exctn_dt",
        "trd_exctn_tm",
        "msg_seq_nb",
        "rptd_pr",
        "entrd_vol_qt",
    ]

    extended_before = len(extended)
    public_before = len(core_public)

    if extended_before:
        extended = extended.sort_values(
            ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"],
            kind="mergesort",
        )
        extended = extended.drop_duplicates(dedup_key, keep="last").reset_index(drop=True)

    if public_before:
        core_public = core_public.sort_values(
            ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"],
            kind="mergesort",
        )
        core_public = core_public.drop_duplicates(dedup_key, keep="last").reset_index(drop=True)

    stats = {
        "raw_rows": int(len(x)),
        "basic_rows": int(basic.sum()),
        "executed_t_basic_rows": int((basic & executed_t).sum()),
        "extended_rows_pre_dedup": int(extended_before),
        "core_public_rows_pre_dedup": int(public_before),
        "extended_rows": int(len(extended)),
        "core_public_rows": int(len(core_public)),
        "extended_duplicates_removed": int(extended_before - len(extended)),
        "core_public_duplicates_removed": int(public_before - len(core_public)),
        "bad_basic_rows": int(len(x) - basic.sum()),
        "non_t_or_bad_basic_rows": int(len(x) - (basic & executed_t).sum()),
        "dropped_asof_after_t_basic": int((basic & executed_t).sum() - (basic & executed_t & no_asof).sum()),
        "dropped_sale_conditions_after_no_asof": int(
            (basic & executed_t & no_asof).sum()
            - (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2).sum()
        ),
        "dropped_when_issued": int(
            (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2).sum()
            - (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2 & not_when_issued).sum()
        ),
        "dropped_special": int(
            (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2 & not_when_issued).sum()
            - (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2 & not_when_issued & not_special).sum()
        ),
        "dropped_commission": int(
            (basic & executed_t & no_asof & regular_sale_1 & regular_sale_2 & not_when_issued & not_special).sum()
            - extended_before
        ),
        "dropped_non_public": int(extended_before - public_before),
    }

    return extended, core_public, stats


def write_parquet_atomic(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return

    tmp = path.with_suffix(f".tmp.{os.getpid()}.parquet")
    if tmp.exists():
        tmp.unlink()

    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def clean_one(root_str: str, row: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    root = Path(root_str)
    raw_path = Path(str(row["output_path"]))
    extended_path, public_path = output_paths(root, row)

    rec: dict[str, Any] = {
        "output_path": str(raw_path),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "source_stage": str(row.get("source_stage", "")),
        "manifest_rows": int(float(row.get("n_rows", 0) or 0)),
        "extended_output_path": str(extended_path),
        "core_public_output_path": str(public_path),
        "ok": False,
        "skipped_existing": False,
        "error": "",
    }

    try:
        if extended_path.exists() and public_path.exists() and not overwrite:
            rec["raw_rows"] = parquet_rows(raw_path)
            rec["extended_rows"] = parquet_rows(extended_path)
            rec["core_public_rows"] = parquet_rows(public_path)
            rec["skipped_existing"] = True
            rec["ok"] = True
            return rec

        raw = read_raw_partition(raw_path)
        extended, public, stats = clean_partition_df(raw, row)

        write_parquet_atomic(extended, extended_path, overwrite=overwrite)
        write_parquet_atomic(public, public_path, overwrite=overwrite)

        rec.update(stats)
        rec["extended_file_size_bytes"] = int(extended_path.stat().st_size) if extended_path.exists() else 0
        rec["core_public_file_size_bytes"] = int(public_path.stat().st_size) if public_path.exists() else 0
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

    manifest = load_final_manifest(root)
    if args.limit_partitions > 0:
        manifest = manifest.head(args.limit_partitions).copy()

    rows = manifest.to_dict("records")

    artifacts = root / "artifacts" / "tables"
    logs = root / "run_logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"partitions_to_clean={len(rows)}")
    print(f"workers={args.workers}")
    print(f"overwrite={args.overwrite}")

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(clean_one, str(root), row, args.overwrite) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                raw_rows = sum(int(r.get("raw_rows") or 0) for r in results if r.get("ok"))
                extended_rows = sum(int(r.get("extended_rows") or 0) for r in results if r.get("ok"))
                public_rows = sum(int(r.get("core_public_rows") or 0) for r in results if r.get("ok"))
                print(
                    f"progress {i}/{len(futures)} ok={ok} failed={failed} "
                    f"raw={raw_rows:,} extended={extended_rows:,} core_public={public_rows:,}",
                    flush=True,
                )

    df = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"])
    ok_df = df.loc[df["ok"] == True].copy()
    failed_df = df.loc[df["ok"] != True].copy()

    manifest_rows = int_sum(manifest, "n_rows")
    raw_rows = int_sum(ok_df, "raw_rows")
    extended_rows = int_sum(ok_df, "extended_rows")
    public_rows = int_sum(ok_df, "core_public_rows")

    partition_summary_path = artifacts / "step04f_trace_clean_partition_summary.csv"
    summary_path = artifacts / "step04f_trace_clean_summary.json"

    df.to_csv(partition_summary_path, index=False)

    summary = {
        "ok": len(failed_df) == 0 and raw_rows == manifest_rows,
        "run_id": run_id,
        "workspace": str(root),
        "partitions_requested": int(len(manifest)),
        "partitions_ok": int(len(ok_df)),
        "partitions_failed": int(len(failed_df)),
        "manifest_rows": manifest_rows,
        "raw_rows_processed": raw_rows,
        "raw_minus_manifest": int(raw_rows - manifest_rows),
        "extended_regular_rows": extended_rows,
        "core_public_rows": public_rows,
        "extended_regular_retention_pct": None if raw_rows == 0 else round(100.0 * extended_rows / raw_rows, 6),
        "core_public_retention_pct": None if raw_rows == 0 else round(100.0 * public_rows / raw_rows, 6),
        "extended_duplicates_removed": int_sum(ok_df, "extended_duplicates_removed"),
        "core_public_duplicates_removed": int_sum(ok_df, "core_public_duplicates_removed"),
        "skipped_existing": int(ok_df["skipped_existing"].fillna(False).astype(bool).sum()) if not ok_df.empty else 0,
        "failed_examples": failed_df.head(5).to_dict("records"),
        "partition_summary": str(partition_summary_path),
        "extended_output_root": str(root / "data" / "processed" / "trace_clean_v1" / "universe=extended_regular"),
        "core_public_output_root": str(root / "data" / "processed" / "trace_clean_v1" / "universe=core_public"),
        "note": "Cleaned local TRACE only. No WRDS. Upload bundle only, not processed parquet.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bundle = logs / f"step04f_trace_clean_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, partition_summary_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"PARTITION_SUMMARY={partition_summary_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 04F TRACE cleaner. Local only. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--limit-partitions", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
