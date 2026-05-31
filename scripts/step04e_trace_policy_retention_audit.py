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

NEEDED_COLUMNS = [
    "trd_exctn_dt",
    "trc_st",
    "asof_cd",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "wis_fl",
    "cmsn_trd",
    "spcl_trd_fl",
    "dissem_fl",
    "rptd_pr",
    "entrd_vol_qt",
]

COUNT_KEYS = [
    "raw_rows",
    "basic_price_volume_date_rows",
    "executed_T_rows",
    "executed_T_basic_rows",
    "core_step_no_asof_rows",
    "core_step_regular_sale_condition_rows",
    "core_step_not_when_issued_rows",
    "core_step_not_special_trade_rows",
    "core_step_no_commission_rows",
    "core_public_regular_rows",
    "regular_all_dissemination_rows",
    "executed_public_rows",
    "executed_all_rows",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "data" / "manifests" / "extractions" / "step04a_trace_final_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final manifest: {path}")
    df = pd.read_csv(path)
    if "ok" in df.columns:
        df = df.loc[truthy_series(df["ok"])].copy()
    if "output_path" not in df.columns:
        raise ValueError("Final manifest lacks output_path")
    df = df.drop_duplicates("output_path", keep="last").reset_index(drop=True)
    return df


def is_missing_code(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip()
    return s.isna() | text.isna() | text.isin(["", "<NA>", "NA", "N/A", "None", "nan", "NaN"])


def code_eq(s: pd.Series, value: str) -> pd.Series:
    return s.astype("string").str.strip().fillna("").eq(value)


def scan_partition(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["output_path"]))
    rec: dict[str, Any] = {
        "output_path": str(path),
        "source_stage": str(row.get("source_stage", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "manifest_rows": int(float(row.get("n_rows", 0) or 0)),
        "ok": False,
        "error": "",
    }
    for key in COUNT_KEYS:
        rec[key] = 0

    if not path.exists():
        rec["error"] = "missing_file"
        return rec

    try:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        selected = [c for c in NEEDED_COLUMNS if c in available]
        tab = pf.read(columns=selected)
        df = tab.to_pandas()

        n = int(pf.metadata.num_rows)
        rec["footer_rows"] = n
        rec["raw_rows"] = n

        for c in NEEDED_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA

        trade_date_ok = pd.to_datetime(df["trd_exctn_dt"], errors="coerce").notna()
        price = pd.to_numeric(df["rptd_pr"], errors="coerce")
        volume = pd.to_numeric(df["entrd_vol_qt"], errors="coerce")

        basic = trade_date_ok & price.notna() & volume.notna() & (price > 0) & (price <= 500) & (volume > 0)
        executed_t = code_eq(df["trc_st"], "T")
        no_asof = is_missing_code(df["asof_cd"])
        regular_sale_1 = is_missing_code(df["sale_cndtn_cd"]) | code_eq(df["sale_cndtn_cd"], "@")
        regular_sale_2 = is_missing_code(df["sale_cndtn2_cd"])
        not_when_issued = ~code_eq(df["wis_fl"], "Y")
        not_special = ~code_eq(df["spcl_trd_fl"], "Y")
        no_commission = ~code_eq(df["cmsn_trd"], "Y")
        public = code_eq(df["dissem_fl"], "Y")

        rec["basic_price_volume_date_rows"] = int(basic.sum())
        rec["executed_T_rows"] = int(executed_t.sum())
        rec["executed_T_basic_rows"] = int((basic & executed_t).sum())

        core = basic & executed_t
        rec["core_step_no_asof_rows"] = int((core & no_asof).sum())
        core = core & no_asof
        rec["core_step_regular_sale_condition_rows"] = int((core & regular_sale_1 & regular_sale_2).sum())
        core = core & regular_sale_1 & regular_sale_2
        rec["core_step_not_when_issued_rows"] = int((core & not_when_issued).sum())
        core = core & not_when_issued
        rec["core_step_not_special_trade_rows"] = int((core & not_special).sum())
        core = core & not_special
        rec["core_step_no_commission_rows"] = int((core & no_commission).sum())
        core = core & no_commission
        rec["core_public_regular_rows"] = int((core & public).sum())

        rec["regular_all_dissemination_rows"] = int(core.sum())
        rec["executed_public_rows"] = int((basic & executed_t & public).sum())
        rec["executed_all_rows"] = int((basic & executed_t).sum())

        rec["dropped_non_T_or_bad_basic"] = int(n - rec["executed_T_basic_rows"])
        rec["dropped_by_core_regular_before_public"] = int(rec["executed_T_basic_rows"] - rec["regular_all_dissemination_rows"])
        rec["dropped_by_public_filter"] = int(rec["regular_all_dissemination_rows"] - rec["core_public_regular_rows"])
        rec["ok"] = True
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def pct(num: int, den: int) -> float | None:
    return None if den == 0 else round(100.0 * num / den, 6)


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest = load_manifest(root)
    if args.limit_partitions > 0:
        manifest = manifest.head(args.limit_partitions).copy()

    out_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"partitions_to_scan={len(rows)}")
    print(f"workers={args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(scan_partition, row) for row in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)
            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                core_public = sum(int(r.get("core_public_regular_rows") or 0) for r in results if r.get("ok"))
                raw = sum(int(r.get("raw_rows") or 0) for r in results if r.get("ok"))
                print(f"progress {i}/{len(futures)} ok={ok} failed={failed} raw={raw:,} core_public={core_public:,}", flush=True)

    df = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"])
    ok_df = df.loc[df["ok"] == True].copy()
    failed_df = df.loc[df["ok"] != True].copy()

    totals = {
        k: int(pd.to_numeric(ok_df.get(k, 0), errors="coerce").fillna(0).sum())
        for k in COUNT_KEYS
    }

    expected_rows = int(pd.to_numeric(manifest["n_rows"], errors="coerce").fillna(0).sum())

    summary = {
        "ok": int(len(failed_df)) == 0 and totals["raw_rows"] == expected_rows,
        "run_id": run_id,
        "workspace": str(root),
        "partitions_scanned": int(len(df)),
        "partitions_ok": int(len(ok_df)),
        "partitions_failed": int(len(failed_df)),
        "expected_rows_from_manifest": expected_rows,
        "raw_rows_scanned": totals["raw_rows"],
        "raw_minus_expected": int(totals["raw_rows"] - expected_rows),
        "counts": totals,
        "retention_pct_of_raw": {k: pct(v, totals["raw_rows"]) for k, v in totals.items()},
        "drop_counts": {
            "drop_bad_basic_or_non_executed_T": int(totals["raw_rows"] - totals["executed_T_basic_rows"]),
            "drop_asof_after_executed_basic": int(totals["executed_T_basic_rows"] - totals["core_step_no_asof_rows"]),
            "drop_sale_conditions_after_no_asof": int(totals["core_step_no_asof_rows"] - totals["core_step_regular_sale_condition_rows"]),
            "drop_when_issued": int(totals["core_step_regular_sale_condition_rows"] - totals["core_step_not_when_issued_rows"]),
            "drop_special_trade": int(totals["core_step_not_when_issued_rows"] - totals["core_step_not_special_trade_rows"]),
            "drop_commission": int(totals["core_step_not_special_trade_rows"] - totals["core_step_no_commission_rows"]),
            "drop_non_public_disseminated": int(totals["core_step_no_commission_rows"] - totals["core_public_regular_rows"]),
        },
        "policy_recommendation": {
            "core_public_regular": "basic price/volume/date; trc_st == T; no asof; sale_cndtn_cd missing/@; sale_cndtn2_cd missing; wis_fl != Y; spcl_trd_fl != Y; cmsn_trd != Y; dissem_fl == Y",
            "extended_regular_all_dissemination": "same as core but do not require dissem_fl == Y",
            "executed_public": "basic price/volume/date; trc_st == T; dissem_fl == Y",
        },
        "failed_examples": failed_df.head(5).to_dict("records"),
    }

    partition_path = out_dir / "step04e_trace_policy_retention_partition_summary.csv"
    summary_path = out_dir / "step04e_trace_policy_retention_summary.json"

    df.to_csv(partition_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bundle = log_dir / f"step04e_trace_policy_retention_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, partition_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"PARTITION_SUMMARY={partition_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 04E local TRACE cleaning-policy retention audit. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--limit-partitions", type=int, default=0)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
