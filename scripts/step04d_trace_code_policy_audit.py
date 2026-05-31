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


CODE_COLUMNS = [
    "trc_st",
    "asof_cd",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "rpt_side_cd",
    "wis_fl",
    "cmsn_trd",
    "spcl_trd_fl",
    "dissem_fl",
    "scrty_type_cd",
    "trdg_mkt_cd",
    "buy_cpcty_cd",
    "sell_cpcty_cd",
    "rptg_party_type",
    "ats_indicator",
]

NUMERIC_QA_COLUMNS = [
    "rptd_pr",
    "entrd_vol_qt",
    "yld_pt",
    "days_to_sttl_ct",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_final_manifest(root: Path) -> pd.DataFrame:
    path = root / "data" / "manifests" / "extractions" / "step04a_trace_final_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 04A final manifest: {path}")

    df = pd.read_csv(path)
    if "ok" in df.columns:
        df = df.loc[truthy_series(df["ok"])].copy()

    if "output_path" not in df.columns:
        raise ValueError("Final manifest is missing output_path.")

    df = df.drop_duplicates("output_path", keep="last").reset_index(drop=True)
    return df


def value_counts_from_table(path: Path, columns: list[str]) -> dict[str, Counter]:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in columns if c in available]

    counters = {c: Counter() for c in columns}
    if not selected:
        return counters

    table = pf.read(columns=selected)

    for col in selected:
        s = table[col].to_pandas()
        vc = s.astype("string").fillna("<NA>").value_counts(dropna=False)
        counters[col].update({str(k): int(v) for k, v in vc.items()})

    for col in columns:
        if col not in selected:
            counters[col].update({"<MISSING_COLUMN>": int(pf.metadata.num_rows)})

    return counters


def numeric_qa_from_table(path: Path, columns: list[str]) -> dict[str, dict[str, Any]]:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in columns if c in available]
    out: dict[str, dict[str, Any]] = {}

    if not selected:
        return {
            c: {
                "n": 0,
                "nulls": int(pf.metadata.num_rows),
                "le_zero": None,
                "lt_zero": None,
                "min": None,
                "max": None,
            }
            for c in columns
        }

    table = pf.read(columns=selected)

    for col in columns:
        if col not in selected:
            out[col] = {
                "n": 0,
                "nulls": int(pf.metadata.num_rows),
                "le_zero": None,
                "lt_zero": None,
                "min": None,
                "max": None,
            }
            continue

        s = pd.to_numeric(table[col].to_pandas(), errors="coerce")
        nonnull = s.dropna()
        out[col] = {
            "n": int(nonnull.shape[0]),
            "nulls": int(s.isna().sum()),
            "le_zero": int((nonnull <= 0).sum()),
            "lt_zero": int((nonnull < 0).sum()),
            "min": float(nonnull.min()) if len(nonnull) else None,
            "max": float(nonnull.max()) if len(nonnull) else None,
        }

    return out


def scan_one(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["output_path"]))
    rec: dict[str, Any] = {
        "output_path": str(path),
        "source_stage": str(row.get("source_stage", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "manifest_rows": int(float(row.get("n_rows", 0) or 0)),
        "exists": path.exists(),
        "ok": False,
        "error": "",
        "code_counts": {},
        "numeric_qa": {},
    }

    if not path.exists():
        rec["error"] = "missing_file"
        return rec

    try:
        pf = pq.ParquetFile(path)
        rec["footer_rows"] = int(pf.metadata.num_rows)
        rec["code_counts"] = {
            k: dict(v) for k, v in value_counts_from_table(path, CODE_COLUMNS).items()
        }
        rec["numeric_qa"] = numeric_qa_from_table(path, NUMERIC_QA_COLUMNS)
        rec["ok"] = True
    except Exception as exc:
        rec["error"] = repr(exc)

    return rec


def merge_code_counts(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged = {c: Counter() for c in CODE_COLUMNS}
    for rec in results:
        if not rec.get("ok"):
            continue
        for col, counts in rec.get("code_counts", {}).items():
            merged[col].update({str(k): int(v) for k, v in counts.items()})
    return {col: dict(counter.most_common()) for col, counter in merged.items()}


def merge_numeric_qa(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for col in NUMERIC_QA_COLUMNS:
        n = 0
        nulls = 0
        le_zero = 0
        lt_zero = 0
        mins = []
        maxs = []

        for rec in results:
            if not rec.get("ok"):
                continue
            qa = rec.get("numeric_qa", {}).get(col, {})
            n += int(qa.get("n") or 0)
            nulls += int(qa.get("nulls") or 0)
            if qa.get("le_zero") is not None:
                le_zero += int(qa.get("le_zero") or 0)
            if qa.get("lt_zero") is not None:
                lt_zero += int(qa.get("lt_zero") or 0)
            if qa.get("min") is not None:
                mins.append(float(qa["min"]))
            if qa.get("max") is not None:
                maxs.append(float(qa["max"]))

        merged[col] = {
            "n": n,
            "nulls": nulls,
            "le_zero": le_zero,
            "lt_zero": lt_zero,
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
        }

    return merged


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest = load_final_manifest(root)
    if args.limit_partitions and args.limit_partitions > 0:
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

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(scan_one, row) for row in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)
            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                scanned_rows = sum(int(r.get("footer_rows") or 0) for r in results if r.get("ok"))
                print(f"progress {i}/{len(futures)} ok={ok} failed={failed} footer_rows={scanned_rows:,}", flush=True)

    ok_results = [r for r in results if r.get("ok")]
    failed_results = [r for r in results if not r.get("ok")]

    code_counts = merge_code_counts(results)
    numeric_qa = merge_numeric_qa(results)

    partition_summary = pd.DataFrame(
        [
            {
                "output_path": r.get("output_path"),
                "source_stage": r.get("source_stage"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "manifest_rows": r.get("manifest_rows"),
                "footer_rows": r.get("footer_rows"),
                "ok": r.get("ok"),
                "error": r.get("error"),
            }
            for r in results
        ]
    ).sort_values(["start_date", "end_date", "output_path"])

    partition_summary_path = out_dir / "step04d_trace_code_policy_partition_summary.csv"
    code_counts_path = out_dir / "step04d_trace_code_policy_counts.json"
    numeric_qa_path = out_dir / "step04d_trace_numeric_qa.json"
    summary_path = out_dir / "step04d_trace_code_policy_summary.json"

    partition_summary.to_csv(partition_summary_path, index=False)
    write_json(code_counts_path, code_counts)
    write_json(numeric_qa_path, numeric_qa)

    total_footer_rows = int(sum(int(r.get("footer_rows") or 0) for r in ok_results))
    expected_rows = int(pd.to_numeric(manifest["n_rows"], errors="coerce").fillna(0).sum())

    summary = {
        "ok": len(failed_results) == 0 and total_footer_rows == expected_rows,
        "run_id": run_id,
        "workspace": str(root),
        "partitions_scanned": int(len(results)),
        "partitions_ok": int(len(ok_results)),
        "partitions_failed": int(len(failed_results)),
        "expected_rows_from_manifest": expected_rows,
        "footer_rows_scanned": total_footer_rows,
        "footer_minus_expected": int(total_footer_rows - expected_rows),
        "failed_examples": failed_results[:5],
        "code_counts_path": str(code_counts_path),
        "numeric_qa_path": str(numeric_qa_path),
        "partition_summary_path": str(partition_summary_path),
        "note": "Local-only selected-column code audit. No WRDS. No extraction.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step04d_trace_code_policy_audit_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, code_counts_path, numeric_qa_path, partition_summary_path]:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"CODE_COUNTS={code_counts_path}")
    print(f"NUMERIC_QA={numeric_qa_path}")
    print(f"PARTITION_SUMMARY={partition_summary_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 04D local TRACE code-policy audit. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--limit-partitions", type=int, default=0)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
