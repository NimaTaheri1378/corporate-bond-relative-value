#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


KEY_TERMS = [
    "cusip",
    "isin",
    "figi",
    "bond",
    "bond_sym",
    "issuer",
    "issue",
    "ticker",
    "company",
    "symbol",
    "permno",
    "permco",
    "gvkey",
    "maturity",
    "mtrty",
    "coupon",
    "rating",
    "amount",
    "outstanding",
    "seniority",
    "call",
    "put",
    "convert",
    "144a",
    "currency",
    "dated",
    "offering",
    "effective",
    "date",
]

TRACE_KEY_COLUMNS = [
    "cusip_id",
    "bond_sym_id",
    "company_symbol",
    "trd_exctn_dt",
    "rptd_pr",
    "entrd_vol_qt",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def parquet_footer(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "file_size_bytes": 0,
        "schema_sha256": "",
        "columns": [],
        "error": "",
    }

    if not path.exists():
        rec["error"] = "missing_file"
        return rec

    try:
        pf = pq.ParquetFile(path)
        schema_text = str(pf.schema_arrow)
        rec["rows"] = int(pf.metadata.num_rows)
        rec["file_size_bytes"] = int(path.stat().st_size)
        rec["schema_sha256"] = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
        rec["columns"] = list(pf.schema_arrow.names)
        return rec
    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def discover_parquet_groups(raw_root: Path, include_trace_raw: bool) -> dict[str, list[Path]]:
    if not raw_root.exists():
        return {}

    groups: dict[str, list[Path]] = defaultdict(list)

    for path in raw_root.rglob("*.parquet"):
        rel = path.relative_to(raw_root)
        if len(rel.parts) < 2:
            table = rel.parts[0].replace(".parquet", "")
        else:
            table = rel.parts[0]

        table_lower = table.lower()
        if not include_trace_raw and table_lower.startswith("trace"):
            continue

        groups[table].append(path)

    return {k: sorted(v) for k, v in sorted(groups.items())}


def classify_table(table: str, columns: list[str]) -> str:
    name = table.lower()
    col_text = " ".join(c.lower() for c in columns)

    if any(x in name for x in ["fisd", "issue", "coupon", "rating", "amount"]):
        return "security_master_candidate"

    if "link" in name or any(x in col_text for x in ["permno", "permco", "gvkey"]):
        return "link_candidate"

    if "return" in name or "bondret" in name or any(x in col_text for x in ["ret", "return"]):
        return "return_panel_candidate"

    if any(x in col_text for x in ["cusip", "issuer", "coupon", "maturity", "rating"]):
        return "security_or_identifier_candidate"

    return "other_raw_table"


def candidate_columns(columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        low = col.lower()
        if any(term in low for term in KEY_TERMS):
            out.append(col)
    return out


def aggregate_inventory(groups: dict[str, list[Path]], workers: int) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    all_tasks: list[tuple[str, Path]] = []
    for table, files in groups.items():
        for path in files:
            all_tasks.append((table, path))

    footer_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fut_to_task = {pool.submit(parquet_footer, path): (table, path) for table, path in all_tasks}
        for fut in as_completed(fut_to_task):
            table, _ = fut_to_task[fut]
            footer_by_table[table].append(fut.result())

    rows = []
    for table, records in sorted(footer_by_table.items()):
        ok_records = [r for r in records if not r.get("error")]
        errors = [r for r in records if r.get("error")]

        total_rows = int(sum(int(r.get("rows") or 0) for r in ok_records))
        total_bytes = int(sum(int(r.get("file_size_bytes") or 0) for r in ok_records))

        schema_counts = Counter(str(r.get("schema_sha256", "")) for r in ok_records)
        schema_counts.pop("", None)

        col_union = sorted(set(c for r in ok_records for c in r.get("columns", [])))
        key_cols = candidate_columns(col_union)

        rows.append(
            {
                "table": table,
                "class": classify_table(table, col_union),
                "n_files": len(records),
                "n_ok_files": len(ok_records),
                "n_error_files": len(errors),
                "footer_rows": total_rows,
                "file_size_bytes": total_bytes,
                "n_schema_fingerprints": len(schema_counts),
                "schema_fingerprint_counts": json.dumps(dict(schema_counts.most_common()), sort_keys=True),
                "n_columns_union": len(col_union),
                "columns_union": "|".join(col_union),
                "candidate_key_columns": "|".join(key_cols),
                "errors_sample": json.dumps(errors[:3], sort_keys=True),
            }
        )

    inv = pd.DataFrame(rows).sort_values(["class", "table"]).reset_index(drop=True)
    return inv, footer_by_table


def selected_candidate_columns(columns: list[str]) -> list[str]:
    keys = candidate_columns(columns)
    preferred = []

    for token in [
        "cusip",
        "issuer",
        "issue",
        "bond",
        "company",
        "symbol",
        "coupon",
        "maturity",
        "mtrty",
        "rating",
        "amount",
        "outstanding",
        "permno",
        "gvkey",
    ]:
        for col in keys:
            if token in col.lower() and col not in preferred:
                preferred.append(col)

    return preferred[:30]


def sample_table_key_profile(
    table: str,
    files: list[Path],
    columns_union: list[str],
    max_files: int,
    max_rows_total: int,
) -> list[dict[str, Any]]:
    columns = selected_candidate_columns(columns_union)
    if not columns:
        return []

    frames = []
    rows_left = max_rows_total

    for path in files[:max_files]:
        if rows_left <= 0:
            break

        try:
            pf = pq.ParquetFile(path)
            available = set(pf.schema_arrow.names)
            selected = [c for c in columns if c in available]
            if not selected:
                continue

            read_rows = 0
            batches = []
            for batch in pf.iter_batches(batch_size=min(50_000, rows_left), columns=selected):
                df = batch.to_pandas()
                batches.append(df)
                read_rows += len(df)
                rows_left -= len(df)
                if rows_left <= 0:
                    break

            if batches:
                part = pd.concat(batches, ignore_index=True)
                frames.append(part)

        except Exception:
            continue

    if not frames:
        return []

    sample = pd.concat(frames, ignore_index=True)
    out = []

    for col in sample.columns:
        s = sample[col]
        out.append(
            {
                "table": table,
                "column": col,
                "sample_rows": int(len(s)),
                "non_null_rows": int(s.notna().sum()),
                "null_rows": int(s.isna().sum()),
                "distinct_non_null": int(s.dropna().astype(str).nunique()),
                "pandas_dtype": str(s.dtype),
                "note": "Counts only; no raw identifier values written.",
            }
        )

    return out


def build_key_profiles(
    groups: dict[str, list[Path]],
    inventory: pd.DataFrame,
    max_files_per_table: int,
    max_rows_per_table: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, inv_row in inventory.iterrows():
        table = str(inv_row["table"])
        class_name = str(inv_row["class"])

        if class_name not in {
            "security_master_candidate",
            "security_or_identifier_candidate",
            "link_candidate",
            "return_panel_candidate",
        }:
            continue

        columns_union = str(inv_row.get("columns_union", "")).split("|")
        columns_union = [c for c in columns_union if c]

        rows.extend(
            sample_table_key_profile(
                table=table,
                files=groups.get(table, []),
                columns_union=columns_union,
                max_files=max_files_per_table,
                max_rows_total=max_rows_per_table,
            )
        )

    return pd.DataFrame(rows)


def sample_manifest_paths(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    if manifest.empty:
        return manifest

    manifest = manifest.copy()
    if "clean_rows" in manifest.columns:
        manifest["clean_rows"] = pd.to_numeric(manifest["clean_rows"], errors="coerce").fillna(0)
        manifest = manifest.loc[manifest["clean_rows"] > 0].copy()

    manifest = manifest.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)

    if len(manifest) <= n:
        return manifest

    idx = sorted(set(round(i * (len(manifest) - 1) / (n - 1)) for i in range(n)))
    return manifest.iloc[idx].reset_index(drop=True)


def trace_key_sample(root: Path, universe: str, sample_partitions: int, max_rows_per_partition: int) -> dict[str, Any]:
    manifest_path = root / "data" / "manifests" / "processed" / f"trace_clean_v1_{universe}_nonempty_manifest.csv"

    rec: dict[str, Any] = {
        "universe": universe,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "sample_partitions": 0,
        "rows_sampled": 0,
        "distinct_cusip_id": 0,
        "distinct_bond_sym_id": 0,
        "distinct_company_symbol": 0,
        "min_trade_date_sample": None,
        "max_trade_date_sample": None,
        "errors": [],
        "note": "Counts only; no raw identifier values written.",
    }

    if not manifest_path.exists():
        return rec

    manifest = pd.read_csv(manifest_path)
    sample = sample_manifest_paths(manifest, sample_partitions)

    frames = []
    for _, row in sample.iterrows():
        path = Path(str(row["output_path"]))
        try:
            pf = pq.ParquetFile(path)
            available = set(pf.schema_arrow.names)
            selected = [c for c in TRACE_KEY_COLUMNS if c in available]
            if not selected:
                continue

            batches = []
            rows_left = max_rows_per_partition
            for batch in pf.iter_batches(batch_size=min(50_000, rows_left), columns=selected):
                df = batch.to_pandas()
                batches.append(df)
                rows_left -= len(df)
                if rows_left <= 0:
                    break

            if batches:
                frames.append(pd.concat(batches, ignore_index=True))

        except Exception as exc:
            rec["errors"].append({"path": str(path), "error": repr(exc)})

    if not frames:
        return rec

    df = pd.concat(frames, ignore_index=True)
    rec["sample_partitions"] = int(len(sample))
    rec["rows_sampled"] = int(len(df))

    for col, out_key in [
        ("cusip_id", "distinct_cusip_id"),
        ("bond_sym_id", "distinct_bond_sym_id"),
        ("company_symbol", "distinct_company_symbol"),
    ]:
        if col in df.columns:
            rec[out_key] = int(df[col].dropna().astype(str).nunique())

    if "trd_exctn_dt" in df.columns:
        dates = pd.to_datetime(df["trd_exctn_dt"], errors="coerce").dropna()
        if not dates.empty:
            rec["min_trade_date_sample"] = str(dates.min().date())
            rec["max_trade_date_sample"] = str(dates.max().date())

    return rec


def package_outputs(root: Path, run_id: str, paths: list[Path]) -> Path:
    bundle = root / "run_logs" / f"step05a_security_master_local_qa_bundle_{run_id}.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle, "w:gz") as tar:
        for path in paths:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(root)))

    return bundle


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    raw_root = root / "data" / "raw" / "wrds" / "v1"
    table_dir = root / "artifacts" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"raw_root={raw_root}")
    print(f"include_trace_raw={args.include_trace_raw}")

    groups = discover_parquet_groups(raw_root, include_trace_raw=args.include_trace_raw)
    print(f"raw_table_groups={len(groups)}")

    inventory, footer_by_table = aggregate_inventory(groups, workers=args.workers)

    inventory_path = table_dir / "step05a_raw_nontrace_table_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    key_profiles = build_key_profiles(
        groups=groups,
        inventory=inventory,
        max_files_per_table=args.profile_files_per_table,
        max_rows_per_table=args.profile_rows_per_table,
    )

    key_profile_path = table_dir / "step05a_raw_nontrace_key_profile.csv"
    key_profiles.to_csv(key_profile_path, index=False)

    trace_samples = {
        "core_public": trace_key_sample(
            root=root,
            universe="core_public",
            sample_partitions=args.trace_sample_partitions,
            max_rows_per_partition=args.trace_max_rows_per_partition,
        ),
        "extended_regular": trace_key_sample(
            root=root,
            universe="extended_regular",
            sample_partitions=args.trace_sample_partitions,
            max_rows_per_partition=args.trace_max_rows_per_partition,
        ),
    }

    trace_sample_path = table_dir / "step05a_trace_clean_key_sample_summary.json"
    write_json(trace_sample_path, trace_samples)

    candidate_tables = inventory.loc[
        inventory["class"].isin(
            [
                "security_master_candidate",
                "security_or_identifier_candidate",
                "link_candidate",
                "return_panel_candidate",
            ]
        )
    ].copy()

    summary = {
        "ok": bool(len(inventory) > 0 and len(candidate_tables) > 0),
        "run_id": run_id,
        "workspace": str(root),
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "include_trace_raw": bool(args.include_trace_raw),
        "raw_table_groups": int(len(groups)),
        "inventory_tables": int(len(inventory)),
        "candidate_tables": int(len(candidate_tables)),
        "candidate_table_names_by_class": {
            class_name: sorted(candidate_tables.loc[candidate_tables["class"] == class_name, "table"].astype(str).tolist())
            for class_name in sorted(candidate_tables["class"].dropna().unique())
        },
        "total_nontrace_footer_rows": int(pd.to_numeric(inventory.get("footer_rows", 0), errors="coerce").fillna(0).sum()) if not inventory.empty else 0,
        "tables_with_errors": int((pd.to_numeric(inventory.get("n_error_files", 0), errors="coerce").fillna(0) > 0).sum()) if not inventory.empty else 0,
        "trace_key_samples": trace_samples,
        "inventory_path": str(inventory_path),
        "key_profile_path": str(key_profile_path),
        "trace_sample_path": str(trace_sample_path),
        "note": "Local-only raw non-TRACE QA. No WRDS, no extraction, no raw identifier values written.",
    }

    summary_path = table_dir / "step05a_security_master_local_qa_summary.json"
    write_json(summary_path, summary)

    bundle = package_outputs(
        root=root,
        run_id=run_id,
        paths=[summary_path, inventory_path, key_profile_path, trace_sample_path],
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"INVENTORY={inventory_path}")
    print(f"KEY_PROFILE={key_profile_path}")
    print(f"TRACE_SAMPLE={trace_sample_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 05A local FISD/security-master raw QA. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--include-trace-raw", action="store_true")
    parser.add_argument("--profile-files-per-table", type=int, default=3)
    parser.add_argument("--profile-rows-per-table", type=int, default=100_000)
    parser.add_argument("--trace-sample-partitions", type=int, default=12)
    parser.add_argument("--trace-max-rows-per-partition", type=int, default=50_000)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
