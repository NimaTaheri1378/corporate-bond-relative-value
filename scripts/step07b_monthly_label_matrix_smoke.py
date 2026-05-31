#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURE_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "model",
    "n_issues",
    "trade_count",
    "gross_volume",
    "weight_sum",
    "years_to_maturity_wavg",
    "maturity_bucket",
    "yld_pt_wavg",
    "fitted_yld_pt",
    "residual_yld_pt",
    "abs_residual_yld_pt",
    "residual_over_rmse",
    "abs_residual_over_rmse",
    "issuer_date_residual_z",
    "issuer_date_abs_residual_z",
    "issuer_date_residual_pctile",
    "issuer_date_bucket_residual_z",
    "issuer_date_bucket_abs_residual_z",
    "is_cheap_vs_curve",
    "is_rich_vs_curve",
    "rptd_pr_wavg",
    "curve_rmse",
    "curve_mae",
    "curve_score",
    "curve_unstable",
    "log_gross_volume",
    "log_weight_sum",
    "log_trade_count",
    "liquidity_weight",
    "curve_support_score",
]

BONDRET_COLUMNS = [
    "date",
    "issue_id",
    "cusip",
    "bond_sym_id",
    "company_symbol",
    "ret_eom",
    "ret_l5m",
    "ret_ldm",
    "price_eom",
    "duration",
    "tmt",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def normalize_id_value(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"<NA>", "NA", "NAN", "NONE", "NULL"}:
        return None
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_id_value).astype("string")


def load_feature_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_residual_features_v1_{universe}_validated_nonempty_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing residual-feature validated manifest: {path}")

    df = pd.read_csv(path)
    df["rows"] = pd.to_numeric(df["rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()
    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def read_feature_partition(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in FEATURE_COLUMNS if c in available]

    if not selected:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = pf.read(columns=selected).to_pandas()

    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df.loc[:, FEATURE_COLUMNS].copy()


def load_feature_sample(root: Path, universe: str, limit_partitions: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_all = load_feature_manifest(root, universe)
    manifest = choose_partitions(manifest_all, limit_partitions)

    records = []
    frames = []

    print(f"feature_partitions={len(manifest)}")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {
            pool.submit(read_feature_partition, Path(str(row["output_path"]))): row
            for _, row in manifest.iterrows()
        }

        for i, fut in enumerate(as_completed(futs), start=1):
            row = futs[fut]
            rec = {
                "output_path": str(row["output_path"]),
                "expected_rows": int(row["rows"]),
                "rows_read": 0,
                "ok": False,
                "error": "",
            }

            try:
                part = fut.result()
                rec["rows_read"] = int(len(part))
                rec["ok"] = True
                frames.append(part)
            except Exception as exc:
                rec["error"] = repr(exc)

            records.append(rec)

            if i == 1 or i % 20 == 0 or i == len(futs):
                ok = sum(1 for r in records if r["ok"])
                rows_read = sum(int(r["rows_read"]) for r in records if r["ok"])
                print(f"progress_features {i}/{len(futs)} ok={ok} rows={rows_read:,}", flush=True)

    detail = pd.DataFrame(records)
    meta = {
        "manifest_partitions_total": int(len(manifest_all)),
        "sample_partitions": int(len(manifest)),
        "sample_ok_partitions": int(detail["ok"].sum()) if not detail.empty else 0,
        "sample_failed_partitions": int((~detail["ok"]).sum()) if not detail.empty else 0,
        "sample_rows": int(detail["rows_read"].sum()) if not detail.empty else 0,
    }

    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FEATURE_COLUMNS)
    return features, meta


def aggregate_features_monthly(features: pd.DataFrame) -> pd.DataFrame:
    x = features.copy()

    if x.empty:
        return pd.DataFrame()

    x["trd_exctn_dt"] = pd.to_datetime(x["trd_exctn_dt"], errors="coerce")
    x["signal_month"] = x["trd_exctn_dt"].dt.to_period("M").dt.to_timestamp("M")
    x["label_month"] = x["signal_month"] + pd.offsets.MonthEnd(1)
    x["issue_key"] = normalize_id_series(x["issue_id"])

    x = x.dropna(subset=["trd_exctn_dt", "signal_month", "label_month", "issue_key"]).copy()

    numeric_cols = [
        "residual_yld_pt",
        "abs_residual_yld_pt",
        "residual_over_rmse",
        "abs_residual_over_rmse",
        "issuer_date_residual_z",
        "issuer_date_abs_residual_z",
        "issuer_date_residual_pctile",
        "issuer_date_bucket_residual_z",
        "issuer_date_bucket_abs_residual_z",
        "years_to_maturity_wavg",
        "yld_pt_wavg",
        "rptd_pr_wavg",
        "curve_rmse",
        "curve_mae",
        "curve_score",
        "n_issues",
        "trade_count",
        "gross_volume",
        "weight_sum",
        "log_gross_volume",
        "log_weight_sum",
        "log_trade_count",
        "liquidity_weight",
        "curve_support_score",
    ]

    for col in numeric_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    for col in ["is_cheap_vs_curve", "is_rich_vs_curve", "curve_unstable"]:
        if col in x.columns:
            x[col] = x[col].fillna(False).astype(bool)

    x = x.sort_values(["signal_month", "issue_key", "trd_exctn_dt"], kind="mergesort")

    g = x.groupby(["signal_month", "label_month", "issue_key"], dropna=True, sort=False)

    out = g.agg(
        n_obs=("issue_key", "size"),
        first_signal_date=("trd_exctn_dt", "min"),
        last_signal_date=("trd_exctn_dt", "max"),
        issuer_id=("issuer_id", "last"),
        mean_residual_yld_pt=("residual_yld_pt", "mean"),
        last_residual_yld_pt=("residual_yld_pt", "last"),
        mean_abs_residual_yld_pt=("abs_residual_yld_pt", "mean"),
        mean_residual_over_rmse=("residual_over_rmse", "mean"),
        last_residual_over_rmse=("residual_over_rmse", "last"),
        mean_issuer_date_residual_z=("issuer_date_residual_z", "mean"),
        last_issuer_date_residual_z=("issuer_date_residual_z", "last"),
        max_abs_issuer_date_residual_z=("issuer_date_abs_residual_z", "max"),
        mean_residual_pctile=("issuer_date_residual_pctile", "mean"),
        last_residual_pctile=("issuer_date_residual_pctile", "last"),
        mean_bucket_residual_z=("issuer_date_bucket_residual_z", "mean"),
        last_bucket_residual_z=("issuer_date_bucket_residual_z", "last"),
        mean_years_to_maturity=("years_to_maturity_wavg", "mean"),
        last_years_to_maturity=("years_to_maturity_wavg", "last"),
        mean_curve_rmse=("curve_rmse", "mean"),
        last_curve_rmse=("curve_rmse", "last"),
        mean_n_issues=("n_issues", "mean"),
        max_n_issues=("n_issues", "max"),
        total_trade_count=("trade_count", "sum"),
        total_gross_volume=("gross_volume", "sum"),
        mean_log_gross_volume=("log_gross_volume", "mean"),
        mean_liquidity_weight=("liquidity_weight", "mean"),
        mean_curve_support_score=("curve_support_score", "mean"),
        cheap_share=("is_cheap_vs_curve", "mean"),
        rich_share=("is_rich_vs_curve", "mean"),
        unstable_curve_share=("curve_unstable", "mean"),
    ).reset_index()

    out["issuer_id"] = pd.to_numeric(out["issuer_id"], errors="coerce")
    out["signal_year"] = pd.to_datetime(out["signal_month"]).dt.year
    out["label_year"] = pd.to_datetime(out["label_month"]).dt.year

    return out


def bondret_files(root: Path) -> list[Path]:
    d = root / "data" / "raw" / "wrds" / "v1" / "bondret_monthly"
    if not d.exists():
        raise FileNotFoundError(f"Missing bondret_monthly raw dir: {d}")
    files = sorted(d.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {d}")
    return files


def read_bondret_file(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in BONDRET_COLUMNS if c in available]

    if not selected:
        return pd.DataFrame(columns=BONDRET_COLUMNS)

    df = pf.read(columns=selected).to_pandas()

    for c in BONDRET_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df.loc[:, BONDRET_COLUMNS].copy()


def load_bondret_monthly(root: Path, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = bondret_files(root)
    frames = []
    records = []

    print(f"bondret_files={len(files)}")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(read_bondret_file, p): p for p in files}

        for i, fut in enumerate(as_completed(futs), start=1):
            p = futs[fut]
            rec = {"path": str(p), "rows": 0, "ok": False, "error": ""}

            try:
                part = fut.result()
                rec["rows"] = int(len(part))
                rec["ok"] = True
                frames.append(part)
            except Exception as exc:
                rec["error"] = repr(exc)

            records.append(rec)

            if i == 1 or i % 25 == 0 or i == len(futs):
                ok = sum(1 for r in records if r["ok"])
                rows = sum(int(r["rows"]) for r in records if r["ok"])
                print(f"progress_bondret {i}/{len(futs)} ok={ok} rows={rows:,}", flush=True)

    if not frames:
        return pd.DataFrame(columns=BONDRET_COLUMNS), {"files": len(files), "ok_files": 0, "failed_files": len(files), "rows": 0}

    df = pd.concat(frames, ignore_index=True)

    df["label_month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    df["issue_key"] = normalize_id_series(df["issue_id"])

    for col in ["ret_eom", "ret_l5m", "ret_ldm", "price_eom", "duration", "tmt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["label_month", "issue_key"]).copy()

    # Keep one record per issue-month. Prefer finite ret_eom and finite price.
    df["_has_ret"] = df["ret_eom"].replace([np.inf, -np.inf], np.nan).notna().astype(int)
    df["_has_price"] = df["price_eom"].replace([np.inf, -np.inf], np.nan).notna().astype(int)

    df = df.sort_values(
        ["label_month", "issue_key", "_has_ret", "_has_price"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    before = len(df)
    df = df.drop_duplicates(["label_month", "issue_key"], keep="last").reset_index(drop=True)

    meta = {
        "files": len(files),
        "ok_files": int(sum(1 for r in records if r["ok"])),
        "failed_files": int(sum(1 for r in records if not r["ok"])),
        "rows_raw": int(sum(int(r["rows"]) for r in records if r["ok"])),
        "rows_after_key_date_drop": int(before),
        "rows_after_issue_month_dedup": int(len(df)),
        "duplicate_issue_month_rows_removed": int(before - len(df)),
        "date_min": None if df.empty else str(df["label_month"].min().date()),
        "date_max": None if df.empty else str(df["label_month"].max().date()),
    }

    return df, meta


def summarize_numeric(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"n": 0, "mean": None, "std": None, "p01": None, "p50": None, "p99": None, "min": None, "max": None}
    return {
        "n": int(len(x)),
        "mean": round(float(x.mean()), 10),
        "std": round(float(x.std(ddof=1)), 10) if len(x) > 1 else 0.0,
        "p01": round(float(x.quantile(0.01)), 10),
        "p50": round(float(x.quantile(0.50)), 10),
        "p99": round(float(x.quantile(0.99)), 10),
        "min": round(float(x.min()), 10),
        "max": round(float(x.max()), 10),
    }


def build_label_matrix(monthly_features: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    ret_cols = [
        "label_month",
        "issue_key",
        "ret_eom",
        "ret_l5m",
        "ret_ldm",
        "price_eom",
        "duration",
        "tmt",
    ]
    r = returns.loc[:, [c for c in ret_cols if c in returns.columns]].copy()

    joined = monthly_features.merge(
        r,
        on=["label_month", "issue_key"],
        how="left",
        copy=False,
    )

    joined["has_ret_eom"] = joined["ret_eom"].replace([np.inf, -np.inf], np.nan).notna()
    joined["label_ret_1m_raw"] = joined["ret_eom"]

    # Conservative label filter for the first baseline. Keep severe losses but remove busted extreme positives.
    joined["label_ret_1m"] = pd.to_numeric(joined["label_ret_1m_raw"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    joined["label_ret_1m_is_usable"] = joined["label_ret_1m"].between(-1.0, 1.0, inclusive="both")
    joined.loc[~joined["label_ret_1m_is_usable"], "label_ret_1m"] = np.nan

    joined["label_available"] = joined["label_ret_1m"].notna()
    joined["signal_to_label_days"] = (joined["label_month"] - joined["last_signal_date"]).dt.days

    return joined


def make_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("signal_year", dropna=False)
    out = g.agg(
        feature_issue_month_rows=("issue_key", "size"),
        labeled_rows=("label_available", "sum"),
        raw_return_rows=("has_ret_eom", "sum"),
        unique_issues=("issue_key", "nunique"),
        mean_label_ret_1m=("label_ret_1m", "mean"),
        median_label_ret_1m=("label_ret_1m", "median"),
        mean_last_residual_z=("last_issuer_date_residual_z", "mean"),
    ).reset_index()

    out["label_coverage_pct"] = (100.0 * out["labeled_rows"] / out["feature_issue_month_rows"]).round(6)
    out["raw_return_coverage_pct"] = (100.0 * out["raw_return_rows"] / out["feature_issue_month_rows"]).round(6)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 07B monthly bondret label-matrix smoke. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--feature-partitions", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    out_dir = root / "data" / "interim" / "monthly_label_matrix_smoke_v1" / f"run_id={run_id}"

    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"feature_partitions={args.feature_partitions}")

    features, feature_meta = load_feature_sample(
        root=root,
        universe=args.universe,
        limit_partitions=args.feature_partitions,
        workers=args.workers,
    )

    monthly_features = aggregate_features_monthly(features)

    returns, return_meta = load_bondret_monthly(root=root, workers=args.workers)
    label_matrix = build_label_matrix(monthly_features, returns)

    # Local sample parquet. Do not upload.
    matrix_path = out_dir / "monthly_label_matrix_smoke.parquet"
    label_matrix.to_parquet(matrix_path, index=False, compression="zstd")

    usable = label_matrix.loc[label_matrix["label_available"]].copy()
    year_summary = make_year_summary(label_matrix)

    year_summary_path = table_dir / "step07b_monthly_label_matrix_year_summary.csv"
    year_summary.to_csv(year_summary_path, index=False)

    label_stats = {
        "label_ret_1m": summarize_numeric(usable["label_ret_1m"] if "label_ret_1m" in usable else pd.Series(dtype=float)),
        "last_issuer_date_residual_z": summarize_numeric(usable["last_issuer_date_residual_z"] if "last_issuer_date_residual_z" in usable else pd.Series(dtype=float)),
        "mean_issuer_date_residual_z": summarize_numeric(usable["mean_issuer_date_residual_z"] if "mean_issuer_date_residual_z" in usable else pd.Series(dtype=float)),
        "signal_to_label_days": summarize_numeric(usable["signal_to_label_days"] if "signal_to_label_days" in usable else pd.Series(dtype=float)),
    }

    coverage_rows = int(len(label_matrix))
    raw_return_rows = int(label_matrix["has_ret_eom"].sum()) if "has_ret_eom" in label_matrix else 0
    labeled_rows = int(label_matrix["label_available"].sum()) if "label_available" in label_matrix else 0
    extreme_or_missing = int(raw_return_rows - labeled_rows)

    summary = {
        "ok": bool(len(label_matrix) > 0 and labeled_rows > 0),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "feature_meta": feature_meta,
        "bondret_meta": return_meta,
        "monthly_feature_rows": int(len(monthly_features)),
        "label_matrix_rows": coverage_rows,
        "raw_return_rows": raw_return_rows,
        "labeled_rows_after_filter": labeled_rows,
        "label_coverage_pct": None if coverage_rows == 0 else round(100.0 * labeled_rows / coverage_rows, 6),
        "raw_return_coverage_pct": None if coverage_rows == 0 else round(100.0 * raw_return_rows / coverage_rows, 6),
        "raw_returns_excluded_by_filter_or_missing": extreme_or_missing,
        "label_filter": "label_ret_1m = ret_eom for next calendar month, finite and between -1.0 and 1.0 inclusive",
        "date_min_signal": None if monthly_features.empty else str(monthly_features["signal_month"].min().date()),
        "date_max_signal": None if monthly_features.empty else str(monthly_features["signal_month"].max().date()),
        "date_min_label": None if monthly_features.empty else str(monthly_features["label_month"].min().date()),
        "date_max_label": None if monthly_features.empty else str(monthly_features["label_month"].max().date()),
        "distinct_issues_in_matrix": int(label_matrix["issue_key"].nunique()) if "issue_key" in label_matrix else 0,
        "distinct_issues_labeled": int(usable["issue_key"].nunique()) if "issue_key" in usable else 0,
        "label_stats": label_stats,
        "year_summary_path": str(year_summary_path),
        "local_matrix_do_not_upload": str(matrix_path),
        "note": "Smoke only. Monthly next-month bondret labels. No WRDS. Upload bundle only, not label matrix parquet.",
    }

    summary_path = table_dir / "step07b_monthly_label_matrix_smoke_summary.json"
    write_json(summary_path, summary)

    bundle = log_dir / f"step07b_monthly_label_matrix_smoke_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, year_summary_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"YEAR_SUMMARY={year_summary_path}")
    print(f"LOCAL_MATRIX_DO_NOT_UPLOAD={matrix_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

