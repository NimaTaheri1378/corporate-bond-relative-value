#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY_FEATURES = [
    "last_issuer_date_residual_z",
    "mean_issuer_date_residual_z",
    "max_abs_issuer_date_residual_z",
    "last_residual_over_rmse",
    "mean_residual_over_rmse",
    "last_residual_pctile",
    "mean_residual_pctile",
    "last_bucket_residual_z",
    "mean_bucket_residual_z",
    "mean_curve_rmse",
    "last_curve_rmse",
    "mean_n_issues",
    "max_n_issues",
    "total_trade_count",
    "total_gross_volume",
    "mean_log_gross_volume",
    "mean_liquidity_weight",
    "mean_curve_support_score",
    "cheap_share",
    "rich_share",
    "unstable_curve_share",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_summary(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "min": None,
            "max": None,
        }

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


def safe_spearman(x: pd.Series, y: pd.Series, min_n: int) -> float | None:
    xx = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    m = xx.notna() & yy.notna()

    if int(m.sum()) < min_n:
        return None

    xr = xx.loc[m].rank(method="average")
    yr = yy.loc[m].rank(method="average")
    val = xr.corr(yr)

    if pd.isna(val):
        return None
    return float(val)


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("signal_year", dropna=False)

    out = g.agg(
        matrix_rows=("issue_key", "size"),
        labeled_rows=("label_available", "sum"),
        unique_issues=("issue_key", "nunique"),
        mean_label_ret_1m=("label_ret_1m", "mean"),
        median_label_ret_1m=("label_ret_1m", "median"),
        mean_last_residual_z=("last_issuer_date_residual_z", "mean"),
        mean_abs_residual_z=("max_abs_issuer_date_residual_z", "mean"),
        mean_curve_rmse=("mean_curve_rmse", "mean"),
    ).reset_index()

    out["label_coverage_pct"] = (100.0 * out["labeled_rows"] / out["matrix_rows"]).round(6)
    return out


def build_monthly_ic(df: pd.DataFrame, min_obs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.loc[df["label_available"]].copy()
    rows: list[dict[str, Any]] = []

    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame()

    for month, g in labeled.groupby("signal_month", sort=True):
        if len(g) < min_obs:
            continue

        for feature in KEY_FEATURES:
            if feature not in g.columns:
                continue

            ic = safe_spearman(g[feature], g["label_ret_1m"], min_n=min_obs)
            if ic is None:
                continue

            rows.append(
                {
                    "signal_month": str(pd.to_datetime(month).date()),
                    "feature": feature,
                    "n_obs": int(len(g)),
                    "spearman_ic": round(float(ic), 10),
                }
            )

    monthly = pd.DataFrame(rows)

    if monthly.empty:
        return monthly, pd.DataFrame()

    summary = (
        monthly.groupby("feature", as_index=False)
        .agg(
            months=("signal_month", "nunique"),
            mean_ic=("spearman_ic", "mean"),
            median_ic=("spearman_ic", "median"),
            std_ic=("spearman_ic", "std"),
            positive_month_share=("spearman_ic", lambda s: float((s > 0).mean())),
            p10_ic=("spearman_ic", lambda s: float(s.quantile(0.10))),
            p90_ic=("spearman_ic", lambda s: float(s.quantile(0.90))),
        )
    )

    summary["mean_ic"] = summary["mean_ic"].round(10)
    summary["median_ic"] = summary["median_ic"].round(10)
    summary["std_ic"] = summary["std_ic"].round(10)
    summary["positive_month_share"] = summary["positive_month_share"].round(6)
    summary["p10_ic"] = summary["p10_ic"].round(10)
    summary["p90_ic"] = summary["p90_ic"].round(10)
    summary = summary.sort_values("mean_ic", ascending=False).reset_index(drop=True)

    return monthly, summary


def add_time_split_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Conservative first-pass blocks. Later modeling will use rolling splits with embargo.
    out["split_block_static"] = "unassigned"
    out.loc[out["signal_year"].between(2002, 2016), "split_block_static"] = "train_2002_2016"
    out.loc[out["signal_year"].between(2017, 2019), "split_block_static"] = "valid_2017_2019"
    out.loc[out["signal_year"].between(2020, 2024), "split_block_static"] = "test_2020_2024"
    out.loc[out["signal_year"].ge(2025), "split_block_static"] = "holdout_or_incomplete_2025"

    # Training eligibility for first baseline. 2025 has partial forward-label coverage.
    out["is_model_eligible_1m"] = (
        out["label_available"].astype(bool)
        & out["signal_month"].notna()
        & out["label_month"].notna()
        & out["signal_year"].between(2002, 2024)
    )

    # A simple one-month horizon embargo note is stored in the summary; actual rolling CV comes later.
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 07C validate/promote monthly label matrix. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--run-id", default="", help="Optional Step 07B run_id to verify.")
    parser.add_argument("--min-ic-obs", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    step07b_summary_path = root / "artifacts" / "tables" / "step07b_monthly_label_matrix_smoke_summary.json"
    step07b_year_path = root / "artifacts" / "tables" / "step07b_monthly_label_matrix_year_summary.csv"

    if not step07b_summary_path.exists():
        raise FileNotFoundError(f"Missing Step 07B summary: {step07b_summary_path}")

    step07b = load_json(step07b_summary_path)

    if args.run_id and str(step07b.get("run_id")) != args.run_id:
        raise RuntimeError(
            f"Step07B summary run_id={step07b.get('run_id')} does not match requested --run-id={args.run_id}"
        )

    matrix_path = Path(str(step07b["local_matrix_do_not_upload"]))
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing local Step 07B matrix parquet: {matrix_path}")

    pf = pq.ParquetFile(matrix_path)
    matrix = pf.read().to_pandas()

    matrix["signal_month"] = pd.to_datetime(matrix["signal_month"], errors="coerce")
    matrix["label_month"] = pd.to_datetime(matrix["label_month"], errors="coerce")
    matrix["first_signal_date"] = pd.to_datetime(matrix["first_signal_date"], errors="coerce")
    matrix["last_signal_date"] = pd.to_datetime(matrix["last_signal_date"], errors="coerce")

    if "label_available" in matrix.columns:
        matrix["label_available"] = matrix["label_available"].fillna(False).astype(bool)
    else:
        matrix["label_available"] = matrix["label_ret_1m"].notna()

    matrix["signal_year"] = matrix["signal_month"].dt.year
    matrix["label_year"] = matrix["label_month"].dt.year

    matrix = add_time_split_flags(matrix)

    expected_rows = int(step07b["label_matrix_rows"])
    expected_labeled = int(step07b["labeled_rows_after_filter"])
    expected_raw_return_rows = int(step07b["raw_return_rows"])

    actual_rows = int(len(matrix))
    actual_labeled = int(matrix["label_available"].sum())
    actual_raw_return_rows = int(matrix["has_ret_eom"].fillna(False).astype(bool).sum()) if "has_ret_eom" in matrix else 0

    if actual_rows != expected_rows:
        raise RuntimeError(f"Row mismatch: matrix={actual_rows}, summary={expected_rows}")
    if actual_labeled != expected_labeled:
        raise RuntimeError(f"Labeled mismatch: matrix={actual_labeled}, summary={expected_labeled}")
    if actual_raw_return_rows != expected_raw_return_rows:
        raise RuntimeError(f"Raw-return mismatch: matrix={actual_raw_return_rows}, summary={expected_raw_return_rows}")

    out_root = (
        root
        / "data"
        / "processed"
        / "monthly_label_matrix_v1"
        / f"universe={args.universe}"
        / "horizon=1m"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    full_path = out_root / "monthly_label_matrix.parquet"
    labeled_path = out_root / "monthly_label_matrix_labeled.parquet"
    eligible_path = out_root / "monthly_label_matrix_model_eligible.parquet"

    matrix.to_parquet(full_path, index=False, compression="zstd")
    matrix.loc[matrix["label_available"]].to_parquet(labeled_path, index=False, compression="zstd")
    matrix.loc[matrix["is_model_eligible_1m"]].to_parquet(eligible_path, index=False, compression="zstd")

    manifest_dir = root / "data" / "manifests" / "processed"
    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    year_summary = build_year_summary(matrix)
    monthly_ic, ic_summary = build_monthly_ic(matrix, min_obs=args.min_ic_obs)

    year_summary_path = table_dir / "step07c_monthly_label_matrix_year_summary.csv"
    monthly_ic_path = table_dir / "step07c_monthly_label_matrix_monthly_ic.csv"
    ic_summary_path = table_dir / "step07c_monthly_label_matrix_ic_summary.csv"
    summary_path = table_dir / "step07c_monthly_label_matrix_validation_summary.json"
    manifest_path = manifest_dir / f"monthly_label_matrix_v1_{args.universe}_1m_manifest.json"

    year_summary.to_csv(year_summary_path, index=False)
    monthly_ic.to_csv(monthly_ic_path, index=False)
    ic_summary.to_csv(ic_summary_path, index=False)

    split_counts = (
        matrix.groupby("split_block_static", dropna=False)
        .agg(
            rows=("issue_key", "size"),
            labeled_rows=("label_available", "sum"),
            model_eligible_rows=("is_model_eligible_1m", "sum"),
            unique_issues=("issue_key", "nunique"),
        )
        .reset_index()
        .to_dict("records")
    )

    label_stats = {
        "label_ret_1m": numeric_summary(matrix.loc[matrix["label_available"], "label_ret_1m"]),
        "last_issuer_date_residual_z_labeled": numeric_summary(matrix.loc[matrix["label_available"], "last_issuer_date_residual_z"]),
        "mean_issuer_date_residual_z_labeled": numeric_summary(matrix.loc[matrix["label_available"], "mean_issuer_date_residual_z"]),
        "signal_to_label_days_labeled": numeric_summary(matrix.loc[matrix["label_available"], "signal_to_label_days"]),
    }

    summary = {
        "ok": True,
        "run_id": run_id,
        "source_step07b_run_id": str(step07b.get("run_id")),
        "workspace": str(root),
        "universe": args.universe,
        "horizon": "1m",
        "rows": actual_rows,
        "labeled_rows": actual_labeled,
        "raw_return_rows": actual_raw_return_rows,
        "model_eligible_rows": int(matrix["is_model_eligible_1m"].sum()),
        "label_coverage_pct": round(100.0 * actual_labeled / actual_rows, 6) if actual_rows else None,
        "distinct_issues": int(matrix["issue_key"].nunique()),
        "distinct_labeled_issues": int(matrix.loc[matrix["label_available"], "issue_key"].nunique()),
        "date_min_signal": str(matrix["signal_month"].min().date()),
        "date_max_signal": str(matrix["signal_month"].max().date()),
        "date_min_label": str(matrix["label_month"].min().date()),
        "date_max_label": str(matrix["label_month"].max().date()),
        "split_counts": split_counts,
        "label_stats": label_stats,
        "ic_summary_rows": int(len(ic_summary)),
        "top_ic_features": ic_summary.head(10).to_dict("records") if not ic_summary.empty else [],
        "output_full_do_not_upload": str(full_path),
        "output_labeled_do_not_upload": str(labeled_path),
        "output_model_eligible_do_not_upload": str(eligible_path),
        "source_step07b_summary": str(step07b_summary_path),
        "source_step07b_year_summary": str(step07b_year_path),
        "year_summary": str(year_summary_path),
        "monthly_ic": str(monthly_ic_path),
        "ic_summary": str(ic_summary_path),
        "leakage_note": "Next-month labels use signal_month end to label_month end. 2025 is held out/incomplete for first baseline; rolling CV with one-month horizon embargo comes in modeling scripts.",
        "note": "Stable monthly label matrix. Local only. Upload bundle only, not parquet.",
    }

    write_json(summary_path, summary)
    write_json(manifest_path, summary)

    bundle = log_dir / f"step07c_monthly_label_matrix_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            summary_path,
            manifest_path,
            year_summary_path,
            monthly_ic_path,
            ic_summary_path,
            step07b_summary_path,
        ]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

