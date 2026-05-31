#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


READ_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "model",
    "n_issues",
    "trade_count",
    "gross_volume",
    "weight_sum",
    "years_to_maturity_wavg",
    "yld_pt_wavg",
    "fitted_yld_pt",
    "residual_yld_pt",
    "rptd_pr_wavg",
    "curve_rmse",
    "curve_mae",
    "curve_score",
    "curve_unstable",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_fit_v1_{universe}_validated_nonempty_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing validated curve-fit manifest: {path}")

    df = pd.read_csv(path)
    df["residual_rows"] = pd.to_numeric(df["residual_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["residual_rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "residuals_output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def maturity_bucket(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    return pd.cut(
        y,
        bins=[0, 1, 3, 5, 7, 10, 20, 30, 40],
        labels=["0_1y", "1_3y", "3_5y", "5_7y", "7_10y", "10_20y", "20_30y", "30_40y"],
        include_lowest=False,
    ).astype("string")


def safe_zscore(s: pd.Series, min_count: int = 4) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=1)

    if x.notna().sum() < min_count or not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    return (x - mu) / sd


def residual_percentile(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    return x.rank(method="average", pct=True)


def read_residual_partition(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in READ_COLUMNS if c in available]

    if not selected:
        return pd.DataFrame(columns=READ_COLUMNS)

    df = pf.read(columns=selected).to_pandas()

    for col in READ_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df.loc[:, READ_COLUMNS].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["trd_exctn_dt"] = pd.to_datetime(x["trd_exctn_dt"], errors="coerce")
    x["issuer_id"] = pd.to_numeric(x["issuer_id"], errors="coerce")
    x["issue_id"] = pd.to_numeric(x["issue_id"], errors="coerce")

    for col in [
        "n_issues",
        "trade_count",
        "gross_volume",
        "weight_sum",
        "years_to_maturity_wavg",
        "yld_pt_wavg",
        "fitted_yld_pt",
        "residual_yld_pt",
        "rptd_pr_wavg",
        "curve_rmse",
        "curve_mae",
        "curve_score",
    ]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x["model"] = x["model"].astype("string")
    x["curve_unstable"] = x["curve_unstable"].fillna(False).astype(bool)

    x["abs_residual_yld_pt"] = x["residual_yld_pt"].abs()
    x["residual_over_rmse"] = x["residual_yld_pt"] / x["curve_rmse"].replace(0, np.nan)
    x["abs_residual_over_rmse"] = x["residual_over_rmse"].abs()

    x["maturity_bucket"] = maturity_bucket(x["years_to_maturity_wavg"])

    # Issuer-date standardization: primary cheap/rich signal.
    group_key = ["trd_exctn_dt", "issuer_id"]
    x["issuer_date_residual_z"] = x.groupby(group_key, sort=False)["residual_yld_pt"].transform(safe_zscore)
    x["issuer_date_abs_residual_z"] = x["issuer_date_residual_z"].abs()
    x["issuer_date_residual_pctile"] = x.groupby(group_key, sort=False)["residual_yld_pt"].transform(residual_percentile)

    # Maturity-bucket standardization inside issuer-date when enough bonds exist.
    bucket_key = ["trd_exctn_dt", "issuer_id", "maturity_bucket"]
    x["issuer_date_bucket_residual_z"] = x.groupby(bucket_key, sort=False)["residual_yld_pt"].transform(safe_zscore)
    x["issuer_date_bucket_abs_residual_z"] = x["issuer_date_bucket_residual_z"].abs()

    x["is_cheap_vs_curve"] = x["residual_yld_pt"] > 0
    x["is_rich_vs_curve"] = x["residual_yld_pt"] < 0

    x["log_gross_volume"] = np.log1p(x["gross_volume"].clip(lower=0))
    x["log_weight_sum"] = np.log1p(x["weight_sum"].clip(lower=0))
    x["log_trade_count"] = np.log1p(x["trade_count"].clip(lower=0))

    x["liquidity_weight"] = x["log_gross_volume"] * x["log_trade_count"]
    x["curve_support_score"] = np.log1p(x["n_issues"].clip(lower=0)) / (1.0 + x["curve_rmse"].clip(lower=0))

    out_cols = [
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

    return x.loc[:, out_cols].sort_values(["trd_exctn_dt", "issuer_id", "issue_id"], kind="mergesort").reset_index(drop=True)


def output_path(root: Path, run_id: str, universe: str, row: dict[str, Any], stable: bool) -> Path:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")

    if stable:
        base = root / "data" / "processed" / "curve_residual_features_v1" / f"universe={universe}"
    else:
        base = root / "data" / "interim" / "curve_residual_features_smoke_v1" / f"run_id={run_id}" / f"universe={universe}"

    return base / f"trd_exctn_dt={start}_to_{end}" / "part.parquet"


def process_one(root_str: str, run_id: str, universe: str, row: dict[str, Any], stable: bool, overwrite: bool) -> dict[str, Any]:
    root = Path(root_str)
    residual_path = Path(str(row["residuals_output_path"]))
    out_path = output_path(root, run_id, universe, row, stable=stable)

    rec: dict[str, Any] = {
        "universe": universe,
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "residuals_input_path": str(residual_path),
        "output_path": str(out_path),
        "expected_residual_rows": int(row.get("residual_rows", 0) or 0),
        "rows_read": 0,
        "rows_written": 0,
        "non_null_issuer_date_z": 0,
        "non_null_bucket_z": 0,
        "cheap_rows": 0,
        "rich_rows": 0,
        "unstable_curve_rows": 0,
        "skipped_existing": False,
        "ok": False,
        "error": "",
    }

    try:
        if out_path.exists() and not overwrite:
            pf = pq.ParquetFile(out_path)
            rec["rows_written"] = int(pf.metadata.num_rows)
            rec["skipped_existing"] = True
            rec["ok"] = True
            return rec

        residuals = read_residual_partition(residual_path)
        rec["rows_read"] = int(len(residuals))

        features = build_features(residuals)
        rec["rows_written"] = int(len(features))
        rec["non_null_issuer_date_z"] = int(features["issuer_date_residual_z"].notna().sum())
        rec["non_null_bucket_z"] = int(features["issuer_date_bucket_residual_z"].notna().sum())
        rec["cheap_rows"] = int(features["is_cheap_vs_curve"].sum())
        rec["rich_rows"] = int(features["is_rich_vs_curve"].sum())
        rec["unstable_curve_rows"] = int(features["curve_unstable"].sum())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        if tmp.exists():
            tmp.unlink()
        features.to_parquet(tmp, index=False, compression="zstd")
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

    manifest_all = load_manifest(root, args.universe)
    manifest = choose_partitions(manifest_all, args.limit_partitions)
    stable = args.limit_partitions <= 0 and args.stable_output

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions={len(rows)}")
    print(f"stable_output={stable}")
    print(f"workers={args.workers}")

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(process_one, str(root), run_id, args.universe, row, stable, args.overwrite)
            for row in rows
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                rows_written = sum(int(r.get("rows_written") or 0) for r in results)
                z_rows = sum(int(r.get("non_null_issuer_date_z") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} failed={failed} rows_written={rows_written:,} z_rows={z_rows:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)
    ok_detail = detail.loc[detail["ok"] == True].copy()

    detail_path = table_dir / "step06g_residual_feature_partition_summary.csv"
    summary_path = table_dir / "step06g_residual_feature_summary.json"

    detail.to_csv(detail_path, index=False)

    feature_manifest = pd.DataFrame(
        {
            "universe": ok_detail["universe"] if not ok_detail.empty else pd.Series(dtype=str),
            "start_date": ok_detail["start_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "end_date": ok_detail["end_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "rows": pd.to_numeric(ok_detail["rows_written"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "non_null_issuer_date_z": pd.to_numeric(ok_detail["non_null_issuer_date_z"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "output_path": ok_detail["output_path"] if not ok_detail.empty else pd.Series(dtype=str),
            "residuals_input_path": ok_detail["residuals_input_path"] if not ok_detail.empty else pd.Series(dtype=str),
        }
    ).sort_values(["start_date", "end_date", "output_path"])

    if stable:
        manifest_path = manifest_dir / f"curve_residual_features_v1_{args.universe}_manifest.csv"
    else:
        manifest_path = manifest_dir / f"curve_residual_features_v1_{args.universe}_smoke_manifest.csv"

    nonempty_manifest_path = manifest_path.with_name(manifest_path.stem.replace("_manifest", "_nonempty_manifest") + ".csv")

    feature_manifest.to_csv(manifest_path, index=False)
    feature_manifest.loc[feature_manifest["rows"] > 0].to_csv(nonempty_manifest_path, index=False)

    rows_written = int_sum(ok_detail, "rows_written")
    expected_rows = int(pd.to_numeric(manifest["residual_rows"], errors="coerce").fillna(0).sum())

    summary = {
        "ok": bool(len(detail) > 0 and detail["ok"].astype(bool).all() and rows_written == expected_rows),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "stable_output": bool(stable),
        "limit_partitions": int(args.limit_partitions),
        "partitions_requested": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()) if not detail.empty else 0,
        "partitions_failed": int((~detail["ok"].astype(bool)).sum()) if not detail.empty else 0,
        "expected_residual_rows": expected_rows,
        "rows_written": rows_written,
        "rows_written_minus_expected": int(rows_written - expected_rows),
        "non_null_issuer_date_z": int_sum(ok_detail, "non_null_issuer_date_z"),
        "non_null_bucket_z": int_sum(ok_detail, "non_null_bucket_z"),
        "cheap_rows": int_sum(ok_detail, "cheap_rows"),
        "rich_rows": int_sum(ok_detail, "rich_rows"),
        "unstable_curve_rows": int_sum(ok_detail, "unstable_curve_rows"),
        "skipped_existing": int(ok_detail["skipped_existing"].fillna(False).astype(bool).sum()) if "skipped_existing" in ok_detail else 0,
        "detail_path": str(detail_path),
        "manifest": str(manifest_path),
        "nonempty_manifest": str(nonempty_manifest_path),
        "output_root_do_not_upload": str(root / ("data/processed/curve_residual_features_v1" if stable else f"data/interim/curve_residual_features_smoke_v1/run_id={run_id}")),
        "note": "Local-only residual feature construction. Upload bundle only, not feature parquet.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step06g_residual_feature_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, detail_path, manifest_path, nonempty_manifest_path]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"DETAIL={detail_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 06G residual feature smoke/scale. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--stable-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

