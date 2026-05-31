#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "years_to_maturity",
    "yld_pt",
    "rptd_pr",
    "entrd_vol_qt",
    "is_curve_ready",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_curve_ready_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"trace_fisd_panel_v1_{universe}_curve_ready_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing curve-ready manifest: {path}")

    df = pd.read_csv(path)
    df["curve_ready_rows"] = pd.to_numeric(df["curve_ready_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["curve_ready_rows"] > 0].copy()
    return df.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)


def choose_partitions(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    idx = sorted(set(round(i * (len(df) - 1) / (limit - 1)) for i in range(limit)))
    return df.iloc[idx].reset_index(drop=True)


def ns_factor_matrix(maturity: np.ndarray, tau: float) -> np.ndarray:
    x = maturity / tau
    slope = np.where(np.isclose(x, 0.0), 1.0, (1.0 - np.exp(-x)) / x)
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(maturity), slope, curvature])


def fit_nelson_siegel(
    maturity: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    tau_grid: np.ndarray,
) -> dict[str, Any] | None:
    valid = (
        np.isfinite(maturity)
        & np.isfinite(y)
        & np.isfinite(weights)
        & (maturity > 0)
        & (maturity <= 40)
        & (weights > 0)
    )

    maturity = maturity[valid]
    y = y[valid]
    weights = weights[valid]

    if len(y) < 4:
        return None

    best = None
    sqrt_w = np.sqrt(weights)

    for tau in tau_grid:
        x = ns_factor_matrix(maturity, float(tau))
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w

        try:
            beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue

        pred = x @ beta
        resid = y - pred
        rmse = float(np.sqrt(np.average(resid**2, weights=weights)))
        mae = float(np.average(np.abs(resid), weights=weights))

        rec = {
            "beta0": float(beta[0]),
            "beta1": float(beta[1]),
            "beta2": float(beta[2]),
            "tau": float(tau),
            "rmse": rmse,
            "mae": mae,
            "n_obs": int(len(y)),
        }

        if best is None or rec["rmse"] < best["rmse"]:
            best = rec

    return best


def read_partition_issue_date_agg(row: dict[str, Any], max_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(str(row["output_path"]))

    rec: dict[str, Any] = {
        "output_path": str(path),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "expected_curve_ready_rows": int(row.get("curve_ready_rows", 0) or 0),
        "rows_read": 0,
        "curve_ready_rows": 0,
        "usable_yield_rows": 0,
        "issue_date_rows": 0,
        "ok": False,
        "error": "",
    }

    try:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        selected = [c for c in READ_COLUMNS if c in available]

        if not selected:
            rec["error"] = "no_selected_columns"
            return pd.DataFrame(), rec

        table = pf.read(columns=selected)
        df = table.to_pandas()

        if max_rows > 0:
            df = df.head(max_rows).copy()

        rec["rows_read"] = int(len(df))

        for col in READ_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        ready = df["is_curve_ready"].astype(bool)
        x = df.loc[ready].copy()
        rec["curve_ready_rows"] = int(len(x))

        if x.empty:
            rec["ok"] = True
            return pd.DataFrame(), rec

        x["trd_exctn_dt"] = pd.to_datetime(x["trd_exctn_dt"], errors="coerce").dt.date
        x["issuer_id"] = pd.to_numeric(x["issuer_id"], errors="coerce")
        x["issue_id"] = pd.to_numeric(x["issue_id"], errors="coerce")
        x["years_to_maturity"] = pd.to_numeric(x["years_to_maturity"], errors="coerce")
        x["yld_pt"] = pd.to_numeric(x["yld_pt"], errors="coerce")
        x["rptd_pr"] = pd.to_numeric(x["rptd_pr"], errors="coerce")
        x["entrd_vol_qt"] = pd.to_numeric(x["entrd_vol_qt"], errors="coerce")

        # Conservative smoke bounds. We are only proving curve mechanics here.
        usable = (
            x["trd_exctn_dt"].notna()
            & x["issuer_id"].notna()
            & x["issue_id"].notna()
            & x["years_to_maturity"].between(0.05, 40.0)
            & x["yld_pt"].notna()
            & x["yld_pt"].between(-20.0, 100.0)
            & x["entrd_vol_qt"].notna()
            & (x["entrd_vol_qt"] > 0)
        )

        x = x.loc[usable, ["trd_exctn_dt", "issuer_id", "issue_id", "years_to_maturity", "yld_pt", "entrd_vol_qt"]].copy()
        rec["usable_yield_rows"] = int(len(x))

        if x.empty:
            rec["ok"] = True
            return pd.DataFrame(), rec

        x["dollar_weight"] = np.log1p(x["entrd_vol_qt"].clip(lower=0))
        x["wx_yld"] = x["yld_pt"] * x["dollar_weight"]
        x["wx_mat"] = x["years_to_maturity"] * x["dollar_weight"]

        grouped = (
            x.groupby(["trd_exctn_dt", "issuer_id", "issue_id"], dropna=True, sort=False)
            .agg(
                trade_count=("issue_id", "size"),
                weight_sum=("dollar_weight", "sum"),
                wx_yld=("wx_yld", "sum"),
                wx_mat=("wx_mat", "sum"),
                gross_volume=("entrd_vol_qt", "sum"),
            )
            .reset_index()
        )

        grouped["yld_pt_wavg"] = grouped["wx_yld"] / grouped["weight_sum"]
        grouped["years_to_maturity_wavg"] = grouped["wx_mat"] / grouped["weight_sum"]

        grouped = grouped.loc[
            grouped["yld_pt_wavg"].between(-20.0, 100.0)
            & grouped["years_to_maturity_wavg"].between(0.05, 40.0)
            & grouped["weight_sum"].gt(0)
        ].copy()

        rec["issue_date_rows"] = int(len(grouped))
        rec["ok"] = True

        return grouped, rec

    except Exception as exc:
        rec["error"] = repr(exc)
        return pd.DataFrame(), rec


def build_curve_groups(issue_date: pd.DataFrame, min_issues: int, max_curves: int) -> pd.DataFrame:
    if issue_date.empty:
        return pd.DataFrame()

    support = (
        issue_date.groupby(["trd_exctn_dt", "issuer_id"], dropna=True)
        .agg(
            n_issues=("issue_id", "nunique"),
            n_issue_date_rows=("issue_id", "size"),
            gross_volume=("gross_volume", "sum"),
            maturity_min=("years_to_maturity_wavg", "min"),
            maturity_max=("years_to_maturity_wavg", "max"),
            maturity_span=("years_to_maturity_wavg", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
    )

    eligible = support.loc[
        (support["n_issues"] >= min_issues)
        & (support["maturity_span"] >= 1.0)
    ].copy()

    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(
        ["trd_exctn_dt", "n_issues", "gross_volume"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    if max_curves > 0 and len(eligible) > max_curves:
        idx = sorted(set(round(i * (len(eligible) - 1) / (max_curves - 1)) for i in range(max_curves)))
        eligible = eligible.iloc[idx].reset_index(drop=True)

    return eligible


def fit_curve_group(args: tuple[int, dict[str, Any], pd.DataFrame, np.ndarray]) -> dict[str, Any]:
    fit_index, group_row, issue_date, tau_grid = args

    trade_date = group_row["trd_exctn_dt"]
    issuer_id = group_row["issuer_id"]

    g = issue_date.loc[
        (issue_date["trd_exctn_dt"] == trade_date)
        & (issue_date["issuer_id"] == issuer_id)
    ].copy()

    maturity = g["years_to_maturity_wavg"].to_numpy(dtype=float)
    y = g["yld_pt_wavg"].to_numpy(dtype=float)
    weights = g["weight_sum"].to_numpy(dtype=float)

    fit = fit_nelson_siegel(maturity, y, weights, tau_grid)

    rec = {
        "fit_index": int(fit_index),
        "trade_date": str(trade_date),
        "n_issues": int(group_row["n_issues"]),
        "n_issue_date_rows": int(group_row["n_issue_date_rows"]),
        "gross_volume": float(group_row["gross_volume"]),
        "maturity_min": float(group_row["maturity_min"]),
        "maturity_max": float(group_row["maturity_max"]),
        "maturity_span": float(group_row["maturity_span"]),
        "ok": fit is not None,
        "error": "",
    }

    if fit is not None:
        rec.update(fit)
    else:
        rec["error"] = "fit_returned_none"

    return rec


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest_all = load_curve_ready_manifest(root, args.universe)
    manifest = choose_partitions(manifest_all, args.limit_partitions)

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions={len(manifest)}")
    print(f"workers={args.workers}")

    partition_records = []
    agg_frames = []

    rows = manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(read_partition_issue_date_agg, row, args.max_rows_per_partition) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            agg, rec = fut.result()
            partition_records.append(rec)
            if not agg.empty:
                agg_frames.append(agg)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in partition_records if r.get("ok"))
                usable = sum(int(r.get("usable_yield_rows") or 0) for r in partition_records)
                issue_rows = sum(int(r.get("issue_date_rows") or 0) for r in partition_records)
                print(f"progress_read {i}/{len(futures)} ok={ok} usable_rows={usable:,} issue_date_rows={issue_rows:,}", flush=True)

    partition_detail = pd.DataFrame(partition_records).sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)
    issue_date = pd.concat(agg_frames, ignore_index=True) if agg_frames else pd.DataFrame()

    groups = build_curve_groups(issue_date, args.min_issues, args.max_curves)

    print(f"issue_date_rows={len(issue_date):,}")
    print(f"eligible_curve_groups={len(groups):,}")

    tau_grid = np.geomspace(0.25, 20.0, args.tau_grid_size)
    fit_records = []

    if not groups.empty:
        tasks = [(i, row.to_dict(), issue_date, tau_grid) for i, (_, row) in enumerate(groups.iterrows())]

        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(fit_curve_group, task) for task in tasks]

            for i, fut in enumerate(as_completed(futures), start=1):
                fit_records.append(fut.result())
                if i == 1 or i % args.progress_every == 0 or i == len(futures):
                    ok = sum(1 for r in fit_records if r.get("ok"))
                    print(f"progress_fit {i}/{len(futures)} ok={ok}", flush=True)

    fit_detail = pd.DataFrame(fit_records)
    ok_fits = fit_detail.loc[fit_detail["ok"] == True].copy() if not fit_detail.empty else pd.DataFrame()

    partition_detail_path = table_dir / "step06a_curve_fit_smoke_partition_detail.csv"
    fit_detail_path = table_dir / "step06a_curve_fit_smoke_fit_detail.csv"
    summary_path = table_dir / "step06a_curve_fit_smoke_summary.json"

    partition_detail.to_csv(partition_detail_path, index=False)
    fit_detail.to_csv(fit_detail_path, index=False)

    rmse_values = pd.to_numeric(ok_fits.get("rmse", pd.Series(dtype=float)), errors="coerce").dropna()

    summary = {
        "ok": bool(len(partition_detail) > 0 and partition_detail["ok"].astype(bool).all() and len(ok_fits) > 0),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "limit_partitions": int(args.limit_partitions),
        "partitions_scanned": int(len(partition_detail)),
        "partitions_ok": int(partition_detail["ok"].astype(bool).sum()) if not partition_detail.empty else 0,
        "partitions_failed": int((~partition_detail["ok"].astype(bool)).sum()) if not partition_detail.empty else 0,
        "rows_read": int(pd.to_numeric(partition_detail.get("rows_read", 0), errors="coerce").fillna(0).sum()) if not partition_detail.empty else 0,
        "curve_ready_rows_read": int(pd.to_numeric(partition_detail.get("curve_ready_rows", 0), errors="coerce").fillna(0).sum()) if not partition_detail.empty else 0,
        "usable_yield_rows": int(pd.to_numeric(partition_detail.get("usable_yield_rows", 0), errors="coerce").fillna(0).sum()) if not partition_detail.empty else 0,
        "issue_date_rows": int(len(issue_date)),
        "eligible_curve_groups": int(len(groups)),
        "curves_fit": int(len(ok_fits)),
        "curves_failed": int(len(fit_detail) - len(ok_fits)) if not fit_detail.empty else 0,
        "min_issues": int(args.min_issues),
        "max_curves": int(args.max_curves),
        "tau_grid_size": int(args.tau_grid_size),
        "usable_yield_pct_of_curve_ready": pct(
            int(pd.to_numeric(partition_detail.get("usable_yield_rows", 0), errors="coerce").fillna(0).sum()) if not partition_detail.empty else 0,
            int(pd.to_numeric(partition_detail.get("curve_ready_rows", 0), errors="coerce").fillna(0).sum()) if not partition_detail.empty else 0,
        ),
        "rmse_summary": {
            "count": int(len(rmse_values)),
            "mean": None if rmse_values.empty else round(float(rmse_values.mean()), 6),
            "median": None if rmse_values.empty else round(float(rmse_values.median()), 6),
            "p90": None if rmse_values.empty else round(float(rmse_values.quantile(0.90)), 6),
            "p99": None if rmse_values.empty else round(float(rmse_values.quantile(0.99)), 6),
        },
        "partition_detail_path": str(partition_detail_path),
        "fit_detail_path": str(fit_detail_path),
        "note": "Local-only Nelson-Siegel smoke. Fit details omit issuer IDs and issue IDs.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step06a_curve_fit_smoke_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, partition_detail_path, fit_detail_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"PARTITION_DETAIL={partition_detail_path}")
    print(f"FIT_DETAIL={fit_detail_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 06A Nelson-Siegel issuer-date curve smoke. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=30)
    p.add_argument("--max-rows-per-partition", type=int, default=1_000_000)
    p.add_argument("--min-issues", type=int, default=4)
    p.add_argument("--max-curves", type=int, default=2000)
    p.add_argument("--tau-grid-size", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=25)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
