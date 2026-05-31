#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


INPUT_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "issue_id",
    "trade_count",
    "gross_volume",
    "weight_sum",
    "yld_pt_wavg",
    "years_to_maturity_wavg",
    "rptd_pr_wavg",
    "min_years_to_maturity",
    "max_years_to_maturity",
    "min_yld_pt",
    "max_yld_pt",
]

PARAM_COLUMNS = [
    "trd_exctn_dt",
    "issuer_id",
    "model",
    "n_issues",
    "n_issue_date_rows",
    "gross_volume",
    "maturity_min",
    "maturity_max",
    "maturity_span",
    "rmse",
    "mae",
    "score",
    "unstable",
    "beta0",
    "beta1",
    "beta2",
    "tau",
    "ridge_alpha",
    "maturity_center",
    "maturity_scale",
    "fit_warning",
]

RESIDUAL_COLUMNS = [
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


def load_validated_manifest(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_inputs_v1_{universe}_validated_nonempty_manifest.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing validated curve-input manifest: {path}")

    df = pd.read_csv(path)
    df["issue_date_rows"] = pd.to_numeric(df["issue_date_rows"], errors="coerce").fillna(0).astype("int64")
    df = df.loc[df["issue_date_rows"] > 0].copy()
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


def weighted_lstsq(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ridge_alpha: float = 0.0,
    penalize_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    sqrt_w = np.sqrt(w)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w

    if ridge_alpha > 0:
        p = np.eye(x.shape[1]) * math.sqrt(ridge_alpha)
        if not penalize_intercept:
            p[0, 0] = 0.0
        xw = np.vstack([xw, p])
        yw = np.concatenate([yw, np.zeros(x.shape[1])])

    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    pred = x @ beta
    resid = y - pred
    rmse = float(np.sqrt(np.average(resid**2, weights=w)))
    mae = float(np.average(np.abs(resid), weights=w))
    max_abs_beta = float(np.max(np.abs(beta))) if len(beta) else 0.0
    return beta, pred, rmse, mae, max_abs_beta


def clean_group_arrays(g: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x = g.copy()
    x["years_to_maturity_wavg"] = pd.to_numeric(x["years_to_maturity_wavg"], errors="coerce")
    x["yld_pt_wavg"] = pd.to_numeric(x["yld_pt_wavg"], errors="coerce")
    x["weight_sum"] = pd.to_numeric(x["weight_sum"], errors="coerce")

    valid = (
        x["years_to_maturity_wavg"].between(0.05, 40.0)
        & x["yld_pt_wavg"].between(-20.0, 100.0)
        & x["weight_sum"].gt(0)
    )

    x = x.loc[valid].copy()
    maturity = x["years_to_maturity_wavg"].to_numpy(dtype=float)
    y = x["yld_pt_wavg"].to_numpy(dtype=float)
    w = x["weight_sum"].to_numpy(dtype=float)
    return x, maturity, y, w


def fit_linear(g: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray] | None:
    xdf, maturity, y, w = clean_group_arrays(g)
    if len(y) < 3:
        return None

    center = float(np.average(maturity, weights=w))
    scale = float(np.std(maturity))
    if not np.isfinite(scale) or scale <= 1e-9:
        return None

    z = (maturity - center) / scale
    x = np.column_stack([np.ones_like(z), z])
    beta, pred, rmse, mae, max_abs_beta = weighted_lstsq(x, y, w)

    rec = {
        "model": "linear",
        "rmse": rmse,
        "mae": mae,
        "score": rmse + 0.002,
        "unstable": False,
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": None,
        "tau": None,
        "ridge_alpha": 0.0,
        "maturity_center": center,
        "maturity_scale": scale,
        "max_abs_beta": max_abs_beta,
        "fit_warning": "",
    }
    return rec, pred


def fit_quadratic(g: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray] | None:
    xdf, maturity, y, w = clean_group_arrays(g)
    if len(y) < 5:
        return None

    center = float(np.average(maturity, weights=w))
    scale = float(np.std(maturity))
    if not np.isfinite(scale) or scale <= 1e-9:
        return None

    z = (maturity - center) / scale
    x = np.column_stack([np.ones_like(z), z, z**2])
    beta, pred, rmse, mae, max_abs_beta = weighted_lstsq(x, y, w, ridge_alpha=0.01)

    unstable = bool(max_abs_beta > 100.0)
    rec = {
        "model": "quadratic",
        "rmse": rmse,
        "mae": mae,
        "score": rmse + 0.003 + (0.05 if unstable else 0.0),
        "unstable": unstable,
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": float(beta[2]),
        "tau": None,
        "ridge_alpha": 0.01,
        "maturity_center": center,
        "maturity_scale": scale,
        "max_abs_beta": max_abs_beta,
        "fit_warning": "large_beta" if unstable else "",
    }
    return rec, pred


def fit_ns_ridge(
    g: pd.DataFrame,
    tau_grid: np.ndarray,
    alpha_grid: list[float],
) -> tuple[dict[str, Any], np.ndarray] | None:
    xdf, maturity, y, w = clean_group_arrays(g)
    if len(y) < 6:
        return None

    best: tuple[dict[str, Any], np.ndarray] | None = None

    for alpha in alpha_grid:
        for tau in tau_grid:
            x = ns_factor_matrix(maturity, float(tau))
            try:
                beta, pred, rmse, mae, max_abs_beta = weighted_lstsq(
                    x,
                    y,
                    w,
                    ridge_alpha=float(alpha),
                    penalize_intercept=False,
                )
            except np.linalg.LinAlgError:
                continue

            tau_boundary = bool(np.isclose(tau, tau_grid.min()) or np.isclose(tau, tau_grid.max()))
            unstable = bool(max_abs_beta > 100.0 or tau_boundary)
            beta_norm = float(np.sqrt(np.sum(beta[1:] ** 2)))

            score = rmse + 0.0005 * beta_norm + 0.01 * math.log1p(float(alpha))
            if tau_boundary:
                score += 0.025
            if max_abs_beta > 100.0:
                score += 0.10
            if max_abs_beta > 500.0:
                score += 0.50

            warning = []
            if tau_boundary:
                warning.append("tau_boundary")
            if max_abs_beta > 100.0:
                warning.append("large_beta")

            rec = {
                "model": "ns_ridge",
                "rmse": float(rmse),
                "mae": float(mae),
                "score": float(score),
                "unstable": unstable,
                "beta0": float(beta[0]),
                "beta1": float(beta[1]),
                "beta2": float(beta[2]),
                "tau": float(tau),
                "ridge_alpha": float(alpha),
                "maturity_center": None,
                "maturity_scale": None,
                "max_abs_beta": max_abs_beta,
                "fit_warning": ";".join(warning),
            }

            if best is None or rec["score"] < best[0]["score"]:
                best = (rec, pred)

    return best


def fit_best_curve(
    g: pd.DataFrame,
    tau_grid: np.ndarray,
    alpha_grid: list[float],
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    xdf, maturity, y, w = clean_group_arrays(g)
    if len(y) < 3:
        return None, pd.DataFrame()

    # Preserve the cleaned group order so predictions align.
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []

    lin = fit_linear(xdf)
    if lin is not None:
        candidates.append(lin)

    quad = fit_quadratic(xdf)
    if quad is not None:
        candidates.append(quad)

    # Try NS only when enough observations and maturity span justify it.
    maturity_span = float(np.nanmax(maturity) - np.nanmin(maturity)) if len(maturity) else 0.0
    if len(y) >= 6 and maturity_span >= 2.0:
        ns = fit_ns_ridge(xdf, tau_grid=tau_grid, alpha_grid=alpha_grid)
        if ns is not None:
            candidates.append(ns)

    if not candidates:
        return None, pd.DataFrame()

    best_rec, best_pred = min(candidates, key=lambda item: item[0]["score"])

    residuals = xdf.copy()
    residuals["fitted_yld_pt"] = best_pred
    residuals["residual_yld_pt"] = residuals["yld_pt_wavg"].to_numpy(dtype=float) - best_pred

    return best_rec, residuals


def read_curve_input(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in INPUT_COLUMNS if c in available]
    if not selected:
        return pd.DataFrame(columns=INPUT_COLUMNS)

    df = pf.read(columns=selected).to_pandas()
    for c in INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df.loc[:, INPUT_COLUMNS].copy()


def select_support_groups(df: pd.DataFrame, min_issues: int, max_curves_per_partition: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    support = (
        df.groupby(["trd_exctn_dt", "issuer_id"], dropna=True)
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

    support = support.loc[
        (support["n_issues"] >= min_issues)
        & (support["maturity_span"] >= 1.0)
    ].copy()

    if support.empty:
        return support

    support = support.sort_values(
        ["trd_exctn_dt", "n_issues", "gross_volume"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    if max_curves_per_partition > 0 and len(support) > max_curves_per_partition:
        support = support.head(max_curves_per_partition).copy()

    return support.reset_index(drop=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return

    tmp = path.with_suffix(f".tmp.{os.getpid()}.parquet")
    if tmp.exists():
        tmp.unlink()

    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def output_paths(root: Path, universe: str, row: dict[str, Any], run_id: str, stable: bool) -> tuple[Path, Path]:
    start = str(row.get("start_date", "")).replace("-", "")
    end = str(row.get("end_date", "")).replace("-", "")

    if stable:
        base = root / "data" / "processed" / "curve_fit_v1" / f"universe={universe}"
    else:
        base = root / "data" / "interim" / "curve_fit_smoke_v1" / f"run_id={run_id}" / f"universe={universe}"

    part = f"trd_exctn_dt={start}_to_{end}"
    return base / "curve_params" / part / "part.parquet", base / "curve_residuals" / part / "part.parquet"


def process_partition(
    root_str: str,
    universe: str,
    row: dict[str, Any],
    run_id: str,
    stable: bool,
    overwrite: bool,
    min_issues: int,
    max_curves_per_partition: int,
    tau_grid: list[float],
    alpha_grid: list[float],
) -> dict[str, Any]:
    t0 = time.time()

    root = Path(root_str)
    input_path = Path(str(row["output_path"]))
    param_path, residual_path = output_paths(root, universe, row, run_id, stable=stable)

    rec: dict[str, Any] = {
        "universe": universe,
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "input_path": str(input_path),
        "params_output_path": str(param_path),
        "residuals_output_path": str(residual_path),
        "input_issue_date_rows": int(row.get("issue_date_rows", 0) or 0),
        "eligible_curve_groups": 0,
        "curves_fit": 0,
        "curves_failed": 0,
        "residual_rows": 0,
        "model_linear": 0,
        "model_quadratic": 0,
        "model_ns_ridge": 0,
        "unstable_curves": 0,
        "median_rmse": None,
        "p90_rmse": None,
        "elapsed_sec": None,
        "skipped_existing": False,
        "ok": False,
        "error": "",
    }

    try:
        if param_path.exists() and residual_path.exists() and not overwrite:
            rec["curves_fit"] = pq.ParquetFile(param_path).metadata.num_rows
            rec["residual_rows"] = pq.ParquetFile(residual_path).metadata.num_rows
            rec["skipped_existing"] = True
            rec["ok"] = True
            rec["elapsed_sec"] = round(time.time() - t0, 3)
            return rec

        df = read_curve_input(input_path)
        groups = select_support_groups(df, min_issues=min_issues, max_curves_per_partition=max_curves_per_partition)
        rec["eligible_curve_groups"] = int(len(groups))

        if groups.empty:
            write_parquet_atomic(pd.DataFrame(columns=PARAM_COLUMNS), param_path, overwrite=True)
            write_parquet_atomic(pd.DataFrame(columns=RESIDUAL_COLUMNS), residual_path, overwrite=True)
            rec["ok"] = True
            rec["elapsed_sec"] = round(time.time() - t0, 3)
            return rec

        tau_arr = np.array(tau_grid, dtype=float)
        alpha_vals = [float(x) for x in alpha_grid]

        params = []
        residual_frames = []

        grouped = {
            key: group.copy()
            for key, group in df.groupby(["trd_exctn_dt", "issuer_id"], dropna=True, sort=False)
        }

        for _, support_row in groups.iterrows():
            key = (support_row["trd_exctn_dt"], support_row["issuer_id"])
            g = grouped.get(key)

            if g is None or g.empty:
                rec["curves_failed"] += 1
                continue

            fit, resid = fit_best_curve(g, tau_grid=tau_arr, alpha_grid=alpha_vals)

            if fit is None or resid.empty:
                rec["curves_failed"] += 1
                continue

            fit_row = {
                "trd_exctn_dt": support_row["trd_exctn_dt"],
                "issuer_id": support_row["issuer_id"],
                "n_issues": int(support_row["n_issues"]),
                "n_issue_date_rows": int(support_row["n_issue_date_rows"]),
                "gross_volume": float(support_row["gross_volume"]),
                "maturity_min": float(support_row["maturity_min"]),
                "maturity_max": float(support_row["maturity_max"]),
                "maturity_span": float(support_row["maturity_span"]),
                **fit,
            }
            params.append(fit_row)

            resid_out = resid.loc[
                :,
                [
                    "trd_exctn_dt",
                    "issuer_id",
                    "issue_id",
                    "trade_count",
                    "gross_volume",
                    "weight_sum",
                    "years_to_maturity_wavg",
                    "yld_pt_wavg",
                    "fitted_yld_pt",
                    "residual_yld_pt",
                    "rptd_pr_wavg",
                ],
            ].copy()

            resid_out["model"] = fit["model"]
            resid_out["n_issues"] = int(support_row["n_issues"])
            resid_out["curve_rmse"] = fit["rmse"]
            resid_out["curve_mae"] = fit["mae"]
            resid_out["curve_score"] = fit["score"]
            resid_out["curve_unstable"] = bool(fit["unstable"])
            residual_frames.append(resid_out.loc[:, RESIDUAL_COLUMNS])

        params_df = pd.DataFrame(params)
        residuals_df = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame(columns=RESIDUAL_COLUMNS)

        for c in PARAM_COLUMNS:
            if c not in params_df.columns:
                params_df[c] = pd.NA
        params_df = params_df.loc[:, PARAM_COLUMNS].sort_values(["trd_exctn_dt", "issuer_id"], kind="mergesort")

        for c in RESIDUAL_COLUMNS:
            if c not in residuals_df.columns:
                residuals_df[c] = pd.NA
        residuals_df = residuals_df.loc[:, RESIDUAL_COLUMNS].sort_values(
            ["trd_exctn_dt", "issuer_id", "issue_id"],
            kind="mergesort",
        )

        write_parquet_atomic(params_df, param_path, overwrite=True)
        write_parquet_atomic(residuals_df, residual_path, overwrite=True)

        rec["curves_fit"] = int(len(params_df))
        rec["residual_rows"] = int(len(residuals_df))
        rec["model_linear"] = int((params_df["model"] == "linear").sum()) if not params_df.empty else 0
        rec["model_quadratic"] = int((params_df["model"] == "quadratic").sum()) if not params_df.empty else 0
        rec["model_ns_ridge"] = int((params_df["model"] == "ns_ridge").sum()) if not params_df.empty else 0
        rec["unstable_curves"] = int(params_df["unstable"].astype(bool).sum()) if not params_df.empty else 0

        rmse = pd.to_numeric(params_df["rmse"], errors="coerce").dropna() if not params_df.empty else pd.Series(dtype=float)
        if not rmse.empty:
            rec["median_rmse"] = round(float(rmse.median()), 6)
            rec["p90_rmse"] = round(float(rmse.quantile(0.90)), 6)

        rec["ok"] = True
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec

    except Exception as exc:
        rec["error"] = repr(exc)
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec


def int_sum(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest_all = load_validated_manifest(root, args.universe)
    manifest = choose_partitions(manifest_all, args.limit_partitions)

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    tau_grid = np.geomspace(args.tau_min, args.tau_max, args.tau_grid_size).tolist()
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    stable = args.limit_partitions <= 0 and args.stable_output

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions={len(rows)}")
    print(f"stable_output={stable}")
    print(f"workers={args.workers}")
    print(f"max_curves_per_partition={args.max_curves_per_partition}")

    results = []

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                process_partition,
                str(root),
                args.universe,
                row,
                run_id,
                stable,
                args.overwrite,
                args.min_issues,
                args.max_curves_per_partition,
                tau_grid,
                alpha_grid,
            )
            for row in rows
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                failed = len(results) - ok
                curves = sum(int(r.get("curves_fit") or 0) for r in results)
                residuals = sum(int(r.get("residual_rows") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} failed={failed} curves={curves:,} residual_rows={residuals:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "params_output_path"]).reset_index(drop=True)
    ok_detail = detail.loc[detail["ok"] == True].copy()

    detail_path = table_dir / "step06e_curve_fit_residuals_partition_summary.csv"
    summary_path = table_dir / "step06e_curve_fit_residuals_summary.json"

    detail.to_csv(detail_path, index=False)

    curve_manifest = pd.DataFrame(
        {
            "universe": ok_detail["universe"] if not ok_detail.empty else pd.Series(dtype=str),
            "start_date": ok_detail["start_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "end_date": ok_detail["end_date"] if not ok_detail.empty else pd.Series(dtype=str),
            "curves_fit": pd.to_numeric(ok_detail["curves_fit"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "residual_rows": pd.to_numeric(ok_detail["residual_rows"], errors="coerce").fillna(0).astype("int64") if not ok_detail.empty else pd.Series(dtype="int64"),
            "params_output_path": ok_detail["params_output_path"] if not ok_detail.empty else pd.Series(dtype=str),
            "residuals_output_path": ok_detail["residuals_output_path"] if not ok_detail.empty else pd.Series(dtype=str),
        }
    ).sort_values(["start_date", "end_date", "params_output_path"])

    if stable:
        manifest_path = manifest_dir / f"curve_fit_v1_{args.universe}_manifest.csv"
    else:
        manifest_path = manifest_dir / f"curve_fit_v1_{args.universe}_smoke_manifest.csv"

    nonempty_manifest_path = manifest_path.with_name(manifest_path.stem.replace("_manifest", "_nonempty_manifest") + ".csv")

    curve_manifest.to_csv(manifest_path, index=False)
    curve_manifest.loc[curve_manifest["curves_fit"] > 0].to_csv(nonempty_manifest_path, index=False)

    total_curves = int_sum(ok_detail, "curves_fit")
    total_residuals = int_sum(ok_detail, "residual_rows")
    unstable = int_sum(ok_detail, "unstable_curves")

    summary = {
        "ok": bool(len(detail) > 0 and detail["ok"].astype(bool).all() and total_curves > 0),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "stable_output": bool(stable),
        "limit_partitions": int(args.limit_partitions),
        "partitions_requested": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()) if not detail.empty else 0,
        "partitions_failed": int((~detail["ok"].astype(bool)).sum()) if not detail.empty else 0,
        "input_issue_date_rows": int_sum(ok_detail, "input_issue_date_rows"),
        "eligible_curve_groups": int_sum(ok_detail, "eligible_curve_groups"),
        "curves_fit": total_curves,
        "curves_failed": int_sum(ok_detail, "curves_failed"),
        "residual_rows": total_residuals,
        "model_linear": int_sum(ok_detail, "model_linear"),
        "model_quadratic": int_sum(ok_detail, "model_quadratic"),
        "model_ns_ridge": int_sum(ok_detail, "model_ns_ridge"),
        "unstable_curves": unstable,
        "unstable_curve_pct": None if total_curves == 0 else round(100.0 * unstable / total_curves, 6),
        "median_partition_rmse_median": None if ok_detail.empty else round(float(pd.to_numeric(ok_detail["median_rmse"], errors="coerce").dropna().median()), 6),
        "p90_partition_rmse_median": None if ok_detail.empty else round(float(pd.to_numeric(ok_detail["p90_rmse"], errors="coerce").dropna().median()), 6),
        "skipped_existing": int(ok_detail["skipped_existing"].fillna(False).astype(bool).sum()) if "skipped_existing" in ok_detail else 0,
        "min_issues": int(args.min_issues),
        "tau_grid_size": int(args.tau_grid_size),
        "alpha_grid": alpha_grid,
        "max_curves_per_partition": int(args.max_curves_per_partition),
        "detail_path": str(detail_path),
        "manifest": str(manifest_path),
        "nonempty_manifest": str(nonempty_manifest_path),
        "output_root_do_not_upload": str(root / ("data/processed/curve_fit_v1" if stable else f"data/interim/curve_fit_smoke_v1/run_id={run_id}")),
        "note": "Local-only guarded curve fit and residual smoke/scale. Upload bundle only, not curve parquet.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step06e_curve_fit_residuals_bundle_{run_id}.tar.gz"
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
    p = argparse.ArgumentParser(description="Step 06E guarded curve fitting and residuals. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--min-issues", type=int, default=4)
    p.add_argument("--max-curves-per-partition", type=int, default=300)
    p.add_argument("--tau-grid-size", type=int, default=25)
    p.add_argument("--tau-min", type=float, default=0.50)
    p.add_argument("--tau-max", type=float, default=15.0)
    p.add_argument("--alpha-grid", default="0.1,1.0,10.0,100.0")
    p.add_argument("--stable-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
