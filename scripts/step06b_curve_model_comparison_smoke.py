#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_step06a_module(root: Path):
    path = root / "scripts" / "step06a_curve_fit_smoke.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 06A script: {path}")

    spec = importlib.util.spec_from_file_location("step06a_curve_fit_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weighted_fit(
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


def clean_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maturity = pd.to_numeric(group["years_to_maturity_wavg"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group["yld_pt_wavg"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(group["weight_sum"], errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(maturity)
        & np.isfinite(y)
        & np.isfinite(w)
        & (maturity > 0.05)
        & (maturity <= 40.0)
        & (y > -20.0)
        & (y < 100.0)
        & (w > 0)
    )

    return maturity[valid], y[valid], w[valid]


def fit_linear(group: pd.DataFrame) -> dict[str, Any] | None:
    maturity, y, w = clean_arrays(group)
    if len(y) < 3:
        return None

    center = float(np.average(maturity, weights=w))
    scale = float(np.nanstd(maturity))
    if not np.isfinite(scale) or scale <= 1e-9:
        return None

    z = (maturity - center) / scale
    x = np.column_stack([np.ones_like(z), z])

    beta, pred, rmse, mae, max_abs_beta = weighted_fit(x, y, w)

    return {
        "model": "linear",
        "ok": True,
        "n_obs": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "max_abs_beta": max_abs_beta,
        "complexity": 2,
        "maturity_center": center,
        "maturity_scale": scale,
        "tau": None,
        "ridge_alpha": 0.0,
        "score": rmse + 0.002,
        "unstable": False,
    }


def fit_quadratic(group: pd.DataFrame) -> dict[str, Any] | None:
    maturity, y, w = clean_arrays(group)
    if len(y) < 5:
        return None

    center = float(np.average(maturity, weights=w))
    scale = float(np.nanstd(maturity))
    if not np.isfinite(scale) or scale <= 1e-9:
        return None

    z = (maturity - center) / scale
    x = np.column_stack([np.ones_like(z), z, z**2])

    beta, pred, rmse, mae, max_abs_beta = weighted_fit(x, y, w, ridge_alpha=0.01)

    unstable = bool(max_abs_beta > 100.0)

    return {
        "model": "quadratic",
        "ok": True,
        "n_obs": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "max_abs_beta": max_abs_beta,
        "complexity": 3,
        "maturity_center": center,
        "maturity_scale": scale,
        "tau": None,
        "ridge_alpha": 0.01,
        "score": rmse + 0.003 + (0.05 if unstable else 0.0),
        "unstable": unstable,
    }


def ns_factor_matrix(maturity: np.ndarray, tau: float) -> np.ndarray:
    x = maturity / tau
    slope = np.where(np.isclose(x, 0.0), 1.0, (1.0 - np.exp(-x)) / x)
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(maturity), slope, curvature])


def fit_ns_ridge(group: pd.DataFrame, tau_grid: np.ndarray, alpha_grid: list[float]) -> dict[str, Any] | None:
    maturity, y, w = clean_arrays(group)
    if len(y) < 6:
        return None

    best: dict[str, Any] | None = None

    for alpha in alpha_grid:
        for tau in tau_grid:
            x = ns_factor_matrix(maturity, float(tau))

            try:
                beta, pred, rmse, mae, max_abs_beta = weighted_fit(
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

            # Score deliberately prefers stable fits over tiny RMSE gains from pathological betas.
            beta_norm = float(np.sqrt(np.sum(beta[1:] ** 2)))
            score = rmse + 0.0005 * beta_norm + 0.01 * math.log1p(alpha)
            if tau_boundary:
                score += 0.025
            if max_abs_beta > 100.0:
                score += 0.10
            if max_abs_beta > 500.0:
                score += 0.50

            rec = {
                "model": "ns_ridge",
                "ok": True,
                "n_obs": int(len(y)),
                "rmse": rmse,
                "mae": mae,
                "max_abs_beta": max_abs_beta,
                "beta0": float(beta[0]),
                "beta1": float(beta[1]),
                "beta2": float(beta[2]),
                "complexity": 3,
                "maturity_center": None,
                "maturity_scale": None,
                "tau": float(tau),
                "ridge_alpha": float(alpha),
                "score": float(score),
                "unstable": unstable,
                "tau_boundary": tau_boundary,
            }

            if best is None or rec["score"] < best["score"]:
                best = rec

    return best


def support_frame(issue_date: pd.DataFrame, min_issues: int) -> pd.DataFrame:
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

    support = support.loc[
        (support["n_issues"] >= min_issues)
        & (support["maturity_span"] >= 1.0)
    ].copy()

    support = support.sort_values(
        ["trd_exctn_dt", "n_issues", "gross_volume"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return support


def choose_groups(support: pd.DataFrame, max_curves: int) -> pd.DataFrame:
    if max_curves <= 0 or len(support) <= max_curves:
        return support.copy()

    idx = sorted(set(round(i * (len(support) - 1) / (max_curves - 1)) for i in range(max_curves)))
    return support.iloc[idx].reset_index(drop=True)


def fit_one_task(task: tuple[int, dict[str, Any], pd.DataFrame, np.ndarray, list[float]]) -> list[dict[str, Any]]:
    fit_index, row, group, tau_grid, alpha_grid = task

    models = [
        fit_linear(group),
        fit_quadratic(group),
        fit_ns_ridge(group, tau_grid=tau_grid, alpha_grid=alpha_grid),
    ]

    out = []

    for rec in models:
        if rec is None:
            continue

        rec = dict(rec)
        rec.update(
            {
                "fit_index": int(fit_index),
                "trade_date": str(row["trd_exctn_dt"]),
                "n_issues": int(row["n_issues"]),
                "n_issue_date_rows": int(row["n_issue_date_rows"]),
                "gross_volume": float(row["gross_volume"]),
                "maturity_min": float(row["maturity_min"]),
                "maturity_max": float(row["maturity_max"]),
                "maturity_span": float(row["maturity_span"]),
            }
        )
        out.append(rec)

    if not out:
        return []

    valid = [r for r in out if r.get("ok")]
    best = min(valid, key=lambda r: r["score"])

    for r in out:
        r["recommended"] = bool(r["model"] == best["model"])
        r["recommended_model"] = str(best["model"])

    return out


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    step06a = load_step06a_module(root)

    manifest_all = step06a.load_curve_ready_manifest(root, args.universe)
    manifest = step06a.choose_partitions(manifest_all, args.limit_partitions)

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
        futures = [
            pool.submit(step06a.read_partition_issue_date_agg, row, args.max_rows_per_partition)
            for row in rows
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            agg, rec = fut.result()
            partition_records.append(rec)
            if not agg.empty:
                agg_frames.append(agg)

            if i == 1 or i % args.progress_every == 0 or i == len(futures):
                ok = sum(1 for r in partition_records if r.get("ok"))
                usable = sum(int(r.get("usable_yield_rows") or 0) for r in partition_records)
                issue_rows = sum(int(r.get("issue_date_rows") or 0) for r in partition_records)
                print(
                    f"progress_read {i}/{len(futures)} ok={ok} "
                    f"usable_rows={usable:,} issue_date_rows={issue_rows:,}",
                    flush=True,
                )

    partition_detail = pd.DataFrame(partition_records).sort_values(
        ["start_date", "end_date", "output_path"]
    ).reset_index(drop=True)

    issue_date = pd.concat(agg_frames, ignore_index=True) if agg_frames else pd.DataFrame()

    support = support_frame(issue_date, min_issues=args.min_issues)
    chosen = choose_groups(support, max_curves=args.max_curves)

    print(f"issue_date_rows={len(issue_date):,}")
    print(f"eligible_curve_groups={len(support):,}")
    print(f"chosen_curve_groups={len(chosen):,}")

    grouped = {
        key: group.copy()
        for key, group in issue_date.groupby(["trd_exctn_dt", "issuer_id"], dropna=True, sort=False)
    }

    tau_grid = np.geomspace(args.tau_min, args.tau_max, args.tau_grid_size)
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    tasks = []
    for i, (_, row) in enumerate(chosen.iterrows()):
        key = (row["trd_exctn_dt"], row["issuer_id"])
        group = grouped.get(key)
        if group is not None and not group.empty:
            tasks.append((i, row.to_dict(), group, tau_grid, alpha_grid))

    fit_records: list[dict[str, Any]] = []

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(fit_one_task, task) for task in tasks]

            for i, fut in enumerate(as_completed(futures), start=1):
                fit_records.extend(fut.result())

                if i == 1 or i % args.progress_every == 0 or i == len(futures):
                    rec_df = pd.DataFrame(fit_records)
                    recommended = int(rec_df.get("recommended", pd.Series(dtype=bool)).fillna(False).sum()) if not rec_df.empty else 0
                    print(f"progress_fit {i}/{len(futures)} recommended={recommended}", flush=True)

    fit_detail = pd.DataFrame(fit_records)

    if not fit_detail.empty:
        recommended = fit_detail.loc[fit_detail["recommended"] == True].copy()
    else:
        recommended = pd.DataFrame()

    partition_detail_path = table_dir / "step06b_curve_model_comparison_partition_detail.csv"
    fit_detail_path = table_dir / "step06b_curve_model_comparison_fit_detail.csv"
    recommended_path = table_dir / "step06b_curve_model_comparison_recommended.csv"
    summary_path = table_dir / "step06b_curve_model_comparison_summary.json"

    partition_detail.to_csv(partition_detail_path, index=False)
    fit_detail.to_csv(fit_detail_path, index=False)
    recommended.to_csv(recommended_path, index=False)

    model_counts = (
        recommended["recommended_model"].value_counts().to_dict()
        if not recommended.empty and "recommended_model" in recommended.columns
        else {}
    )

    rmse_by_model = {}
    if not recommended.empty:
        for model, g in recommended.groupby("recommended_model"):
            rmse = pd.to_numeric(g["rmse"], errors="coerce").dropna()
            rmse_by_model[str(model)] = {
                "count": int(len(rmse)),
                "median": None if rmse.empty else round(float(rmse.median()), 6),
                "p90": None if rmse.empty else round(float(rmse.quantile(0.90)), 6),
                "p99": None if rmse.empty else round(float(rmse.quantile(0.99)), 6),
                "mean": None if rmse.empty else round(float(rmse.mean()), 6),
            }

    unstable_recommended = (
        int(recommended["unstable"].astype(bool).sum())
        if not recommended.empty and "unstable" in recommended.columns
        else 0
    )

    ns_rows = fit_detail.loc[fit_detail["model"] == "ns_ridge"].copy() if not fit_detail.empty else pd.DataFrame()
    ns_unstable = int(ns_rows["unstable"].astype(bool).sum()) if not ns_rows.empty else 0
    ns_boundary = int(ns_rows.get("tau_boundary", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not ns_rows.empty else 0

    summary = {
        "ok": bool(len(partition_detail) > 0 and partition_detail["ok"].astype(bool).all() and len(recommended) > 0),
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
        "eligible_curve_groups": int(len(support)),
        "chosen_curve_groups": int(len(chosen)),
        "fit_rows": int(len(fit_detail)),
        "recommended_curves": int(len(recommended)),
        "recommended_model_counts": {str(k): int(v) for k, v in model_counts.items()},
        "recommended_unstable_count": int(unstable_recommended),
        "recommended_unstable_pct": pct(int(unstable_recommended), int(len(recommended))),
        "ns_ridge_fit_rows": int(len(ns_rows)),
        "ns_ridge_unstable_count": int(ns_unstable),
        "ns_ridge_tau_boundary_count": int(ns_boundary),
        "rmse_by_recommended_model": rmse_by_model,
        "min_issues": int(args.min_issues),
        "max_curves": int(args.max_curves),
        "tau_grid_size": int(args.tau_grid_size),
        "tau_min": float(args.tau_min),
        "tau_max": float(args.tau_max),
        "alpha_grid": alpha_grid,
        "partition_detail_path": str(partition_detail_path),
        "fit_detail_path": str(fit_detail_path),
        "recommended_path": str(recommended_path),
        "note": "Local-only curve-family comparison smoke. Fit details omit issuer IDs and issue IDs.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step06b_curve_model_comparison_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in [summary_path, partition_detail_path, fit_detail_path, recommended_path]:
            tar.add(path, arcname=str(path.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"PARTITION_DETAIL={partition_detail_path}")
    print(f"FIT_DETAIL={fit_detail_path}")
    print(f"RECOMMENDED={recommended_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 06B guarded curve-family comparison smoke. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--limit-partitions", type=int, default=30)
    p.add_argument("--max-rows-per-partition", type=int, default=1_000_000)
    p.add_argument("--min-issues", type=int, default=4)
    p.add_argument("--max-curves", type=int, default=2000)
    p.add_argument("--tau-grid-size", type=int, default=25)
    p.add_argument("--tau-min", type=float, default=0.50)
    p.add_argument("--tau-max", type=float, default=15.0)
    p.add_argument("--alpha-grid", default="0.1,1.0,10.0,100.0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=25)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
