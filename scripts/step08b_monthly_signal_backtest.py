#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TARGET = "label_ret_1m"

BASE_SIGNALS = [
    "last_residual_over_rmse",
    "last_residual_pctile",
    "last_issuer_date_residual_z",
    "last_bucket_residual_z",
    "mean_issuer_date_residual_z",
    "cheap_share",
]

COST_BPS_GRID = [0.0, 5.0, 10.0, 25.0, 50.0]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_matrix(root: Path, universe: str) -> pd.DataFrame:
    path = (
        root
        / "data"
        / "processed"
        / "monthly_label_matrix_v1"
        / f"universe={universe}"
        / "horizon=1m"
        / "monthly_label_matrix_model_eligible.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing model-eligible monthly matrix: {path}")

    df = pq.ParquetFile(path).read().to_pandas()

    df["signal_month"] = pd.to_datetime(df["signal_month"], errors="coerce")
    df["label_month"] = pd.to_datetime(df["label_month"], errors="coerce")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "is_model_eligible_1m" in df.columns:
        df = df.loc[df["is_model_eligible_1m"].fillna(False).astype(bool)].copy()

    df = df.loc[df[TARGET].notna()].copy()

    if "issue_key" not in df.columns:
        raise ValueError("Expected issue_key in monthly label matrix.")

    df["issue_key"] = df["issue_key"].astype("string")
    return df.reset_index(drop=True)


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_2002_2016": df.loc[df["signal_year"].between(2002, 2016)].copy(),
        "valid_2017_2019": df.loc[df["signal_year"].between(2017, 2019)].copy(),
        "test_2020_2024": df.loc[df["signal_year"].between(2020, 2024)].copy(),
        "all_2002_2024": df.loc[df["signal_year"].between(2002, 2024)].copy(),
    }


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    available = [c for c in ["last_residual_over_rmse", "last_residual_pctile", "last_issuer_date_residual_z", "last_bucket_residual_z"] if c in out.columns]

    rank_cols = []
    for col in available:
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rc = f"_rank_{col}"
        out[rc] = out.groupby("signal_month", sort=False)[col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").rank(method="average", pct=True)
        )
        rank_cols.append(rc)

    if rank_cols:
        out["composite_residual_rank"] = out[rank_cols].mean(axis=1)

    return out


def safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    xx = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    m = xx.notna() & yy.notna()

    if int(m.sum()) < 25:
        return None

    val = xx.loc[m].rank(method="average").corr(yy.loc[m].rank(method="average"))
    if pd.isna(val):
        return None
    return float(val)


def choose_portfolios(
    month_frame: pd.DataFrame,
    signal_col: str,
    min_month_obs: int,
    tail_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    x = month_frame.copy()
    x[signal_col] = pd.to_numeric(x[signal_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x[TARGET] = pd.to_numeric(x[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=[signal_col, TARGET, "issue_key"]).copy()

    if len(x) < min_month_obs:
        return None, None

    lo = x[signal_col].quantile(tail_frac)
    hi = x[signal_col].quantile(1.0 - tail_frac)

    short = x.loc[x[signal_col] <= lo].copy()
    long = x.loc[x[signal_col] >= hi].copy()

    if len(long) == 0 or len(short) == 0:
        return None, None

    return long, short


def turnover(prev: set[str] | None, current: set[str]) -> float:
    if prev is None:
        return 1.0
    if not current and not prev:
        return 0.0
    if not current:
        return 1.0
    kept = len(prev & current)
    return float(1.0 - kept / max(len(current), 1))


def run_signal_backtest(
    df: pd.DataFrame,
    sample_name: str,
    signal_col: str,
    min_month_obs: int,
    tail_frac: float,
    cost_bps_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    prev_long: set[str] | None = None
    prev_short: set[str] | None = None

    for month, g in df.groupby("signal_month", sort=True):
        long, short = choose_portfolios(g, signal_col, min_month_obs=min_month_obs, tail_frac=tail_frac)

        if long is None or short is None:
            continue

        long_set = set(long["issue_key"].astype(str))
        short_set = set(short["issue_key"].astype(str))

        long_turnover = turnover(prev_long, long_set)
        short_turnover = turnover(prev_short, short_set)
        ls_turnover = long_turnover + short_turnover

        long_ret = float(long[TARGET].mean())
        short_ret = float(short[TARGET].mean())
        ls_ret = long_ret - short_ret

        ic = safe_spearman(g[signal_col], g[TARGET])

        row = {
            "sample": sample_name,
            "signal": signal_col,
            "signal_month": pd.to_datetime(month),
            "n_obs_month": int(g[[signal_col, TARGET]].dropna().shape[0]),
            "n_long": int(len(long)),
            "n_short": int(len(short)),
            "long_ret_gross": long_ret,
            "short_leg_ret_gross": short_ret,
            "long_short_ret_gross": ls_ret,
            "long_turnover": long_turnover,
            "short_turnover": short_turnover,
            "long_short_turnover": ls_turnover,
            "spearman_ic": ic,
        }

        for cost_bps in cost_bps_grid:
            cost = float(cost_bps) / 10_000.0
            row[f"long_ret_net_cost_{cost_bps:g}bps"] = long_ret - cost * long_turnover
            row[f"long_short_ret_net_cost_{cost_bps:g}bps"] = ls_ret - cost * ls_turnover

        rows.append(row)

        prev_long = long_set
        prev_short = short_set

    monthly = pd.DataFrame(rows)

    if monthly.empty:
        return monthly, pd.DataFrame()

    metric_rows = []
    return_cols = ["long_ret_gross", "long_short_ret_gross"]
    for cost_bps in cost_bps_grid:
        return_cols.append(f"long_ret_net_cost_{cost_bps:g}bps")
        return_cols.append(f"long_short_ret_net_cost_{cost_bps:g}bps")

    for ret_col in return_cols:
        r = pd.to_numeric(monthly[ret_col], errors="coerce").dropna()
        if r.empty:
            continue

        cumulative = (1.0 + r).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0

        metric_rows.append(
            {
                "sample": sample_name,
                "signal": signal_col,
                "return_col": ret_col,
                "months": int(len(r)),
                "mean_monthly_return": round(float(r.mean()), 10),
                "median_monthly_return": round(float(r.median()), 10),
                "std_monthly_return": round(float(r.std(ddof=1)), 10) if len(r) > 1 else 0.0,
                "annualized_return_approx": round(float(12.0 * r.mean()), 10),
                "annualized_vol": round(float(np.sqrt(12.0) * r.std(ddof=1)), 10) if len(r) > 1 else 0.0,
                "sharpe_approx": None if len(r) <= 1 or r.std(ddof=1) == 0 else round(float(np.sqrt(12.0) * r.mean() / r.std(ddof=1)), 6),
                "cumulative_return": round(float(cumulative.iloc[-1] - 1.0), 10),
                "max_drawdown": round(float(drawdown.min()), 10),
                "positive_month_share": round(float((r > 0).mean()), 6),
                "mean_monthly_ic": round(float(pd.to_numeric(monthly["spearman_ic"], errors="coerce").mean()), 10),
                "median_monthly_ic": round(float(pd.to_numeric(monthly["spearman_ic"], errors="coerce").median()), 10),
                "mean_long_turnover": round(float(monthly["long_turnover"].mean()), 10),
                "mean_short_turnover": round(float(monthly["short_turnover"].mean()), 10),
                "mean_long_short_turnover": round(float(monthly["long_short_turnover"].mean()), 10),
                "mean_n_long": round(float(monthly["n_long"].mean()), 3),
                "mean_n_short": round(float(monthly["n_short"].mean()), 3),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    return monthly, metrics


def run(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    df = load_matrix(root, args.universe)
    df = add_composite_scores(df)

    signal_cols = [s for s in BASE_SIGNALS + ["composite_residual_rank"] if s in df.columns]
    if not signal_cols:
        raise RuntimeError("No signal columns available.")

    splits = split_frame(df)
    cost_grid = [float(x) for x in args.cost_bps_grid.split(",") if x.strip()]

    all_monthly = []
    all_metrics = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"rows={len(df):,}")
    print(f"signals={signal_cols}")
    print(f"tail_frac={args.tail_frac}")
    print(f"cost_grid={cost_grid}")

    for sample_name, frame in splits.items():
        print(f"sample={sample_name} rows={len(frame):,}", flush=True)

        for signal_col in signal_cols:
            monthly, metrics = run_signal_backtest(
                df=frame,
                sample_name=sample_name,
                signal_col=signal_col,
                min_month_obs=args.min_month_obs,
                tail_frac=args.tail_frac,
                cost_bps_grid=cost_grid,
            )

            if not monthly.empty:
                all_monthly.append(monthly)
            if not metrics.empty:
                all_metrics.append(metrics)

    monthly_all = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
    metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = table_dir / "step08b_monthly_signal_backtest_monthly_returns.csv"
    metrics_path = table_dir / "step08b_monthly_signal_backtest_metrics.csv"
    summary_path = table_dir / "step08b_monthly_signal_backtest_summary.json"

    monthly_all.to_csv(monthly_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)

    test_net_col = f"long_short_ret_net_cost_{args.headline_cost_bps:g}bps"
    headline = metrics_all.loc[
        (metrics_all["sample"] == "test_2020_2024")
        & (metrics_all["return_col"] == test_net_col)
    ].copy()

    if not headline.empty:
        headline = headline.sort_values("sharpe_approx", ascending=False)
        best_test = headline.head(10).to_dict("records")
    else:
        best_test = []

    gross_test = metrics_all.loc[
        (metrics_all["sample"] == "test_2020_2024")
        & (metrics_all["return_col"] == "long_short_ret_gross")
    ].sort_values("sharpe_approx", ascending=False).head(10)

    summary = {
        "ok": bool(not metrics_all.empty and len(best_test) > 0),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "rows_loaded": int(len(df)),
        "signals": signal_cols,
        "tail_frac": float(args.tail_frac),
        "min_month_obs": int(args.min_month_obs),
        "cost_bps_grid": cost_grid,
        "headline_cost_bps": float(args.headline_cost_bps),
        "headline_test_net_return_col": test_net_col,
        "best_test_net_long_short": best_test,
        "best_test_gross_long_short": gross_test.to_dict("records"),
        "metrics_path": str(metrics_path),
        "monthly_returns_path": str(monthly_path),
        "method_note": "Equal-weight monthly top-vs-bottom tail portfolios by signal. Net cost subtracts one-way bps times long+short turnover. First month assumes full turnover.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step08b_monthly_signal_backtest_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, metrics_path, monthly_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"MONTHLY_RETURNS={monthly_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 08B monthly residual-signal decile backtest. No WRDS.")
    p.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    p.add_argument("--universe", default="core_public")
    p.add_argument("--tail-frac", type=float, default=0.10)
    p.add_argument("--min-month-obs", type=int, default=100)
    p.add_argument("--cost-bps-grid", default="0,5,10,25,50")
    p.add_argument("--headline-cost-bps", type=float, default=10.0)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
