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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "last_residual_over_rmse",
    "last_residual_pctile",
    "last_issuer_date_residual_z",
    "last_bucket_residual_z",
    "mean_residual_pctile",
    "mean_issuer_date_residual_z",
    "mean_residual_over_rmse",
    "mean_bucket_residual_z",
    "cheap_share",
    "rich_share",
    "max_abs_issuer_date_residual_z",
    "mean_curve_rmse",
    "last_curve_rmse",
    "mean_n_issues",
    "max_n_issues",
    "total_trade_count",
    "total_gross_volume",
    "mean_log_gross_volume",
    "mean_liquidity_weight",
    "mean_curve_support_score",
    "unstable_curve_share",
    "mean_years_to_maturity",
    "last_years_to_maturity",
]

TARGET = "label_ret_1m"


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
        raise FileNotFoundError(f"Missing monthly label matrix: {path}")

    df = pq.ParquetFile(path).read().to_pandas()

    df["signal_month"] = pd.to_datetime(df["signal_month"], errors="coerce")
    df["label_month"] = pd.to_datetime(df["label_month"], errors="coerce")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "is_model_eligible_1m" in df.columns:
        df = df.loc[df["is_model_eligible_1m"].fillna(False).astype(bool)].copy()

    df = df.loc[df[TARGET].notna()].copy()
    return df.reset_index(drop=True)


def available_features(df: pd.DataFrame, min_non_null: int) -> list[str]:
    out = []
    for f in FEATURES:
        if f not in df.columns:
            continue
        s = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= min_non_null:
            df[f] = s
            out.append(f)
    if not out:
        raise RuntimeError("No usable features found.")
    return out


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_2002_2016": df.loc[df["signal_year"].between(2002, 2016)].copy(),
        "valid_2017_2019": df.loc[df["signal_year"].between(2017, 2019)].copy(),
        "test_2020_2024": df.loc[df["signal_year"].between(2020, 2024)].copy(),
    }


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def add_signal_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["signal_month", "issue_key", TARGET]].copy()

    # Positive residual means higher yield than fitted curve: cheap vs curve.
    signal_cols = [
        "last_residual_over_rmse",
        "last_residual_pctile",
        "last_issuer_date_residual_z",
        "last_bucket_residual_z",
        "mean_issuer_date_residual_z",
        "cheap_share",
    ]

    for col in signal_cols:
        if col in frame.columns:
            out[f"pred_signal__{col}"] = pd.to_numeric(frame[col], errors="coerce")

    return out


def add_model_predictions(model: Pipeline, frame: pd.DataFrame, features: list[str], model_name: str) -> pd.DataFrame:
    out = frame[["signal_month", "issue_key", TARGET]].copy()
    out[f"pred_model__{model_name}"] = model.predict(frame[features])
    return out


def spearman_ic(x: pd.Series, y: pd.Series) -> float | None:
    xx = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 25:
        return None
    val = xx.loc[m].rank(method="average").corr(yy.loc[m].rank(method="average"))
    if pd.isna(val):
        return None
    return float(val)


def decile_spread(g: pd.DataFrame, pred_col: str) -> float | None:
    x = pd.to_numeric(g[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(g[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
    m = x.notna() & y.notna()

    if int(m.sum()) < 100:
        return None

    gg = pd.DataFrame({"pred": x.loc[m], "y": y.loc[m]})
    lo = gg["pred"].quantile(0.10)
    hi = gg["pred"].quantile(0.90)

    low = gg.loc[gg["pred"] <= lo, "y"]
    high = gg.loc[gg["pred"] >= hi, "y"]

    if low.empty or high.empty:
        return None

    return float(high.mean() - low.mean())


def monthly_metrics(preds: pd.DataFrame, pred_col: str, min_month_obs: int) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []

    for month, g in preds.groupby("signal_month", sort=True):
        valid = g[[pred_col, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < min_month_obs:
            continue

        ic = spearman_ic(valid[pred_col], valid[TARGET])
        spread = decile_spread(valid, pred_col)

        rows.append(
            {
                "signal_month": str(pd.to_datetime(month).date()),
                "n_obs": int(len(valid)),
                "spearman_ic": ic,
                "top_bottom_decile_spread_1m": spread,
            }
        )

    monthly = pd.DataFrame(rows)

    if monthly.empty:
        summary = {
            "pred_col": pred_col,
            "months": 0,
            "mean_ic": None,
            "median_ic": None,
            "ic_std": None,
            "icir_annualized": None,
            "positive_ic_share": None,
            "mean_decile_spread_1m": None,
            "median_decile_spread_1m": None,
            "spread_std": None,
            "spread_ir_annualized": None,
        }
        return summary, monthly

    ic = pd.to_numeric(monthly["spearman_ic"], errors="coerce").dropna()
    sp = pd.to_numeric(monthly["top_bottom_decile_spread_1m"], errors="coerce").dropna()

    summary = {
        "pred_col": pred_col,
        "months": int(monthly["signal_month"].nunique()),
        "mean_ic": None if ic.empty else round(float(ic.mean()), 10),
        "median_ic": None if ic.empty else round(float(ic.median()), 10),
        "ic_std": None if len(ic) <= 1 else round(float(ic.std(ddof=1)), 10),
        "icir_annualized": None if len(ic) <= 1 or ic.std(ddof=1) == 0 else round(float(np.sqrt(12) * ic.mean() / ic.std(ddof=1)), 6),
        "positive_ic_share": None if ic.empty else round(float((ic > 0).mean()), 6),
        "mean_decile_spread_1m": None if sp.empty else round(float(sp.mean()), 10),
        "median_decile_spread_1m": None if sp.empty else round(float(sp.median()), 10),
        "spread_std": None if len(sp) <= 1 else round(float(sp.std(ddof=1)), 10),
        "spread_ir_annualized": None if len(sp) <= 1 or sp.std(ddof=1) == 0 else round(float(np.sqrt(12) * sp.mean() / sp.std(ddof=1)), 6),
    }

    return summary, monthly


def evaluate_prediction_frame(
    frame_name: str,
    preds: pd.DataFrame,
    min_month_obs: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    metric_rows = []
    monthly_frames = []

    pred_cols = [c for c in preds.columns if c.startswith("pred_")]

    for pred_col in pred_cols:
        m, monthly = monthly_metrics(preds, pred_col, min_month_obs=min_month_obs)
        m["sample"] = frame_name
        metric_rows.append(m)

        if not monthly.empty:
            monthly = monthly.copy()
            monthly["sample"] = frame_name
            monthly["pred_col"] = pred_col
            monthly_frames.append(monthly)

    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    return metric_rows, monthly_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 08A transparent monthly baseline models. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--min-non-null", type=int, default=1000)
    parser.add_argument("--min-month-obs", type=int, default=100)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    df = load_matrix(root, args.universe)
    feats = available_features(df, min_non_null=args.min_non_null)

    splits = split_data(df)
    train = splits["train_2002_2016"]
    valid = splits["valid_2017_2019"]
    test = splits["test_2020_2024"]

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"Empty split. train={len(train)} valid={len(valid)} test={len(test)}"
        )

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    table_dir = root / "artifacts" / "tables"
    model_dir = root / "artifacts" / "model_cards"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"rows_total={len(df):,}")
    print(f"train={len(train):,} valid={len(valid):,} test={len(test):,}")
    print(f"features={len(feats)}")

    all_metric_rows = []
    all_monthly_rows = []

    # 1. No-model signal baselines.
    for sample_name, frame in splits.items():
        preds = add_signal_predictions(frame)
        metric_rows, monthly = evaluate_prediction_frame(sample_name, preds, min_month_obs=args.min_month_obs)
        all_metric_rows.extend(metric_rows)
        if not monthly.empty:
            all_monthly_rows.append(monthly)

    # 2. Ridge grid using validation IC.
    ridge_valid_scores = []
    ridge_models: dict[float, Pipeline] = {}

    for alpha in alphas:
        print(f"fit_ridge_alpha={alpha}", flush=True)
        model = make_model(alpha)
        model.fit(train[feats], train[TARGET])
        ridge_models[alpha] = model

        valid_preds = add_model_predictions(model, valid, feats, f"ridge_alpha_{alpha:g}")
        metric_rows, monthly = evaluate_prediction_frame("valid_2017_2019", valid_preds, min_month_obs=args.min_month_obs)

        m = metric_rows[0] if metric_rows else {}
        ridge_valid_scores.append(
            {
                "alpha": alpha,
                "valid_mean_ic": m.get("mean_ic"),
                "valid_mean_decile_spread_1m": m.get("mean_decile_spread_1m"),
                "valid_icir_annualized": m.get("icir_annualized"),
            }
        )

        all_metric_rows.extend(metric_rows)
        if not monthly.empty:
            all_monthly_rows.append(monthly)

    scores = pd.DataFrame(ridge_valid_scores)
    scores_nonnull = scores.dropna(subset=["valid_mean_ic"]).copy()
    if scores_nonnull.empty:
        best_alpha = alphas[0]
    else:
        best_alpha = float(scores_nonnull.sort_values("valid_mean_ic", ascending=False).iloc[0]["alpha"])

    print(f"best_alpha={best_alpha}", flush=True)

    # Refit on train+valid, evaluate train/valid/test.
    train_valid = pd.concat([train, valid], ignore_index=True)
    best_model = make_model(best_alpha)
    best_model.fit(train_valid[feats], train_valid[TARGET])

    best_model_name = f"ridge_best_alpha_{best_alpha:g}"
    best_eval_frames = {
        "train_2002_2016": train,
        "valid_2017_2019": valid,
        "test_2020_2024": test,
        "train_valid_2002_2019": train_valid,
    }

    for sample_name, frame in best_eval_frames.items():
        preds = add_model_predictions(best_model, frame, feats, best_model_name)
        metric_rows, monthly = evaluate_prediction_frame(sample_name, preds, min_month_obs=args.min_month_obs)
        all_metric_rows.extend(metric_rows)
        if not monthly.empty:
            all_monthly_rows.append(monthly)

    metrics = pd.DataFrame(all_metric_rows)
    monthly_metrics_df = pd.concat(all_monthly_rows, ignore_index=True) if all_monthly_rows else pd.DataFrame()

    metrics_path = table_dir / "step08a_monthly_baseline_metrics.csv"
    monthly_metrics_path = table_dir / "step08a_monthly_baseline_monthly_metrics.csv"
    ridge_grid_path = table_dir / "step08a_ridge_validation_grid.csv"
    coef_path = table_dir / "step08a_ridge_coefficients.csv"
    summary_path = table_dir / "step08a_monthly_baseline_summary.json"

    metrics.to_csv(metrics_path, index=False)
    monthly_metrics_df.to_csv(monthly_metrics_path, index=False)
    scores.to_csv(ridge_grid_path, index=False)

    ridge = best_model.named_steps["ridge"]
    coef = pd.DataFrame({"feature": feats, "coef": ridge.coef_}).sort_values("coef", ascending=False)
    coef.to_csv(coef_path, index=False)

    test_metrics = metrics.loc[
        (metrics["sample"] == "test_2020_2024")
        & (metrics["pred_col"] == f"pred_model__{best_model_name}")
    ].copy()

    signal_test = metrics.loc[
        (metrics["sample"] == "test_2020_2024")
        & (metrics["pred_col"].isin(
            [
                "pred_signal__last_residual_over_rmse",
                "pred_signal__last_issuer_date_residual_z",
                "pred_signal__last_residual_pctile",
                "pred_signal__last_bucket_residual_z",
            ]
        ))
    ].copy()

    summary = {
        "ok": bool(not test_metrics.empty),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "target": TARGET,
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "rows_test": int(len(test)),
        "features": feats,
        "n_features": int(len(feats)),
        "best_alpha": float(best_alpha),
        "best_model_name": best_model_name,
        "test_best_model_metrics": test_metrics.to_dict("records"),
        "test_signal_baseline_metrics": signal_test.to_dict("records"),
        "top_positive_coefficients": coef.head(10).to_dict("records"),
        "top_negative_coefficients": coef.tail(10).to_dict("records"),
        "metrics_path": str(metrics_path),
        "monthly_metrics_path": str(monthly_metrics_path),
        "ridge_grid_path": str(ridge_grid_path),
        "coef_path": str(coef_path),
        "method_note": "Transparent baseline only. Train 2002-2016, validation 2017-2019, test 2020-2024. 2025 excluded as incomplete/holdout.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step08a_monthly_baseline_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, metrics_path, monthly_metrics_path, ridge_grid_path, coef_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"MONTHLY_METRICS={monthly_metrics_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
