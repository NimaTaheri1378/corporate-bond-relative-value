#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
        raise FileNotFoundError(f"Missing model-eligible monthly matrix: {path}")

    df = pq.ParquetFile(path).read().to_pandas()

    df["signal_month"] = pd.to_datetime(df["signal_month"], errors="coerce")
    df["label_month"] = pd.to_datetime(df["label_month"], errors="coerce")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "is_model_eligible_1m" in df.columns:
        df = df.loc[df["is_model_eligible_1m"].fillna(False).astype(bool)].copy()

    df = df.loc[df[TARGET].notna()].copy()
    df["issue_key"] = df["issue_key"].astype("string")
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame, min_non_null: int) -> list[str]:
    usable = []

    for col in FEATURES:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= min_non_null:
            df[col] = s.astype("float32")
            usable.append(col)

    if not usable:
        raise RuntimeError("No usable features found.")

    return usable


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_2002_2016": df.loc[df["signal_year"].between(2002, 2016)].copy(),
        "valid_2017_2019": df.loc[df["signal_year"].between(2017, 2019)].copy(),
        "test_2020_2024": df.loc[df["signal_year"].between(2020, 2024)].copy(),
    }


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


def turnover(prev: set[str] | None, current: set[str]) -> float:
    if prev is None:
        return 1.0
    if not current:
        return 1.0
    return float(1.0 - len(prev & current) / max(len(current), 1))


def monthly_signal_metrics(
    df: pd.DataFrame,
    pred_col: str,
    sample: str,
    min_month_obs: int,
    tail_frac: float,
    cost_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    prev_long: set[str] | None = None
    prev_short: set[str] | None = None

    for month, g in df.groupby("signal_month", sort=True):
        x = pd.to_numeric(g[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(g[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = x.notna() & y.notna() & g["issue_key"].notna()

        gg = g.loc[m, ["signal_month", "issue_key", TARGET]].copy()
        gg["pred"] = x.loc[m].to_numpy()

        if len(gg) < min_month_obs:
            continue

        lo = gg["pred"].quantile(tail_frac)
        hi = gg["pred"].quantile(1.0 - tail_frac)

        short = gg.loc[gg["pred"] <= lo].copy()
        long = gg.loc[gg["pred"] >= hi].copy()

        if long.empty or short.empty:
            continue

        long_set = set(long["issue_key"].astype(str))
        short_set = set(short["issue_key"].astype(str))

        long_turn = turnover(prev_long, long_set)
        short_turn = turnover(prev_short, short_set)
        ls_turn = long_turn + short_turn

        long_ret = float(long[TARGET].mean())
        short_ret = float(short[TARGET].mean())
        gross = long_ret - short_ret
        net = gross - (cost_bps / 10000.0) * ls_turn

        rows.append(
            {
                "sample": sample,
                "signal_month": pd.to_datetime(month),
                "n_obs": int(len(gg)),
                "n_long": int(len(long)),
                "n_short": int(len(short)),
                "long_ret": long_ret,
                "short_leg_ret": short_ret,
                "long_short_ret_gross": gross,
                "long_short_ret_net": net,
                "long_turnover": long_turn,
                "short_turnover": short_turn,
                "long_short_turnover": ls_turn,
                "spearman_ic": safe_spearman(gg["pred"], gg[TARGET]),
            }
        )

        prev_long = long_set
        prev_short = short_set

    monthly = pd.DataFrame(rows)

    if monthly.empty:
        return {
            "sample": sample,
            "pred_col": pred_col,
            "months": 0,
            "mean_ic": None,
            "mean_long_short_ret_gross": None,
            "mean_long_short_ret_net": None,
            "sharpe_net": None,
        }, monthly

    rnet = pd.to_numeric(monthly["long_short_ret_net"], errors="coerce").dropna()
    rgross = pd.to_numeric(monthly["long_short_ret_gross"], errors="coerce").dropna()
    ic = pd.to_numeric(monthly["spearman_ic"], errors="coerce").dropna()

    cumulative_net = (1.0 + rnet).cumprod()
    dd = cumulative_net / cumulative_net.cummax() - 1.0

    summary = {
        "sample": sample,
        "pred_col": pred_col,
        "months": int(len(monthly)),
        "mean_ic": None if ic.empty else round(float(ic.mean()), 10),
        "median_ic": None if ic.empty else round(float(ic.median()), 10),
        "positive_ic_share": None if ic.empty else round(float((ic > 0).mean()), 6),
        "mean_long_short_ret_gross": None if rgross.empty else round(float(rgross.mean()), 10),
        "mean_long_short_ret_net": None if rnet.empty else round(float(rnet.mean()), 10),
        "annualized_return_net_approx": None if rnet.empty else round(float(12.0 * rnet.mean()), 10),
        "annualized_vol_net": None if len(rnet) <= 1 else round(float(np.sqrt(12.0) * rnet.std(ddof=1)), 10),
        "sharpe_net": None if len(rnet) <= 1 or rnet.std(ddof=1) == 0 else round(float(np.sqrt(12.0) * rnet.mean() / rnet.std(ddof=1)), 6),
        "cumulative_return_net": None if rnet.empty else round(float(cumulative_net.iloc[-1] - 1.0), 10),
        "max_drawdown_net": None if rnet.empty else round(float(dd.min()), 10),
        "positive_net_month_share": None if rnet.empty else round(float((rnet > 0).mean()), 6),
        "mean_turnover": round(float(monthly["long_short_turnover"].mean()), 10),
        "mean_n_long": round(float(monthly["n_long"].mean()), 3),
        "mean_n_short": round(float(monthly["n_short"].mean()), 3),
    }

    return summary, monthly



def lgb_x(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.loc[:, features].copy()
    for col in features:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
    return out


def lgb_y(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame[TARGET], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )


def make_model(params: dict[str, Any], n_jobs: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(params.get("n_estimators", 2000)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 63)),
        max_depth=int(params.get("max_depth", -1)),
        min_child_samples=int(params.get("min_child_samples", 200)),
        subsample=float(params.get("subsample", 0.90)),
        subsample_freq=1,
        colsample_bytree=float(params.get("colsample_bytree", 0.90)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"name": "lgbm_small", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 300, "reg_lambda": 10.0},
        {"name": "lgbm_medium", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 200, "reg_lambda": 5.0},
        {"name": "lgbm_large_regularized", "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 300, "reg_lambda": 20.0},
        {"name": "lgbm_shallow", "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5, "min_child_samples": 500, "reg_lambda": 10.0},
        {"name": "lgbm_deep_sparse", "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 500, "colsample_bytree": 0.75, "reg_lambda": 50.0},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 08D CPU LightGBM monthly model. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-non-null", type=int, default=1000)
    parser.add_argument("--min-month-obs", type=int, default=100)
    parser.add_argument("--tail-frac", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--max-boost-rounds", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    df = load_matrix(root, args.universe)
    feats = prepare_features(df, min_non_null=args.min_non_null)

    splits = split_data(df)
    train = splits["train_2002_2016"]
    valid = splits["valid_2017_2019"]
    test = splits["test_2020_2024"]

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Empty split: train={len(train)} valid={len(valid)} test={len(test)}")

    table_dir = root / "artifacts" / "tables"
    model_dir = root / "artifacts" / "model_cards"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"lightgbm={lgb.__version__}")
    print(f"rows train={len(train):,} valid={len(valid):,} test={len(test):,}")
    print(f"features={len(feats)} n_jobs={args.n_jobs}")

    grid_results = []
    monthly_frames = []

    best_score = -1e9
    best_cfg = None
    best_iter = None

    for cfg in candidate_grid():
        cfg = dict(cfg)
        cfg["n_estimators"] = args.max_boost_rounds

        print(f"fit_candidate={cfg['name']}", flush=True)

        model = make_model(cfg, n_jobs=args.n_jobs, seed=args.seed)
        model.fit(
            lgb_x(train, feats),
            lgb_y(train),
            eval_set=[(lgb_x(valid, feats), lgb_y(valid))],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        best_iteration = int(getattr(model, "best_iteration_", None) or args.max_boost_rounds)

        valid_frame = valid[["signal_month", "issue_key", TARGET]].copy()
        valid_frame["pred_lgbm"] = model.predict(lgb_x(valid, feats), num_iteration=best_iteration)

        valid_metrics, valid_monthly = monthly_signal_metrics(
            valid_frame,
            pred_col="pred_lgbm",
            sample="valid_2017_2019",
            min_month_obs=args.min_month_obs,
            tail_frac=args.tail_frac,
            cost_bps=args.cost_bps,
        )

        row = {
            **cfg,
            "best_iteration": best_iteration,
            "valid_mean_ic": valid_metrics["mean_ic"],
            "valid_net_ret": valid_metrics["mean_long_short_ret_net"],
            "valid_sharpe_net": valid_metrics["sharpe_net"],
        }
        grid_results.append(row)

        print(json.dumps(row), flush=True)

        score = valid_metrics["mean_ic"]
        if score is not None and score > best_score:
            best_score = float(score)
            best_cfg = cfg
            best_iter = best_iteration

    if best_cfg is None:
        raise RuntimeError("No LightGBM candidate produced validation IC.")

    print(f"best_cfg={best_cfg['name']} best_iter={best_iter} best_valid_ic={best_score}", flush=True)

    train_valid = pd.concat([train, valid], ignore_index=True)

    final_cfg = dict(best_cfg)
    final_cfg["n_estimators"] = int(best_iter)

    final_model = make_model(final_cfg, n_jobs=args.n_jobs, seed=args.seed)
    final_model.fit(lgb_x(train_valid, feats), lgb_y(train_valid))

    eval_frames = {
        "train_2002_2016": train,
        "valid_2017_2019": valid,
        "test_2020_2024": test,
        "train_valid_2002_2019": train_valid,
    }

    metric_rows = []

    for sample, frame in eval_frames.items():
        preds = frame[["signal_month", "issue_key", TARGET]].copy()
        preds["pred_lgbm"] = final_model.predict(lgb_x(frame, feats))
        metrics, monthly = monthly_signal_metrics(
            preds,
            pred_col="pred_lgbm",
            sample=sample,
            min_month_obs=args.min_month_obs,
            tail_frac=args.tail_frac,
            cost_bps=args.cost_bps,
        )
        metric_rows.append(metrics)

        if not monthly.empty:
            monthly["pred_col"] = "pred_lgbm"
            monthly_frames.append(monthly)

    metrics_df = pd.DataFrame(metric_rows)
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    grid_df = pd.DataFrame(grid_results)

    importance = pd.DataFrame(
        {
            "feature": feats,
            "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": final_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    metrics_path = table_dir / "step08d_lightgbm_metrics.csv"
    monthly_path = table_dir / "step08d_lightgbm_monthly_returns.csv"
    grid_path = table_dir / "step08d_lightgbm_validation_grid.csv"
    importance_path = table_dir / "step08d_lightgbm_feature_importance.csv"
    summary_path = table_dir / "step08d_lightgbm_summary.json"
    model_card_path = model_dir / "step08d_lightgbm_model_card.json"
    model_path = model_dir / "step08d_lightgbm_model.txt"

    metrics_df.to_csv(metrics_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    grid_df.to_csv(grid_path, index=False)
    importance.to_csv(importance_path, index=False)
    final_model.booster_.save_model(str(model_path))

    test_metrics = metrics_df.loc[metrics_df["sample"] == "test_2020_2024"].to_dict("records")

    summary = {
        "ok": bool(test_metrics),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "lightgbm_version": lgb.__version__,
        "device": "cpu",
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "rows_test": int(len(test)),
        "features": feats,
        "n_features": int(len(feats)),
        "best_config": best_cfg,
        "best_iteration": int(best_iter),
        "best_valid_mean_ic": round(float(best_score), 10),
        "test_metrics": test_metrics,
        "top_feature_importance_gain": importance.head(15).to_dict("records"),
        "metrics_path": str(metrics_path),
        "monthly_returns_path": str(monthly_path),
        "grid_path": str(grid_path),
        "importance_path": str(importance_path),
        "model_card_path": str(model_card_path),
        "model_file_local_do_not_upload": str(model_path),
        "method_note": "CPU LightGBM tabular model. GPU LightGBM is unavailable in this environment. Train 2002-2016, validation 2017-2019, test 2020-2024.",
    }

    write_json(summary_path, summary)
    write_json(model_card_path, summary)

    bundle = log_dir / f"step08d_lightgbm_monthly_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, model_card_path, metrics_path, monthly_path, grid_path, importance_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"MONTHLY_RETURNS={monthly_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
