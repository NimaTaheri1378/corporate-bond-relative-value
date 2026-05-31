#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_2002_2016": df.loc[df["signal_year"].between(2002, 2016)].copy(),
        "valid_2017_2019": df.loc[df["signal_year"].between(2017, 2019)].copy(),
        "test_2020_2024": df.loc[df["signal_year"].between(2020, 2024)].copy(),
    }


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


def fit_preprocessor(train: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    x = train[features].to_numpy(dtype=np.float32)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std <= 1e-8] = 1.0

    y = train[TARGET].to_numpy(dtype=np.float32)
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y))
    if not np.isfinite(y_std) or y_std <= 1e-8:
        y_std = 1.0

    return {"median": med, "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std}


def transform_x(df: pd.DataFrame, features: list[str], prep: dict[str, Any]) -> np.ndarray:
    x = df[features].to_numpy(dtype=np.float32)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(prep["median"], inds[1])
    x = (x - prep["mean"]) / prep["std"]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype("float32")


def transform_y(df: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    y = df[TARGET].to_numpy(dtype=np.float32)
    y = (y - prep["y_mean"]) / prep["y_std"]
    return y.astype("float32")


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = n_features

        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []

    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        pb = model(xb).detach().cpu().numpy()
        preds.append(pb)

    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


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


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_frame: pd.DataFrame,
    valid_x: np.ndarray,
    prep: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

    best_state = None
    best_score = -1e9
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        valid_pred_std = predict(model, valid_x, device=device, batch_size=args.predict_batch_size)
        valid_pred = valid_pred_std * prep["y_std"] + prep["y_mean"]

        tmp = valid_frame[["signal_month", "issue_key", TARGET]].copy()
        tmp["pred_mlp"] = valid_pred

        valid_metrics, _ = monthly_signal_metrics(
            tmp,
            pred_col="pred_mlp",
            sample="valid_2017_2019",
            min_month_obs=args.min_month_obs,
            tail_frac=args.tail_frac,
            cost_bps=args.cost_bps,
        )

        score = valid_metrics["mean_ic"]
        if score is None:
            score = -1e9

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "valid_mean_ic": valid_metrics["mean_ic"],
            "valid_net_ret": valid_metrics["mean_long_short_ret_net"],
            "valid_sharpe_net": valid_metrics["sharpe_net"],
        }
        history.append(rec)

        print(json.dumps(rec), flush=True)

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_valid_mean_ic": best_score}


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 08C GPU PyTorch MLP monthly baseline. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--predict-batch-size", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--min-non-null", type=int, default=1000)
    parser.add_argument("--min-month-obs", type=int, default=100)
    parser.add_argument("--tail-frac", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    df = load_matrix(root, args.universe)
    feats = prepare_features(df, min_non_null=args.min_non_null)

    splits = split_data(df)
    train = splits["train_2002_2016"]
    valid = splits["valid_2017_2019"]
    test = splits["test_2020_2024"]

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Empty split: train={len(train)} valid={len(valid)} test={len(test)}")

    prep = fit_preprocessor(train, feats)

    train_x = transform_x(train, feats, prep)
    train_y = transform_y(train, prep)
    valid_x = transform_x(valid, feats, prep)
    test_x = transform_x(test, feats, prep)

    model = MLP(n_features=len(feats), hidden=hidden, dropout=args.dropout)

    table_dir = root / "artifacts" / "tables"
    model_dir = root / "artifacts" / "model_cards"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"rows train={len(train):,} valid={len(valid):,} test={len(test):,}")
    print(f"features={len(feats)} hidden={hidden}")

    model, history, best = train_model(
        model=model,
        train_x=train_x,
        train_y=train_y,
        valid_frame=valid,
        valid_x=valid_x,
        prep=prep,
        device=device,
        args=args,
    )

    eval_frames = {
        "train_2002_2016": train,
        "valid_2017_2019": valid,
        "test_2020_2024": test,
    }
    eval_x = {
        "train_2002_2016": train_x,
        "valid_2017_2019": valid_x,
        "test_2020_2024": test_x,
    }

    metric_rows = []
    monthly_frames = []

    for sample, frame in eval_frames.items():
        pred_std = predict(model, eval_x[sample], device=device, batch_size=args.predict_batch_size)
        pred = pred_std * prep["y_std"] + prep["y_mean"]

        pframe = frame[["signal_month", "issue_key", TARGET]].copy()
        pframe["pred_mlp"] = pred

        metrics, monthly = monthly_signal_metrics(
            pframe,
            pred_col="pred_mlp",
            sample=sample,
            min_month_obs=args.min_month_obs,
            tail_frac=args.tail_frac,
            cost_bps=args.cost_bps,
        )
        metric_rows.append(metrics)

        if not monthly.empty:
            monthly["pred_col"] = "pred_mlp"
            monthly_frames.append(monthly)

    metrics_df = pd.DataFrame(metric_rows)
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    history_path = table_dir / "step08c_gpu_mlp_training_history.csv"
    metrics_path = table_dir / "step08c_gpu_mlp_metrics.csv"
    monthly_path = table_dir / "step08c_gpu_mlp_monthly_returns.csv"
    summary_path = table_dir / "step08c_gpu_mlp_summary.json"
    model_card_path = model_dir / "step08c_gpu_mlp_model_card.json"
    model_path = model_dir / "step08c_gpu_mlp_state_dict.pt"

    history.to_csv(history_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": feats,
            "preprocessor": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in prep.items()},
            "args": vars(args),
            "run_id": run_id,
        },
        model_path,
    )

    test_metrics = metrics_df.loc[metrics_df["sample"] == "test_2020_2024"].to_dict("records")

    summary = {
        "ok": bool(test_metrics),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "rows_test": int(len(test)),
        "features": feats,
        "n_features": int(len(feats)),
        "hidden": hidden,
        "dropout": float(args.dropout),
        "best_epoch": int(best["best_epoch"]),
        "best_valid_mean_ic": round(float(best["best_valid_mean_ic"]), 10),
        "test_metrics": test_metrics,
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "monthly_returns_path": str(monthly_path),
        "model_card_path": str(model_card_path),
        "model_state_dict_local_do_not_upload": str(model_path),
        "method_note": "GPU PyTorch MLP baseline. Train 2002-2016, validation 2017-2019, test 2020-2024. Compare against transparent residual signal baselines.",
    }

    write_json(summary_path, summary)
    write_json(model_card_path, summary)

    bundle = log_dir / f"step08c_gpu_mlp_monthly_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, model_card_path, history_path, metrics_path, monthly_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"MONTHLY_RETURNS={monthly_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
