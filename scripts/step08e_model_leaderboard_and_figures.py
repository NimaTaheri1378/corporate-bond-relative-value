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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "rank_group": "",
        "model_or_signal": "",
        "source_step": "",
        "sample": "test_2020_2024",
        "months": None,
        "mean_monthly_ic": None,
        "median_monthly_ic": None,
        "annualized_return": None,
        "annualized_vol": None,
        "sharpe": None,
        "cumulative_return": None,
        "max_drawdown": None,
        "positive_month_share": None,
        "mean_monthly_return": None,
        "mean_turnover": None,
        "notes": "",
    }
    base.update(kwargs)
    rows.append(base)


def build_leaderboard(root: Path, headline_cost_bps: float) -> pd.DataFrame:
    table_dir = root / "artifacts" / "tables"
    rows: list[dict[str, Any]] = []

    # Step 08B signal backtest: fully comparable net-of-cost portfolio metrics.
    signal_metrics = read_csv_if_exists(table_dir / "step08b_monthly_signal_backtest_metrics.csv")
    if not signal_metrics.empty:
        ret_col = f"long_short_ret_net_cost_{headline_cost_bps:g}bps"
        m = signal_metrics.loc[
            (signal_metrics["sample"].astype(str) == "test_2020_2024")
            & (signal_metrics["return_col"].astype(str) == ret_col)
        ].copy()

        for _, r in m.iterrows():
            add_row(
                rows,
                rank_group="transparent_signal_net",
                model_or_signal=str(r.get("signal")),
                source_step="08B",
                sample=str(r.get("sample")),
                months=int(r.get("months")) if pd.notna(r.get("months")) else None,
                mean_monthly_ic=float(r.get("mean_monthly_ic")) if pd.notna(r.get("mean_monthly_ic")) else None,
                median_monthly_ic=float(r.get("median_monthly_ic")) if pd.notna(r.get("median_monthly_ic")) else None,
                annualized_return=float(r.get("annualized_return_approx")) if pd.notna(r.get("annualized_return_approx")) else None,
                annualized_vol=float(r.get("annualized_vol")) if pd.notna(r.get("annualized_vol")) else None,
                sharpe=float(r.get("sharpe_approx")) if pd.notna(r.get("sharpe_approx")) else None,
                cumulative_return=float(r.get("cumulative_return")) if pd.notna(r.get("cumulative_return")) else None,
                max_drawdown=float(r.get("max_drawdown")) if pd.notna(r.get("max_drawdown")) else None,
                positive_month_share=float(r.get("positive_month_share")) if pd.notna(r.get("positive_month_share")) else None,
                mean_monthly_return=float(r.get("mean_monthly_return")) if pd.notna(r.get("mean_monthly_return")) else None,
                mean_turnover=float(r.get("mean_long_short_turnover")) if pd.notna(r.get("mean_long_short_turnover")) else None,
                notes=f"Net of {headline_cost_bps:g} bps turnover cost.",
            )

    # Step 08C GPU MLP.
    mlp_metrics = read_csv_if_exists(table_dir / "step08c_gpu_mlp_metrics.csv")
    if not mlp_metrics.empty:
        m = mlp_metrics.loc[mlp_metrics["sample"].astype(str) == "test_2020_2024"].copy()
        for _, r in m.iterrows():
            add_row(
                rows,
                rank_group="gpu_mlp_net",
                model_or_signal="gpu_mlp",
                source_step="08C",
                sample=str(r.get("sample")),
                months=int(r.get("months")) if pd.notna(r.get("months")) else None,
                mean_monthly_ic=float(r.get("mean_ic")) if pd.notna(r.get("mean_ic")) else None,
                median_monthly_ic=float(r.get("median_ic")) if pd.notna(r.get("median_ic")) else None,
                annualized_return=float(r.get("annualized_return_net_approx")) if pd.notna(r.get("annualized_return_net_approx")) else None,
                annualized_vol=float(r.get("annualized_vol_net")) if pd.notna(r.get("annualized_vol_net")) else None,
                sharpe=float(r.get("sharpe_net")) if pd.notna(r.get("sharpe_net")) else None,
                cumulative_return=float(r.get("cumulative_return_net")) if pd.notna(r.get("cumulative_return_net")) else None,
                max_drawdown=float(r.get("max_drawdown_net")) if pd.notna(r.get("max_drawdown_net")) else None,
                positive_month_share=float(r.get("positive_net_month_share")) if pd.notna(r.get("positive_net_month_share")) else None,
                mean_monthly_return=float(r.get("mean_long_short_ret_net")) if pd.notna(r.get("mean_long_short_ret_net")) else None,
                mean_turnover=float(r.get("mean_turnover")) if pd.notna(r.get("mean_turnover")) else None,
                notes=f"PyTorch GPU MLP, net of {headline_cost_bps:g} bps turnover cost.",
            )

    # Step 08D LightGBM.
    lgbm_metrics = read_csv_if_exists(table_dir / "step08d_lightgbm_metrics.csv")
    if not lgbm_metrics.empty:
        m = lgbm_metrics.loc[lgbm_metrics["sample"].astype(str) == "test_2020_2024"].copy()
        for _, r in m.iterrows():
            add_row(
                rows,
                rank_group="lightgbm_cpu_net",
                model_or_signal="lightgbm_cpu",
                source_step="08D",
                sample=str(r.get("sample")),
                months=int(r.get("months")) if pd.notna(r.get("months")) else None,
                mean_monthly_ic=float(r.get("mean_ic")) if pd.notna(r.get("mean_ic")) else None,
                median_monthly_ic=float(r.get("median_ic")) if pd.notna(r.get("median_ic")) else None,
                annualized_return=float(r.get("annualized_return_net_approx")) if pd.notna(r.get("annualized_return_net_approx")) else None,
                annualized_vol=float(r.get("annualized_vol_net")) if pd.notna(r.get("annualized_vol_net")) else None,
                sharpe=float(r.get("sharpe_net")) if pd.notna(r.get("sharpe_net")) else None,
                cumulative_return=float(r.get("cumulative_return_net")) if pd.notna(r.get("cumulative_return_net")) else None,
                max_drawdown=float(r.get("max_drawdown_net")) if pd.notna(r.get("max_drawdown_net")) else None,
                positive_month_share=float(r.get("positive_net_month_share")) if pd.notna(r.get("positive_net_month_share")) else None,
                mean_monthly_return=float(r.get("mean_long_short_ret_net")) if pd.notna(r.get("mean_long_short_ret_net")) else None,
                mean_turnover=float(r.get("mean_turnover")) if pd.notna(r.get("mean_turnover")) else None,
                notes=f"CPU LightGBM, net of {headline_cost_bps:g} bps turnover cost.",
            )

    # Step 08A Ridge: diagnostic only, because it was not run through the costed turnover engine.
    ridge_metrics = read_csv_if_exists(table_dir / "step08a_monthly_baseline_metrics.csv")
    if not ridge_metrics.empty:
        m = ridge_metrics.loc[
            (ridge_metrics["sample"].astype(str) == "test_2020_2024")
            & ridge_metrics["pred_col"].astype(str).str.contains("ridge_best", na=False)
        ].copy()

        for _, r in m.iterrows():
            mean_spread = r.get("mean_decile_spread_1m")
            add_row(
                rows,
                rank_group="ridge_diagnostic_no_cost",
                model_or_signal=str(r.get("pred_col")),
                source_step="08A",
                sample=str(r.get("sample")),
                months=int(r.get("months")) if pd.notna(r.get("months")) else None,
                mean_monthly_ic=float(r.get("mean_ic")) if pd.notna(r.get("mean_ic")) else None,
                median_monthly_ic=float(r.get("median_ic")) if pd.notna(r.get("median_ic")) else None,
                annualized_return=float(12.0 * mean_spread) if pd.notna(mean_spread) else None,
                annualized_vol=None,
                sharpe=float(r.get("spread_ir_annualized")) if pd.notna(r.get("spread_ir_annualized")) else None,
                cumulative_return=None,
                max_drawdown=None,
                positive_month_share=float(r.get("positive_ic_share")) if pd.notna(r.get("positive_ic_share")) else None,
                mean_monthly_return=float(mean_spread) if pd.notna(mean_spread) else None,
                mean_turnover=None,
                notes="Diagnostic ridge spread only; not costed through turnover engine.",
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("No model metrics found. Expected Step 08A/B/C/D artifact tables.")

    out = out.sort_values(["sample", "sharpe", "annualized_return"], ascending=[True, False, False]).reset_index(drop=True)
    out["leaderboard_rank"] = np.arange(1, len(out) + 1)
    return out


def load_monthly_series(root: Path, headline_cost_bps: float) -> pd.DataFrame:
    table_dir = root / "artifacts" / "tables"
    frames = []

    sig_path = table_dir / "step08b_monthly_signal_backtest_monthly_returns.csv"
    if sig_path.exists():
        s = pd.read_csv(sig_path)
        cost_col = f"long_short_ret_net_cost_{headline_cost_bps:g}bps"
        m = s.loc[
            (s["sample"].astype(str) == "test_2020_2024")
            & (s["signal"].astype(str) == "composite_residual_rank")
        ].copy()
        if cost_col in m.columns:
            frames.append(
                pd.DataFrame(
                    {
                        "signal_month": pd.to_datetime(m["signal_month"]),
                        "strategy": "composite_residual_rank",
                        "net_return": pd.to_numeric(m[cost_col], errors="coerce"),
                    }
                )
            )

    mlp_path = table_dir / "step08c_gpu_mlp_monthly_returns.csv"
    if mlp_path.exists():
        m = pd.read_csv(mlp_path)
        m = m.loc[m["sample"].astype(str) == "test_2020_2024"].copy()
        if "long_short_ret_net" in m.columns:
            frames.append(
                pd.DataFrame(
                    {
                        "signal_month": pd.to_datetime(m["signal_month"]),
                        "strategy": "gpu_mlp",
                        "net_return": pd.to_numeric(m["long_short_ret_net"], errors="coerce"),
                    }
                )
            )

    lgb_path = table_dir / "step08d_lightgbm_monthly_returns.csv"
    if lgb_path.exists():
        m = pd.read_csv(lgb_path)
        m = m.loc[m["sample"].astype(str) == "test_2020_2024"].copy()
        if "long_short_ret_net" in m.columns:
            frames.append(
                pd.DataFrame(
                    {
                        "signal_month": pd.to_datetime(m["signal_month"]),
                        "strategy": "lightgbm_cpu",
                        "net_return": pd.to_numeric(m["long_short_ret_net"], errors="coerce"),
                    }
                )
            )

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).dropna(subset=["signal_month", "net_return"])
    out = out.sort_values(["strategy", "signal_month"]).reset_index(drop=True)
    out["cumulative_return"] = out.groupby("strategy")["net_return"].transform(lambda s: (1.0 + s).cumprod() - 1.0)
    return out


def make_figures(root: Path, leaderboard: pd.DataFrame, monthly: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    fig_dir = root / "artifacts" / "figures_static"
    html_dir = root / "artifacts" / "figures_interactive"
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    top = leaderboard.loc[leaderboard["sample"] == "test_2020_2024"].copy()
    top = top.sort_values("sharpe", ascending=False).head(12)

    # Figure 1: Sharpe leaderboard.
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = top["model_or_signal"].astype(str).tolist()
    ax.barh(labels, top["sharpe"].astype(float))
    ax.invert_yaxis()
    ax.set_title("Test 2020–2024 Net Sharpe by Model / Signal")
    ax.set_xlabel("Net Sharpe, monthly long-short")
    fig.tight_layout()
    p = fig_dir / "step08e_test_net_sharpe_leaderboard.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    # Figure 2: return vs drawdown.
    dd = top.copy()
    dd = dd.loc[dd["max_drawdown"].notna()].copy()
    if not dd.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(dd["max_drawdown"].astype(float), dd["annualized_return"].astype(float))
        for _, r in dd.iterrows():
            ax.annotate(str(r["model_or_signal"])[:28], (float(r["max_drawdown"]), float(r["annualized_return"])), fontsize=8)
        ax.set_title("Test Return vs Drawdown")
        ax.set_xlabel("Max drawdown")
        ax.set_ylabel("Annualized net return")
        fig.tight_layout()
        p = fig_dir / "step08e_test_return_vs_drawdown.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # Figure 3: cumulative returns for main strategies.
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for strategy, g in monthly.groupby("strategy"):
            ax.plot(g["signal_month"], g["cumulative_return"], label=strategy)
        ax.set_title("Test 2020–2024 Cumulative Net Return")
        ax.set_xlabel("Signal month")
        ax.set_ylabel("Cumulative net return")
        ax.legend()
        fig.tight_layout()
        p = fig_dir / "step08e_test_cumulative_net_return.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # Optional interactive HTML if plotly is available.
    try:
        import plotly.express as px

        if not monthly.empty:
            fig_html = px.line(
                monthly,
                x="signal_month",
                y="cumulative_return",
                color="strategy",
                title="Test 2020–2024 Cumulative Net Return",
            )
            p = html_dir / "step08e_test_cumulative_net_return.html"
            fig_html.write_html(str(p), include_plotlyjs="cdn")
            paths.append(p)

        fig_html = px.bar(
            top.sort_values("sharpe", ascending=True),
            x="sharpe",
            y="model_or_signal",
            orientation="h",
            title="Test 2020–2024 Net Sharpe Leaderboard",
        )
        p = html_dir / "step08e_test_net_sharpe_leaderboard.html"
        fig_html.write_html(str(p), include_plotlyjs="cdn")
        paths.append(p)

    except Exception:
        pass

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 08E model leaderboard and visuals. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--headline-cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = build_leaderboard(root, headline_cost_bps=args.headline_cost_bps)
    monthly = load_monthly_series(root, headline_cost_bps=args.headline_cost_bps)

    leaderboard_path = table_dir / "step08e_model_leaderboard.csv"
    monthly_path = table_dir / "step08e_test_monthly_strategy_returns.csv"
    summary_path = table_dir / "step08e_model_leaderboard_summary.json"

    leaderboard.to_csv(leaderboard_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    fig_paths = make_figures(root, leaderboard, monthly)

    test_rows = leaderboard.loc[leaderboard["sample"] == "test_2020_2024"].copy()
    test_rows = test_rows.sort_values("sharpe", ascending=False)

    summary = {
        "ok": bool(not test_rows.empty),
        "run_id": run_id,
        "workspace": str(root),
        "headline_cost_bps": float(args.headline_cost_bps),
        "best_test_by_sharpe": test_rows.head(10).to_dict("records"),
        "headline_strategy": "composite_residual_rank",
        "headline_reason": "Best risk-adjusted out-of-sample net performance among transparent signals, Ridge, GPU MLP, and CPU LightGBM.",
        "leaderboard_path": str(leaderboard_path),
        "monthly_returns_path": str(monthly_path),
        "figure_paths": [str(p) for p in fig_paths],
        "note": "No WRDS. Consolidates Step 08A/B/C/D artifacts into model leaderboard and publication figures.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step08e_model_leaderboard_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, leaderboard_path, monthly_path, *fig_paths]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"LEADERBOARD={leaderboard_path}")
    print(f"MONTHLY_RETURNS={monthly_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
