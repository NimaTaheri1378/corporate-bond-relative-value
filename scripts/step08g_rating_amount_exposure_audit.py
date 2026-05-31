#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TARGET = "label_ret_1m"

RANK_COMPONENTS = [
    "last_residual_over_rmse",
    "last_residual_pctile",
    "last_issuer_date_residual_z",
    "last_bucket_residual_z",
]

SIGNALS = [
    "composite_residual_rank",
    "last_issuer_date_residual_z",
    "last_residual_over_rmse",
    "last_bucket_residual_z",
]

CHARACTERISTICS = [
    "mean_years_to_maturity",
    "last_years_to_maturity",
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
    "coupon",
    "amount_outstanding",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def normalize_id_value(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"<NA>", "NA", "NAN", "NONE", "NULL"}:
        return None
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_id_value).astype("string")


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
        raise FileNotFoundError(f"Missing model-eligible matrix: {path}")

    df = pq.ParquetFile(path).read().to_pandas()

    df["signal_month"] = pd.to_datetime(df["signal_month"], errors="coerce")
    df["label_month"] = pd.to_datetime(df["label_month"], errors="coerce")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "is_model_eligible_1m" in df.columns:
        df = df.loc[df["is_model_eligible_1m"].fillna(False).astype(bool)].copy()

    df = df.loc[df[TARGET].notna()].copy()
    df["issue_key"] = normalize_id_series(df["issue_key"])

    return df.reset_index(drop=True)


def add_residual_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rank_cols = []
    for col in RANK_COMPONENTS:
        if col not in out.columns:
            continue

        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        rc = f"_rank_{col}"
        out[rc] = out.groupby("signal_month", sort=False)[col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").rank(method="average", pct=True)
        )
        rank_cols.append(rc)

    if rank_cols:
        out["composite_residual_rank"] = out[rank_cols].mean(axis=1)

    for col in [
        "mean_years_to_maturity",
        "last_years_to_maturity",
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
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def load_static_issue_master(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "security_master_v1" / "fisd_issue_master.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    cols = [
        "issue_id",
        "complete_cusip",
        "issuer_id",
        "issuer_cusip",
        "coupon",
        "coupon_type",
        "bond_type",
        "security_level",
        "offering_amt",
        "principal_amt",
        "maturity",
        "offering_date",
        "rule_144a",
        "private_placement",
        "asset_backed",
        "convertible",
        "putable",
        "perpetual",
        "redeemable",
    ]

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in cols if c in available]
    df = pf.read(columns=selected).to_pandas()

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["issue_key"] = normalize_id_series(df["issue_id"])
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    df["principal_amt"] = pd.to_numeric(df["principal_amt"], errors="coerce")
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["offering_date"] = pd.to_datetime(df["offering_date"], errors="coerce")

    # One row per issue for static characteristics.
    df = df.sort_values(["issue_key", "maturity", "coupon"], na_position="last", kind="mergesort")
    df = df.dropna(subset=["issue_key"]).drop_duplicates("issue_key", keep="last").reset_index(drop=True)

    keep_cols = [
        "issue_key",
        "coupon",
        "coupon_type",
        "bond_type",
        "security_level",
        "offering_amt",
        "principal_amt",
        "maturity",
        "offering_date",
        "rule_144a",
        "private_placement",
        "asset_backed",
        "convertible",
        "putable",
        "perpetual",
        "redeemable",
    ]
    return df.loc[:, keep_cols].copy()


def rating_bucket_one(x: Any) -> str:
    if x is None or pd.isna(x):
        return "missing"

    s = str(x).strip().upper()
    s = s.replace(" ", "").replace("_", "").replace("/", "")

    if not s or s in {"<NA>", "NA", "NAN", "NONE", "NULL", "NR", "NOTRATED"}:
        return "missing"

    # Default / distressed.
    if s.startswith(("D", "SD", "RD")):
        return "default"
    if s.startswith(("CCC", "CC", "C")) or s.startswith(("CAA", "CA")):
        return "ccc_or_lower"

    # High yield.
    if s.startswith(("BB", "BA")):
        return "bb"
    if s.startswith("B") and not s.startswith(("BAA", "BBB")):
        return "b"

    # Investment grade.
    if s.startswith(("BBB", "BAA")):
        return "bbb"
    if s.startswith(("A", "AA", "AAA")):
        return "a_or_better"

    return "other"


def load_rating_history(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "security_master_v1" / "fisd_rating_history.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    cols = ["issue_id", "rating", "rating_date", "rating_status", "rating_type"]

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in cols if c in available]
    df = pf.read(columns=selected).to_pandas()

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["issue_key"] = normalize_id_series(df["issue_id"])
    df["event_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    df["rating"] = df["rating"].astype("string")
    df["rating_bucket"] = df["rating"].map(rating_bucket_one).astype("string")

    df = df.dropna(subset=["issue_key", "event_date"]).copy()
    df = df.sort_values(["issue_key", "event_date", "rating"], na_position="last", kind="mergesort")
    df = df.drop_duplicates(["issue_key", "event_date"], keep="last").reset_index(drop=True)

    return df.loc[:, ["issue_key", "event_date", "rating", "rating_bucket", "rating_status", "rating_type"]].copy()


def load_amount_history(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "security_master_v1" / "fisd_amount_outstanding_history.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    cols = ["issue_id", "effective_date", "amount_outstanding", "action_amount", "action_type"]

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    selected = [c for c in cols if c in available]
    df = pf.read(columns=selected).to_pandas()

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["issue_key"] = normalize_id_series(df["issue_id"])
    df["event_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["amount_outstanding"] = pd.to_numeric(df["amount_outstanding"], errors="coerce")
    df["action_amount"] = pd.to_numeric(df["action_amount"], errors="coerce")

    df = df.dropna(subset=["issue_key", "event_date"]).copy()
    df = df.sort_values(["issue_key", "event_date"], na_position="last", kind="mergesort")
    df = df.drop_duplicates(["issue_key", "event_date"], keep="last").reset_index(drop=True)

    return df.loc[:, ["issue_key", "event_date", "amount_outstanding", "action_amount", "action_type"]].copy()


def asof_enrich_monthly(
    matrix: pd.DataFrame,
    history: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    left = matrix.loc[:, ["issue_key", "signal_month"]].drop_duplicates().reset_index(drop=True)
    left["row_id"] = np.arange(len(left))
    left["event_date"] = left["signal_month"]
    left["_is_left"] = 1

    hist = history.loc[:, ["issue_key", "event_date", *value_cols]].copy()
    hist["_is_left"] = 0
    hist["row_id"] = -1

    events = pd.concat(
        [
            left.loc[:, ["issue_key", "event_date", "row_id", "_is_left"]],
            hist.loc[:, ["issue_key", "event_date", "row_id", "_is_left", *value_cols]],
        ],
        ignore_index=True,
        sort=False,
    )

    # Historical events first on same date, then signal row.
    events = events.sort_values(["issue_key", "event_date", "_is_left"], kind="mergesort")

    for c in value_cols:
        events[c] = events.groupby("issue_key", sort=False)[c].ffill()

    enriched = events.loc[events["_is_left"].eq(1), ["row_id", *value_cols]].copy()
    enriched = enriched.sort_values("row_id").reset_index(drop=True)
    enriched = enriched.rename(columns={c: f"{prefix}_{c}" for c in value_cols})

    out = left.loc[:, ["issue_key", "signal_month", "row_id"]].merge(enriched, on="row_id", how="left")
    return out.drop(columns=["row_id"])


def amount_bucket_by_month(df: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(df["amount_outstanding"], errors="coerce")

    def qcut3(x: pd.Series) -> pd.Series:
        z = pd.to_numeric(x, errors="coerce")
        if z.notna().sum() < 100 or z.nunique(dropna=True) < 3:
            return pd.Series(pd.NA, index=x.index, dtype="string")
        try:
            return pd.qcut(z, q=3, labels=["small_amt", "mid_amt", "large_amt"], duplicates="drop").astype("string")
        except Exception:
            return pd.Series(pd.NA, index=x.index, dtype="string")

    return amount.groupby(df["signal_month"], sort=False).transform(qcut3)


def add_issue_exposures(df: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    static = load_static_issue_master(root)
    ratings = load_rating_history(root)
    amounts = load_amount_history(root)

    out = df.merge(static, on="issue_key", how="left", copy=False)

    rating_asof = asof_enrich_monthly(
        out,
        ratings,
        value_cols=["rating", "rating_bucket", "rating_status", "rating_type"],
        prefix="asof",
    )

    amount_asof = asof_enrich_monthly(
        out,
        amounts,
        value_cols=["amount_outstanding", "action_amount", "action_type"],
        prefix="asof",
    )

    out = out.merge(rating_asof, on=["issue_key", "signal_month"], how="left", copy=False)
    out = out.merge(amount_asof, on=["issue_key", "signal_month"], how="left", copy=False)

    out["rating_bucket"] = out["asof_rating_bucket"].fillna("missing").astype("string")
    out["amount_outstanding"] = pd.to_numeric(out["asof_amount_outstanding"], errors="coerce")
    out["amount_bucket"] = amount_bucket_by_month(out)

    meta = {
        "static_issue_rows": int(len(static)),
        "rating_history_rows": int(len(ratings)),
        "amount_history_rows": int(len(amounts)),
        "rows": int(len(out)),
        "rating_non_missing_rows": int(out["asof_rating"].notna().sum()),
        "rating_coverage_pct": round(100.0 * out["asof_rating"].notna().mean(), 6),
        "amount_non_missing_rows": int(out["amount_outstanding"].notna().sum()),
        "amount_coverage_pct": round(100.0 * out["amount_outstanding"].notna().mean(), 6),
        "coupon_non_missing_rows": int(out["coupon"].notna().sum()),
        "coupon_coverage_pct": round(100.0 * out["coupon"].notna().mean(), 6),
    }

    return out, meta


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_2002_2016": df.loc[df["signal_year"].between(2002, 2016)].copy(),
        "valid_2017_2019": df.loc[df["signal_year"].between(2017, 2019)].copy(),
        "test_2020_2024": df.loc[df["signal_year"].between(2020, 2024)].copy(),
        "all_2002_2024": df.loc[df["signal_year"].between(2002, 2024)].copy(),
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


def select_tails(g: pd.DataFrame, signal: str, tail_frac: float, min_obs: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    x = g.copy()
    x[signal] = pd.to_numeric(x[signal], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x[TARGET] = pd.to_numeric(x[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=[signal, TARGET, "issue_key"])

    if len(x) < min_obs:
        return None, None

    lo = x[signal].quantile(tail_frac)
    hi = x[signal].quantile(1.0 - tail_frac)

    short = x.loc[x[signal] <= lo].copy()
    long = x.loc[x[signal] >= hi].copy()

    if long.empty or short.empty:
        return None, None

    return long, short


def select_bucket_neutral_tails(
    g: pd.DataFrame,
    signal: str,
    bucket_col: str,
    tail_frac: float,
    min_obs_total: int,
    min_bucket_obs: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if bucket_col not in g.columns:
        return None, None

    long_parts = []
    short_parts = []

    for _, b in g.dropna(subset=[bucket_col]).groupby(bucket_col, sort=False):
        long, short = select_tails(b, signal=signal, tail_frac=tail_frac, min_obs=min_bucket_obs)
        if long is not None and short is not None:
            long_parts.append(long)
            short_parts.append(short)

    if not long_parts or not short_parts:
        return None, None

    long_all = pd.concat(long_parts, ignore_index=True)
    short_all = pd.concat(short_parts, ignore_index=True)

    if len(long_all) < min_obs_total * tail_frac or len(short_all) < min_obs_total * tail_frac:
        return None, None

    return long_all, short_all


def leg_characteristics(frame: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}

    for col in CHARACTERISTICS:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{col}_mean"] = None if s.dropna().empty else float(s.mean())

    return out


def bucket_shares(frame: pd.DataFrame, col: str, prefix: str) -> dict[str, float]:
    if col not in frame.columns or frame.empty:
        return {}

    vc = frame[col].astype("string").fillna("missing").value_counts(normalize=True)
    return {f"{prefix}_{str(k)}_share": float(v) for k, v in vc.items()}


def run_one_backtest(
    df: pd.DataFrame,
    sample: str,
    signal: str,
    construction: str,
    bucket_col: str | None,
    tail_frac: float,
    cost_bps: float,
    min_month_obs: int,
    min_bucket_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    exposure_rows = []

    prev_long: set[str] | None = None
    prev_short: set[str] | None = None

    for month, g in df.groupby("signal_month", sort=True):
        if construction == "global":
            long, short = select_tails(g, signal=signal, tail_frac=tail_frac, min_obs=min_month_obs)
        else:
            long, short = select_bucket_neutral_tails(
                g,
                signal=signal,
                bucket_col=str(bucket_col),
                tail_frac=tail_frac,
                min_obs_total=min_month_obs,
                min_bucket_obs=min_bucket_obs,
            )

        if long is None or short is None:
            continue

        long_set = set(long["issue_key"].astype(str))
        short_set = set(short["issue_key"].astype(str))

        long_turn = turnover(prev_long, long_set)
        short_turn = turnover(prev_short, short_set)
        ls_turn = long_turn + short_turn

        long_ret = float(long[TARGET].mean())
        short_ret = float(short[TARGET].mean())
        gross = long_ret - short_ret
        net = gross - (cost_bps / 10_000.0) * ls_turn

        monthly_rows.append(
            {
                "sample": sample,
                "signal": signal,
                "construction": construction,
                "bucket_col": bucket_col or "",
                "signal_month": pd.to_datetime(month),
                "n_obs_month": int(g[[signal, TARGET]].dropna().shape[0]),
                "n_long": int(len(long)),
                "n_short": int(len(short)),
                "long_ret": long_ret,
                "short_leg_ret": short_ret,
                "long_short_ret_gross": gross,
                "long_short_ret_net": net,
                "long_turnover": long_turn,
                "short_turnover": short_turn,
                "long_short_turnover": ls_turn,
                "spearman_ic": safe_spearman(g[signal], g[TARGET]),
            }
        )

        exp = {
            "sample": sample,
            "signal": signal,
            "construction": construction,
            "bucket_col": bucket_col or "",
            "signal_month": pd.to_datetime(month),
            "n_long": int(len(long)),
            "n_short": int(len(short)),
        }

        long_chars = leg_characteristics(long)
        short_chars = leg_characteristics(short)

        for k, v in long_chars.items():
            exp[f"long_{k}"] = v
        for k, v in short_chars.items():
            exp[f"short_{k}"] = v

        for col in CHARACTERISTICS:
            lk = f"long_{col}_mean"
            sk = f"short_{col}_mean"
            if lk in exp and sk in exp and exp[lk] is not None and exp[sk] is not None:
                exp[f"delta_{col}_mean"] = exp[lk] - exp[sk]

        exp.update(bucket_shares(long, "rating_bucket", "long_rating"))
        exp.update(bucket_shares(short, "rating_bucket", "short_rating"))
        exp.update(bucket_shares(long, "amount_bucket", "long_amount"))
        exp.update(bucket_shares(short, "amount_bucket", "short_amount"))

        exposure_rows.append(exp)

        prev_long = long_set
        prev_short = short_set

    return pd.DataFrame(monthly_rows), pd.DataFrame(exposure_rows)


def summarize_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if monthly.empty:
        return pd.DataFrame()

    for keys, g in monthly.groupby(["sample", "signal", "construction", "bucket_col"], dropna=False):
        sample, signal, construction, bucket_col = keys
        r = pd.to_numeric(g["long_short_ret_net"], errors="coerce").dropna()
        rg = pd.to_numeric(g["long_short_ret_gross"], errors="coerce").dropna()
        ic = pd.to_numeric(g["spearman_ic"], errors="coerce").dropna()

        if r.empty:
            continue

        cumulative = (1.0 + r).cumprod()
        dd = cumulative / cumulative.cummax() - 1.0

        rows.append(
            {
                "sample": sample,
                "signal": signal,
                "construction": construction,
                "bucket_col": bucket_col,
                "months": int(len(r)),
                "mean_monthly_ic": None if ic.empty else round(float(ic.mean()), 10),
                "median_monthly_ic": None if ic.empty else round(float(ic.median()), 10),
                "positive_ic_share": None if ic.empty else round(float((ic > 0).mean()), 6),
                "mean_monthly_return_gross": None if rg.empty else round(float(rg.mean()), 10),
                "mean_monthly_return_net": round(float(r.mean()), 10),
                "annualized_return_net_approx": round(float(12.0 * r.mean()), 10),
                "annualized_vol_net": round(float(np.sqrt(12.0) * r.std(ddof=1)), 10) if len(r) > 1 else 0.0,
                "sharpe_net": None if len(r) <= 1 or r.std(ddof=1) == 0 else round(float(np.sqrt(12.0) * r.mean() / r.std(ddof=1)), 6),
                "cumulative_return_net": round(float(cumulative.iloc[-1] - 1.0), 10),
                "max_drawdown_net": round(float(dd.min()), 10),
                "positive_net_month_share": round(float((r > 0).mean()), 6),
                "mean_turnover": round(float(g["long_short_turnover"].mean()), 10),
                "mean_n_long": round(float(g["n_long"].mean()), 3),
                "mean_n_short": round(float(g["n_short"].mean()), 3),
            }
        )

    return pd.DataFrame(rows)


def summarize_exposures(exposure: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty:
        return pd.DataFrame()

    numeric_cols = [c for c in exposure.columns if c.startswith("delta_")]
    share_cols = [c for c in exposure.columns if c.startswith("long_rating_") or c.startswith("short_rating_") or c.startswith("long_amount_") or c.startswith("short_amount_")]

    rows = []
    for keys, g in exposure.groupby(["sample", "signal", "construction", "bucket_col"], dropna=False):
        sample, signal, construction, bucket_col = keys
        rec = {
            "sample": sample,
            "signal": signal,
            "construction": construction,
            "bucket_col": bucket_col,
            "months": int(g["signal_month"].nunique()),
        }

        for col in numeric_cols:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if not s.empty:
                rec[f"{col}_avg"] = round(float(s.mean()), 10)

        for col in share_cols:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if not s.empty:
                rec[f"{col}_avg"] = round(float(s.mean()), 10)

        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 08G rating and amount exposure audit. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--tail-frac", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--min-month-obs", type=int, default=100)
    parser.add_argument("--min-bucket-obs", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"loading_matrix={args.universe}", flush=True)

    df = add_residual_signals(load_matrix(root, args.universe))

    print(f"matrix_rows={len(df):,}")
    print("loading_asof_rating_amount_exposures", flush=True)

    df, exposure_meta = add_issue_exposures(df, root)

    signal_cols = [s for s in SIGNALS if s in df.columns]
    if not signal_cols:
        raise RuntimeError("No requested signal columns available.")

    splits = split_frame(df)

    jobs = []
    for sample_name, frame in splits.items():
        for signal in signal_cols:
            jobs.append((sample_name, frame, signal, "global", None))
            jobs.append((sample_name, frame, signal, "rating_neutral", "rating_bucket"))
            jobs.append((sample_name, frame, signal, "amount_neutral", "amount_bucket"))

    all_monthly = []
    all_exposure = []

    print(f"signals={signal_cols}")
    print(f"jobs={len(jobs)}")

    for sample_name, frame, signal, construction, bucket_col in jobs:
        monthly, exposure = run_one_backtest(
            df=frame,
            sample=sample_name,
            signal=signal,
            construction=construction,
            bucket_col=bucket_col,
            tail_frac=args.tail_frac,
            cost_bps=args.cost_bps,
            min_month_obs=args.min_month_obs,
            min_bucket_obs=args.min_bucket_obs,
        )

        if not monthly.empty:
            all_monthly.append(monthly)
        if not exposure.empty:
            all_exposure.append(exposure)

    monthly_all = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
    exposure_all = pd.concat(all_exposure, ignore_index=True) if all_exposure else pd.DataFrame()

    metrics = summarize_returns(monthly_all)
    exposure_summary = summarize_exposures(exposure_all)

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = table_dir / "step08g_rating_amount_exposure_monthly_returns.csv"
    metrics_path = table_dir / "step08g_rating_amount_exposure_metrics.csv"
    exposure_path = table_dir / "step08g_rating_amount_exposure_detail.csv"
    exposure_summary_path = table_dir / "step08g_rating_amount_exposure_summary.csv"
    summary_path = table_dir / "step08g_rating_amount_exposure_audit_summary.json"

    monthly_all.to_csv(monthly_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    exposure_all.to_csv(exposure_path, index=False)
    exposure_summary.to_csv(exposure_summary_path, index=False)

    test = metrics.loc[metrics["sample"] == "test_2020_2024"].copy()
    best_test = test.sort_values("sharpe_net", ascending=False).head(20) if not test.empty else pd.DataFrame()

    headline = test.loc[
        (test["signal"] == "composite_residual_rank")
        & (test["construction"].isin(["global", "rating_neutral", "amount_neutral"]))
    ].sort_values("construction")

    headline_exposure = exposure_summary.loc[
        (exposure_summary["sample"] == "test_2020_2024")
        & (exposure_summary["signal"] == "composite_residual_rank")
        & (exposure_summary["construction"].isin(["global", "rating_neutral", "amount_neutral"]))
    ].sort_values("construction")

    summary = {
        "ok": bool(not metrics.empty and not best_test.empty),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "tail_frac": float(args.tail_frac),
        "cost_bps": float(args.cost_bps),
        "signals": signal_cols,
        "rows_loaded": int(len(df)),
        "asof_exposure_meta": exposure_meta,
        "best_test_rows": best_test.to_dict("records"),
        "headline_composite_rating_amount_robustness": headline.to_dict("records"),
        "headline_composite_exposure_summary": headline_exposure.to_dict("records"),
        "metrics_path": str(metrics_path),
        "monthly_returns_path": str(monthly_path),
        "exposure_summary_path": str(exposure_summary_path),
        "exposure_detail_path": str(exposure_path),
        "method_note": "Robustness check: global, as-of rating-bucket-neutral, and amount-outstanding-bucket-neutral portfolios. No WRDS.",
    }

    write_json(summary_path, summary)

    bundle = log_dir / f"step08g_rating_amount_exposure_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [summary_path, metrics_path, monthly_path, exposure_summary_path, exposure_path]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={metrics_path}")
    print(f"EXPOSURE_SUMMARY={exposure_summary_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
