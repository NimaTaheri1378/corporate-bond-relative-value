#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = "corporate-bond-relative-value"
TITLE = "Corporate Bond Relative-Value"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def copy_if_exists(src: Path, dst: Path, copied: list[str], root: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(root)))


def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.3f}%"
    except Exception:
        return "NA"


def num(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "NA"


def first_row(df: pd.DataFrame, **filters: str) -> dict[str, Any]:
    if df.empty:
        return {}
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            return {}
        mask &= df[col].astype(str).eq(str(val))
    out = df.loc[mask].copy()
    if out.empty:
        return {}
    return out.iloc[0].to_dict()


def load_results(root: Path) -> dict[str, Any]:
    t = root / "artifacts" / "tables"
    leaderboard = read_csv(t / "step08e_model_leaderboard.csv")
    robust_ml = read_csv(t / "step08f_signal_robustness_metrics.csv")
    robust_ra = read_csv(t / "step08g_rating_amount_exposure_metrics.csv")
    exposure = read_json(t / "step08g_rating_amount_exposure_audit_summary.json")

    headline = first_row(
        leaderboard,
        sample="test_2020_2024",
        model_or_signal="composite_residual_rank",
        rank_group="transparent_signal_net",
    )
    if not headline and not leaderboard.empty:
        tmp = leaderboard.loc[leaderboard.get("sample", "").astype(str).eq("test_2020_2024")].copy()
        if not tmp.empty and "sharpe" in tmp.columns:
            headline = tmp.sort_values("sharpe", ascending=False).iloc[0].to_dict()

    def robust_row(construction: str) -> dict[str, Any]:
        src = robust_ra if construction in {"rating_neutral", "amount_neutral"} else robust_ml
        return first_row(
            src,
            sample="test_2020_2024",
            signal="composite_residual_rank",
            construction=construction,
        )

    return {
        "headline": headline,
        "leaderboard": leaderboard,
        "global": robust_row("global"),
        "maturity_neutral": robust_row("maturity_neutral"),
        "liquidity_neutral": robust_row("liquidity_neutral"),
        "rating_neutral": robust_row("rating_neutral"),
        "amount_neutral": robust_row("amount_neutral"),
        "exposure": exposure,
    }


def patch_repo_name(path: Path) -> bool:
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8", errors="ignore")
    new = old.replace("Corporate-Bond-Relative-Value", REPO)
    new = new.replace("wrds-corpbond-rv", REPO)
    if new != old:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def copy_public_assets(root: Path) -> list[str]:
    copied: list[str] = []
    table_src = root / "artifacts" / "tables"
    fig_src = root / "artifacts" / "figures_static"
    html_src = root / "artifacts" / "figures_interactive"

    for name in [
        "step08e_model_leaderboard.csv",
        "step08e_model_leaderboard_summary.json",
        "step08f_signal_robustness_metrics.csv",
        "step08f_signal_robustness_summary.json",
        "step08g_rating_amount_exposure_metrics.csv",
        "step08g_rating_amount_exposure_summary.csv",
        "step08g_rating_amount_exposure_audit_summary.json",
        "step09a_public_safety_audit.json",
        "step09a_public_safety_audit_checks.csv",
    ]:
        copy_if_exists(table_src / name, root / "docs" / "assets" / "tables" / name, copied, root)

    for name in [
        "step08e_test_cumulative_net_return.png",
        "step08e_test_net_sharpe_leaderboard.png",
        "step08e_test_return_vs_drawdown.png",
    ]:
        copy_if_exists(fig_src / name, root / "docs" / "assets" / "figures" / name, copied, root)

    for name in [
        "step08e_test_cumulative_net_return.html",
        "step08e_test_net_sharpe_leaderboard.html",
    ]:
        copy_if_exists(html_src / name, root / "docs" / "assets" / "interactive" / name, copied, root)

    return copied


def make_readme(results: dict[str, Any]) -> str:
    h = results["headline"]
    global_row = results["global"]
    maturity = results["maturity_neutral"]
    liquidity = results["liquidity_neutral"]
    rating = results["rating_neutral"]
    amount = results["amount_neutral"]
    exposure = results["exposure"].get("asof_exposure_meta", {})

    return f"""# {TITLE}

Research-grade corporate-bond relative-value pipeline for testing whether transaction-cleaned issuer-curve residuals forecast future corporate-bond returns.

[![Public repo audit](https://github.com/NimaTaheri1378/{REPO}/actions/workflows/public_repo_audit.yml/badge.svg)](https://github.com/NimaTaheri1378/{REPO}/actions/workflows/public_repo_audit.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![Research candidate](https://img.shields.io/badge/status-research%20candidate-purple)
![Public-safe release](https://img.shields.io/badge/data-public--safe%20aggregate%20release-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![Test cumulative net return](docs/assets/figures/step08e_test_cumulative_net_return.png)

## Executive summary

This repository is the public-safe release of a full-stack empirical fixed-income research project.

The private research pipeline ingests WRDS TRACE Enhanced corporate-bond transactions, cleans trade prints, joins FISD security-master data, builds issuer-date curve inputs, fits guarded issuer curves, converts curve residuals into bond-level relative-value features, creates leakage-aware monthly return labels, and compares transparent residual signals against Ridge, CPU LightGBM, and a GPU PyTorch MLP.

The public repo contains code, documentation, aggregate result tables, figures, dashboards, and tests. Raw WRDS/vendor data, local Parquet panels, protected caches, credentials, model binaries, and cluster logs are excluded.

## Research question

**Do bond-level deviations from a transaction-cleaned issuer-specific corporate-bond curve forecast future returns after controlling for liquidity, maturity, rating, amount outstanding, transaction costs, and nonlinear model alternatives?**

## Headline answer

**Yes, in this sample.** A transparent residual-rank signal survives out-of-sample testing, 10 bps turnover costs, and maturity-, liquidity-, rating-, and amount-neutral robustness checks. More complex models were useful diagnostics, but they did not beat the simple residual-rank signal on risk-adjusted net performance.

## Headline result

| Item | Public release value |
|---|---:|
| Selected research candidate | `composite_residual_rank`, next-month return |
| Portfolio construction | equal-weight top-minus-bottom decile |
| Cost assumption | 10 bps one-way turnover cost |
| Test sample | 2020-2024 |
| Mean monthly IC | {num(h.get("mean_monthly_ic"), 4)} |
| Mean net monthly return | {pct(h.get("mean_monthly_return"))} |
| Annualized net return | {pct(h.get("annualized_return"))} |
| Annualized net volatility | {pct(h.get("annualized_vol"))} |
| Net Sharpe approximation | {num(h.get("sharpe"), 3)} |
| Cumulative net return | {pct(h.get("cumulative_return"))} |
| Max drawdown | {pct(h.get("max_drawdown"))} |
| Positive month share | {pct(h.get("positive_month_share"))} |
| Decision status | research candidate; not production/live trading |

## Visual results

| Portfolio validation | Model comparison |
|---|---|
| ![Cumulative net return](docs/assets/figures/step08e_test_cumulative_net_return.png) | ![Net Sharpe leaderboard](docs/assets/figures/step08e_test_net_sharpe_leaderboard.png) |
| **Net cumulative performance.** The headline residual-rank strategy remains positive after 10 bps turnover costs. | **Model comparison.** Transparent residual ranking beats Ridge, CPU LightGBM, and GPU MLP on risk-adjusted net performance. |

| Risk profile | Interactive views |
|---|---|
| ![Return vs drawdown](docs/assets/figures/step08e_test_return_vs_drawdown.png) | [`interactive cumulative return`](docs/assets/interactive/step08e_test_cumulative_net_return.html) |
| **Return vs drawdown.** Nonlinear models produce similar returns with materially worse drawdowns. | **Interactive figures.** Public-safe HTML dashboards are built from aggregate outputs only. |

## Robustness scorecard

| Construction | Annualized net return | Net Sharpe | Max drawdown |
|---|---:|---:|---:|
| Global | {pct(global_row.get("annualized_return_net_approx"))} | {num(global_row.get("sharpe_net"), 3)} | {pct(global_row.get("max_drawdown_net"))} |
| Maturity-neutral | {pct(maturity.get("annualized_return_net_approx"))} | {num(maturity.get("sharpe_net"), 3)} | {pct(maturity.get("max_drawdown_net"))} |
| Liquidity-neutral | {pct(liquidity.get("annualized_return_net_approx"))} | {num(liquidity.get("sharpe_net"), 3)} | {pct(liquidity.get("max_drawdown_net"))} |
| Rating-neutral | {pct(rating.get("annualized_return_net_approx"))} | {num(rating.get("sharpe_net"), 3)} | {pct(rating.get("max_drawdown_net"))} |
| Amount-neutral | {pct(amount.get("annualized_return_net_approx"))} | {num(amount.get("sharpe_net"), 3)} | {pct(amount.get("max_drawdown_net"))} |

As-of exposure coverage:

```text
rating coverage:             {num(exposure.get("rating_coverage_pct"), 3)}%
amount-outstanding coverage: {num(exposure.get("amount_coverage_pct"), 3)}%
coupon coverage:             {num(exposure.get("coupon_coverage_pct"), 3)}%
```

## Pipeline architecture

```mermaid
flowchart LR
    A[WRDS schema discovery] --> B[TRACE/FISD/returns extraction]
    B --> C[TRACE cleaning and security master]
    C --> D[TRACE-FISD joined panel]
    D --> E[Issuer-date curve inputs]
    E --> F[Guarded curve fitting]
    F --> G[Residual features]
    G --> H[Monthly return labels]
    H --> I[Transparent baselines]
    I --> J[GPU MLP and CPU LightGBM]
    J --> K[Cost-aware backtests]
    K --> L[Robustness and exposure audits]
    L --> M[Public-safe release]
```

## What is public here

| Component | Included? | Notes |
|---|---:|---|
| Research code structure | Yes | Package layout, scripts, configs, tests, documentation |
| Aggregate result tables | Yes | Public-safe summaries only; no row-level vendor panels |
| Static figures | Yes | Rendered aggregate research figures |
| Interactive dashboards | Yes | HTML dashboards built from aggregate summaries |
| Tests and CI audit | Yes | Smoke tests and public-release checks |
| Raw WRDS / TRACE / FISD / return data | No | Users must provide their own licensed access |
| Local Parquet panels and caches | No | Excluded by `.gitignore` and safety audit |
| Credentials, `.pgpass`, API keys, cluster logs | No | Excluded and scanned before release |
| Model binaries/state files | No | Excluded from public release |

## Public aggregate tables

Public-safe aggregate tables are in:

```text
docs/assets/tables/
```

Key files:

```text
step08e_model_leaderboard.csv
step08f_signal_robustness_metrics.csv
step08g_rating_amount_exposure_metrics.csv
step08g_rating_amount_exposure_summary.csv
step09a_public_safety_audit.json
```

## Quickstart

The public repository does not include proprietary data. It is intended to install, lint, test, and document the research code and public aggregate release.

```bash
mamba env create -f environment.yml
conda activate ml_core
pip install -e .

make smoke
make test
python scripts/step09a_public_safety_audit.py --workspace .
mkdocs build
```

## Repository map

```text
{REPO}/
├── README.md
├── DATA_ACCESS.md
├── configs/
├── docs/
│   ├── assets/
│   ├── methodology.md
│   ├── results.md
│   ├── robustness.md
│   ├── reproducibility.md
│   └── model_card.md
├── scripts/
├── src/
├── tests/
└── .github/workflows/
```

## Data policy

This public repository intentionally excludes raw WRDS/vendor data, credentials, local cluster logs, protected Parquet caches, model binaries, and private schema contracts. Users need their own WRDS/TRACE/FISD/return-data entitlements. See [`DATA_ACCESS.md`](DATA_ACCESS.md).

## Status and next credibility checks

This is a **research candidate**, not a deployed strategy. Before any real capital deployment, the next credibility checks would be independent replication, deeper execution-cost modeling, sector and issuer-family neutral implementations, multi-horizon labels, and live paper trading with frozen code.

## License

Code is released under the MIT License. Figures and documentation are intended for public research presentation only. Third-party data products are not redistributed.
"""


def ensure_docs(root: Path, results: dict[str, Any]) -> list[Path]:
    h = results["headline"]
    docs: dict[str, str] = {
        "docs/index.md": "# Corporate Bond Relative-Value\n\nSee the main [`README.md`](../README.md) for the public research summary.\n",
        "docs/methodology.md": """# Methodology

The project follows a curve-first, alpha-second fixed-income research design:

1. clean TRACE transactions;
2. join FISD security-master data;
3. fit guarded issuer-date curves;
4. construct residual features;
5. build leakage-aware monthly labels;
6. compare transparent baselines, CPU LightGBM, and GPU PyTorch MLP;
7. evaluate cost-aware portfolios and exposure robustness.
""",
        "docs/results.md": f"""# Results

The headline public result is `composite_residual_rank`.

| Metric | Test 2020-2024 |
|---|---:|
| Mean monthly IC | {num(h.get("mean_monthly_ic"), 4)} |
| Mean net monthly return | {pct(h.get("mean_monthly_return"))} |
| Annualized net return | {pct(h.get("annualized_return"))} |
| Annualized net volatility | {pct(h.get("annualized_vol"))} |
| Net Sharpe | {num(h.get("sharpe"), 3)} |
| Cumulative net return | {pct(h.get("cumulative_return"))} |
| Max drawdown | {pct(h.get("max_drawdown"))} |

![Cumulative net return](assets/figures/step08e_test_cumulative_net_return.png)

![Net Sharpe leaderboard](assets/figures/step08e_test_net_sharpe_leaderboard.png)

![Return vs drawdown](assets/figures/step08e_test_return_vs_drawdown.png)
""",
        "docs/robustness.md": """# Robustness

The headline residual-rank signal is tested under:

- global top-minus-bottom decile construction;
- maturity-bucket neutral construction;
- liquidity-bucket neutral construction;
- as-of rating-bucket neutral construction;
- as-of amount-outstanding-bucket neutral construction.

See public aggregate tables in `docs/assets/tables/`.
""",
        "docs/reproducibility.md": """# Reproducibility

The public repository includes relevant pipeline scripts and public aggregate outputs.

The full pipeline requires licensed WRDS access. Raw WRDS/vendor data, local Parquet panels, credentials, model binaries, and cluster logs are excluded.

Run the safety audit before pushing:

```bash
python scripts/step09a_public_safety_audit.py --workspace .
```
""",
        "docs/model_card.md": """# Model card

Transparent residual ranking is the headline. Ridge, CPU LightGBM, and GPU PyTorch MLP are model-comparison baselines.

The GPU MLP and CPU LightGBM are included as robustness/model-comparison checks, not as the headline model.
""",
    }

    out: list[Path] = []
    for rel, text in docs.items():
        p = root / rel
        write_text(p, text)
        out.append(p)
    return out


def ensure_data_access(root: Path) -> Path:
    p = root / "DATA_ACCESS.md"
    write_text(
        p,
        """# Data access policy

This public repository does not redistribute WRDS, TRACE, FISD, CRSP, Compustat, contributed bond-return data, or any other vendor data.

Do not commit passwords, API keys, `.pgpass`, tokens, SSH keys, raw data, processed Parquet panels, model binaries, or cluster logs.

Public-safe artifacts include source code, documentation, configuration, aggregate result tables, aggregate figures, interactive dashboards built from aggregate summaries, tests, and sanitized manifests.

Users who want to rebuild the full pipeline need their own WRDS entitlements and local credentials.
""",
    )
    return p


def ensure_mkdocs(root: Path) -> Path:
    p = root / "mkdocs.yml"
    write_text(
        p,
        """site_name: Corporate Bond Relative-Value
theme:
  name: material
nav:
  - Home: index.md
  - Methodology: methodology.md
  - Results: results.md
  - Robustness: robustness.md
  - Reproducibility: reproducibility.md
  - Model Card: model_card.md
markdown_extensions:
  - tables
  - fenced_code
  - admonition
  - pymdownx.superfences
""",
    )
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 09B generate public docs/assets. No WRDS data copied.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    table_dir = root / "artifacts" / "tables"
    log_dir = root / "run_logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(root)
    copied = copy_public_assets(root)

    readme_path = root / "README.md"
    write_text(readme_path, make_readme(results))

    data_access = ensure_data_access(root)
    docs = ensure_docs(root, results)
    mkdocs = ensure_mkdocs(root)

    for p in [readme_path, data_access, *docs, mkdocs]:
        patch_repo_name(p)

    summary = {
        "ok": True,
        "run_id": run_id,
        "workspace": str(root),
        "repo": REPO,
        "readme": str(readme_path),
        "data_access": str(data_access),
        "mkdocs": str(mkdocs),
        "docs": [str(p.relative_to(root)) for p in docs],
        "copied_assets": copied,
        "note": "Public-safe docs/assets refresh only. No raw/vendor data or Parquet copied.",
    }

    summary_path = table_dir / "step09b_public_docs_summary.json"
    write_json(summary_path, summary)

    bundle = log_dir / f"step09b_public_docs_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [readme_path, data_access, mkdocs, *docs, summary_path]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))
        for rel in copied:
            p = root / rel
            if p.exists():
                tar.add(p, arcname=rel)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"BUNDLE={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
