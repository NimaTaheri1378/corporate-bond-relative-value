#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORKSPACE_DEFAULT = "/home/nt612/github/Corporate Bond Relative-Value"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_robustness_figure(root: Path) -> Path | None:
    import matplotlib.pyplot as plt

    fig_dir = root / "docs" / "assets" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ml_path = root / "docs" / "assets" / "tables" / "step08f_signal_robustness_metrics.csv"
    ra_path = root / "docs" / "assets" / "tables" / "step08g_rating_amount_exposure_metrics.csv"

    frames = []
    for path in [ml_path, ra_path]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    keep = df.loc[
        (df["sample"].astype(str) == "test_2020_2024")
        & (df["signal"].astype(str) == "composite_residual_rank")
        & (df["construction"].astype(str).isin([
            "global",
            "maturity_neutral",
            "liquidity_neutral",
            "rating_neutral",
            "amount_neutral",
        ]))
    ].copy()

    if keep.empty:
        return None

    order = ["global", "maturity_neutral", "liquidity_neutral", "rating_neutral", "amount_neutral"]
    keep["construction"] = pd.Categorical(keep["construction"], categories=order, ordered=True)
    keep = keep.sort_values("construction")

    keep["sharpe_net"] = pd.to_numeric(keep["sharpe_net"], errors="coerce")
    keep["annualized_return_net_approx"] = pd.to_numeric(keep["annualized_return_net_approx"], errors="coerce") * 100.0

    labels = [str(x).replace("_", " ") for x in keep["construction"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, keep["sharpe_net"])
    ax.invert_yaxis()
    ax.set_title("Composite Residual Rank: Robustness Net Sharpe")
    ax.set_xlabel("Net Sharpe, test 2020–2024")

    for bar, ret in zip(bars, keep["annualized_return_net_approx"], strict=False):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        label = "" if pd.isna(ret) else f"  {width:.2f} Sharpe, {ret:.1f}% ann."
        ax.text(width, y, label, va="center", fontsize=9)

    fig.tight_layout()
    out = fig_dir / "step08f_robustness_scorecard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def patch_mkdocs(root: Path) -> None:
    write(
        root / "mkdocs.yml",
        """site_name: Corporate Bond Relative-Value
repo_url: https://github.com/NimaTaheri1378/corporate-bond-relative-value
repo_name: NimaTaheri1378/corporate-bond-relative-value

theme:
  name: material

nav:
  - Home: index.md
  - Methodology: methodology.md
  - Results: results.md
  - Robustness: robustness.md
  - Data and extraction:
      - Extraction: extraction.md
      - Data contracts: data_contracts.md
      - API: api.md
  - Reproducibility: reproducibility.md
  - Model Card: model_card.md

markdown_extensions:
  - tables
  - fenced_code
  - admonition
  - pymdownx.superfences
""",
    )


def patch_docs_index(root: Path) -> None:
    write(
        root / "docs" / "index.md",
        """# Corporate Bond Relative-Value

Public-safe empirical fixed-income research repo for TRACE/FISD corporate-bond issuer-curve relative-value signals.

## Start here

- [Methodology](methodology.md)
- [Results](results.md)
- [Robustness](robustness.md)
- [Reproducibility](reproducibility.md)
- [Model card](model_card.md)

## Headline

The headline research candidate is `composite_residual_rank`: a transparent issuer-curve residual ranking signal evaluated with leakage-aware next-month labels and cost-aware long-short portfolios.

Raw WRDS/vendor data, row-level Parquet panels, credentials, cluster logs, and model binaries are intentionally excluded from this public release.
""",
    )


def patch_readme(root: Path, robustness_fig: Path | None) -> None:
    readme = root / "README.md"
    if not readme.exists():
        return

    s = readme.read_text(encoding="utf-8", errors="ignore")
    old = """| Risk profile | Interactive views |
|---|---|
| ![Return vs drawdown](docs/assets/figures/step08e_test_return_vs_drawdown.png) | [`interactive cumulative return`](docs/assets/interactive/step08e_test_cumulative_net_return.html) |
| **Return vs drawdown.** Nonlinear models produce similar returns with materially worse drawdowns. | **Interactive figures.** Public-safe HTML dashboards are built from aggregate outputs only. |"""

    if robustness_fig is not None:
        new = """| Risk profile | Robustness scorecard |
|---|---|
| ![Return vs drawdown](docs/assets/figures/step08e_test_return_vs_drawdown.png) | ![Robustness scorecard](docs/assets/figures/step08f_robustness_scorecard.png) |
| **Return vs drawdown.** Nonlinear models produce similar returns with materially worse drawdowns. | **Robustness.** The headline signal survives maturity-, liquidity-, rating-, and amount-neutral variants. |"""
    else:
        new = """| Risk profile | Interactive dashboards |
|---|---|
| ![Return vs drawdown](docs/assets/figures/step08e_test_return_vs_drawdown.png) | ![Net Sharpe leaderboard](docs/assets/figures/step08e_test_net_sharpe_leaderboard.png) |
| **Return vs drawdown.** Nonlinear models produce similar returns with materially worse drawdowns. | **Interactive dashboards.** See `docs/assets/interactive/` for public-safe HTML dashboards. |"""

    if old in s:
        s = s.replace(old, new)
    else:
        # Also patch the common one-line bottom-right link if the table changed slightly.
        s = s.replace(
            "[`interactive cumulative return`](docs/assets/interactive/step08e_test_cumulative_net_return.html)",
            "![Robustness scorecard](docs/assets/figures/step08f_robustness_scorecard.png)" if robustness_fig else "![Net Sharpe leaderboard](docs/assets/figures/step08e_test_net_sharpe_leaderboard.png)",
        )

    write(readme, s)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fix MkDocs strict warnings and README missing bottom-right figure.")
    parser.add_argument("--workspace", default=WORKSPACE_DEFAULT)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    fig = make_robustness_figure(root)
    patch_mkdocs(root)
    patch_docs_index(root)
    patch_readme(root, fig)

    summary = {
        "ok": True,
        "workspace": str(root),
        "mkdocs": str(root / "mkdocs.yml"),
        "docs_index": str(root / "docs" / "index.md"),
        "robustness_figure": str(fig) if fig else None,
        "readme": str(root / "README.md"),
    }
    out = root / "artifacts" / "tables" / "fix_mkdocs_ci_and_figures_summary.json"
    write(out, json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
