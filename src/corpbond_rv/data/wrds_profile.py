from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from corpbond_rv.data.table_contracts import (
    TableContract,
    load_contracts,
    validate_contracts_against_schema,
)
from corpbond_rv.data.wrds_sql import build_profile_sql, build_sample_sql
from corpbond_rv.utils.paths import ensure_dir


def _connect_wrds():
    import wrds

    return wrds.Connection()


def _try_statement_timeout(conn, seconds: int | None) -> None:
    if not seconds or seconds <= 0:
        return
    try:
        conn.raw_sql(f"set statement_timeout to '{int(seconds)}s'")
    except Exception:
        pass


def profile_one(conn, contract: TableContract, *, sample_rows: int, output_dir: Path) -> dict[str, object]:
    started = time.time()
    row: dict[str, object] = {
        "contract": contract.name,
        "role": contract.role,
        "priority": contract.priority,
        "library": contract.library,
        "table": contract.table,
        "fqname": contract.fqname,
        "date_column": contract.date_column or "",
        "partition": contract.partition,
        "enabled_by_default": contract.enabled_by_default,
        "profile_ok": False,
        "sample_ok": False,
        "error": "",
    }
    try:
        prof = conn.raw_sql(build_profile_sql(contract))
        if len(prof):
            for col, value in prof.iloc[0].items():
                if pd.isna(value):
                    row[col] = ""
                elif hasattr(value, "isoformat"):
                    row[col] = value.isoformat()
                else:
                    row[col] = int(value) if str(col) == "n_rows" else value
        row["profile_ok"] = True
    except Exception as exc:
        row["error"] = f"profile_failed: {exc!r}"

    try:
        sample = conn.raw_sql(build_sample_sql(contract, sample_rows))
        sample_dir = ensure_dir(output_dir / "samples")
        sample_path = sample_dir / f"{contract.name}.csv"
        sample.to_csv(sample_path, index=False)
        row["sample_path"] = str(sample_path)
        row["sample_n_rows"] = int(len(sample))
        row["sample_ok"] = True
    except Exception as exc:
        msg = f"sample_failed: {exc!r}"
        row["error"] = (str(row.get("error") or "") + " | " + msg).strip(" |")

    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def write_markdown_report(
    path: Path,
    profiles: pd.DataFrame,
    validation: pd.DataFrame,
    contracts_path: Path,
    schema_csv: Path,
) -> None:
    def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
        if df.empty:
            return "_No rows._\n"
        cols = [c for c in columns if c in df.columns]
        shown = df.loc[:, cols].head(max_rows).copy()
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in shown.iterrows():
            vals = [str(r[c]).replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        if len(df) > max_rows:
            lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
        return "\n".join(lines) + "\n"

    text = [
        "# Step 02 WRDS table contract profile",
        "",
        f"Run UTC: `{datetime.now(UTC).isoformat()}`",
        f"Contracts: `{contracts_path}`",
        f"Schema inventory: `{schema_csv}`",
        "",
        "## Contract validation",
        "",
        md_table(
            validation,
            [
                "contract", "role", "fqname", "table_exists", "n_available_columns",
                "n_requested_columns", "missing_required_columns", "valid",
            ],
        ),
        "",
        "## WRDS table profiles",
        "",
        md_table(
            profiles,
            [
                "contract", "role", "fqname", "n_rows", "min_date", "max_date",
                "date_column", "partition", "profile_ok", "sample_ok", "elapsed_sec", "error",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- `trace_enhanced_clean` is the primary transaction source for daily price/liquidity construction.",
        "- `bondret_monthly` is the primary monthly return-label source and pricing reference.",
        "- FISD tables give issue terms, issuer metadata, coupons, ratings, and amount-outstanding history.",
        "- Link tables map bonds/issuers to CRSP/Compustat entities for later firm-state features.",
        "",
        "Send this report and the Step 02 log archive before the first large extraction.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def make_profile_visuals(profiles: pd.DataFrame, artifacts_dir: Path) -> None:
    try:
        import plotly.express as px
    except Exception:
        return
    fig_dir = ensure_dir(artifacts_dir / "figures_interactive")
    table_dir = ensure_dir(artifacts_dir / "tables")
    df = profiles.copy()
    if "n_rows" in df.columns:
        df["n_rows_numeric"] = pd.to_numeric(df["n_rows"], errors="coerce")
    else:
        df["n_rows_numeric"] = None
    df.to_csv(table_dir / "step02_wrds_table_profiles.csv", index=False)
    shown = df.sort_values(["priority", "contract"]).copy()
    if shown["n_rows_numeric"].notna().any():
        fig = px.bar(
            shown,
            x="contract",
            y="n_rows_numeric",
            color="role",
            hover_data=["fqname", "min_date", "max_date", "partition", "profile_ok", "sample_ok"],
            log_y=True,
            title="Step 02 WRDS table inventory: row counts by selected contract",
            labels={"n_rows_numeric": "Rows, log scale", "contract": "Table contract"},
        )
        fig.update_layout(height=650, xaxis_tickangle=-35, margin=dict(l=70, r=30, t=80, b=160))
        fig.write_html(fig_dir / "step02_wrds_table_profile.html", include_plotlyjs="cdn", full_html=True)


def profile_contracts(
    *,
    contracts_path: Path,
    schema_csv: Path,
    output_dir: Path,
    sample_rows: int = 5,
    statement_timeout_seconds: int | None = 180,
    artifacts_dir: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = load_contracts(contracts_path)
    validation = validate_contracts_against_schema(contracts, schema_csv)
    validation.to_csv(output_dir / "contract_validation.csv", index=False)

    bad_required = validation[~validation["valid"]]
    if not bad_required.empty:
        # Still write the validation report, but fail early before hitting WRDS.
        write_markdown_report(output_dir / "step02_profile_report.md", pd.DataFrame(), validation, contracts_path, schema_csv)
        raise RuntimeError(
            "Some table contracts are invalid against Step 01 schema inventory: "
            + ", ".join(bad_required["contract"].astype(str).tolist())
        )

    rows = []
    conn = _connect_wrds()
    try:
        _try_statement_timeout(conn, statement_timeout_seconds)
        for contract in contracts.values():
            rows.append(profile_one(conn, contract, sample_rows=sample_rows, output_dir=output_dir))
    finally:
        try:
            conn.close()
        except Exception:
            pass

    profiles = pd.DataFrame(rows)
    profiles.to_csv(output_dir / "table_profiles.csv", index=False)
    (output_dir / "table_profiles.json").write_text(
        profiles.to_json(orient="records", indent=2, date_format="iso") + "\n",
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "step02_profile_report.md", profiles, validation, contracts_path, schema_csv)
    if artifacts_dir is not None:
        make_profile_visuals(profiles, artifacts_dir)
    return {
        "n_contracts": int(len(contracts)),
        "n_valid_contracts": int(validation["valid"].sum()),
        "n_profile_ok": int(profiles["profile_ok"].sum()) if "profile_ok" in profiles.columns else 0,
        "n_sample_ok": int(profiles["sample_ok"].sum()) if "sample_ok" in profiles.columns else 0,
        "output_dir": str(output_dir),
    }
