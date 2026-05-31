from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from corpbond_rv.utils.logging import setup_logging
from corpbond_rv.utils.paths import ensure_dir, project_root


DEFAULT_LIBRARY_PATTERNS = [
    "trace_enhanced",
    "trace_standard",
    "trace",
    "fisd_fisd",
    "fisd_common",
    "fisd",
    "wrdsapps_bondret",
    "bondret",
    "contrib_corporate_bond_returns",
    "corporate_bond_returns",
    "wrdsapps_link_crsp_bond",
    "link_crsp_bond",
    "contrib_bond_firm_link",
    "bond_firm_link",
    "crsp",
    "comp",
    "compustat",
]

DEFAULT_TABLE_KEYWORDS = [
    "trace",
    "trade",
    "trans",
    "bond",
    "issue",
    "issuer",
    "fisd",
    "return",
    "ret",
    "link",
    "cusip",
    "permno",
    "rating",
    "security",
    "master",
    "amount",
    "coupon",
    "maturity",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value).strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out


def _match_libraries(libraries: list[str], patterns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    lower_patterns = [p.lower() for p in patterns]
    for lib in sorted(map(str, libraries)):
        lib_lower = lib.lower()
        reasons = []
        for pattern, pattern_lower in zip(patterns, lower_patterns, strict=False):
            if lib_lower == pattern_lower:
                reasons.append(f"exact:{pattern}")
            elif pattern_lower in lib_lower:
                reasons.append(f"contains:{pattern}")
        if reasons:
            rows.append({"library": lib, "match_reason": ";".join(reasons)})
    return pd.DataFrame(rows, columns=["library", "match_reason"])


def _safe_list_tables(conn, library: str) -> list[str]:
    try:
        return list(map(str, conn.list_tables(library=library)))
    except TypeError:
        return list(map(str, conn.list_tables(library)))


def _safe_describe_table(conn, library: str, table: str) -> pd.DataFrame:
    desc = conn.describe_table(library=library, table=table)
    if desc is None:
        return pd.DataFrame()
    if not isinstance(desc, pd.DataFrame):
        desc = pd.DataFrame(desc)
    desc = desc.reset_index(drop=True)
    desc.columns = [str(c) for c in desc.columns]
    desc.insert(0, "table", table)
    desc.insert(0, "library", library)
    return desc


def _select_tables_for_description(
    tables: list[str],
    table_keywords: list[str],
    describe_level: str,
    max_describe_per_library: int,
) -> tuple[list[str], str]:
    if describe_level == "none":
        return [], "none"
    if describe_level == "all":
        selected = list(tables)
        reason = "all"
    else:
        lower_keywords = [k.lower() for k in table_keywords]
        selected = [
            table
            for table in tables
            if any(keyword in table.lower() for keyword in lower_keywords)
        ]
        reason = "keyword"
        if not selected:
            selected = list(tables[: min(20, len(tables))])
            reason = "fallback_first_20"

    if max_describe_per_library and max_describe_per_library > 0:
        selected = selected[:max_describe_per_library]
        reason = f"{reason}_capped_{max_describe_per_library}"
    return selected, reason


def _rows_to_md(df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return "_No display columns._\n"
    shown = df.loc[:, use_cols].head(max_rows)
    lines = []
    lines.append("| " + " | ".join(shown.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(shown.columns)) + " |")
    for _, row in shown.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in shown.columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def _write_markdown_inventory(
    out_path: Path,
    summary: dict[str, Any],
    target_libs: pd.DataFrame,
    tables: pd.DataFrame,
    schema: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    n_schema_tables = 0
    if not schema.empty and {"library", "table"}.issubset(schema.columns):
        n_schema_tables = int(schema[["library", "table"]].drop_duplicates().shape[0])

    schema_cols = list(schema.columns[:8]) if not schema.empty else []
    text = [
        "# WRDS Schema Discovery Inventory",
        "",
        f"- Run UTC: `{summary.get('run_timestamp_utc')}`",
        f"- Total libraries visible: `{summary.get('n_libraries_visible')}`",
        f"- Candidate libraries matched: `{summary.get('n_candidate_libraries')}`",
        f"- Total candidate tables listed: `{summary.get('n_candidate_tables')}`",
        f"- Tables described: `{n_schema_tables}`",
        "",
        "## Candidate libraries",
        "",
        _rows_to_md(target_libs, ["library", "match_reason", "n_tables", "n_described_tables"]),
        "",
        "## Candidate tables, first rows",
        "",
        _rows_to_md(tables, ["library", "table"], max_rows=120),
        "",
        "## Schema columns, first rows",
        "",
        _rows_to_md(schema, schema_cols, max_rows=120),
        "",
        "## Description failures",
        "",
        _rows_to_md(failures, ["library", "table", "error"], max_rows=120),
        "",
        "## Next action",
        "",
        "Send this entire `data/manifests/wrds_schema/` folder plus the run log tarball.",
        "The next step will choose exact extraction tables and write safe, chunked pull scripts.",
        "",
    ]
    out_path.write_text("\n".join(text), encoding="utf-8")


def discover_wrds_schema(
    *,
    config_path: Path,
    output_dir: Path,
    describe_level: str,
    max_describe_per_library: int,
    library_patterns: list[str] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    logger = setup_logging(output_dir / "wrds_discovery.log")

    cfg = _read_yaml(config_path)
    wrds_cfg = cfg.get("wrds", {}) if isinstance(cfg, dict) else {}
    discovery_cfg = wrds_cfg.get("discovery", {}) if isinstance(wrds_cfg, dict) else {}

    patterns = library_patterns or wrds_cfg.get("target_library_patterns") or DEFAULT_LIBRARY_PATTERNS
    table_keywords = wrds_cfg.get("table_name_keywords") or DEFAULT_TABLE_KEYWORDS

    patterns = _dedupe_keep_order(list(map(str, patterns)))
    table_keywords = _dedupe_keep_order(list(map(str, table_keywords)))

    if describe_level == "config":
        describe_level = str(discovery_cfg.get("describe_level", "likely"))

    if max_describe_per_library < 0:
        max_describe_per_library = int(discovery_cfg.get("max_describe_per_library", 300))

    logger.info("Starting WRDS schema discovery")
    logger.info("Config path: %s", config_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Describe level: %s", describe_level)
    logger.info("Max describe per library: %s", max_describe_per_library)
    logger.info("Library patterns: %s", patterns)

    t0 = time.time()

    try:
        import wrds
    except Exception as exc:
        raise RuntimeError("Could not import wrds package in this environment") from exc

    conn = wrds.Connection()
    try:
        libraries = sorted(map(str, conn.list_libraries()))
        libs_df = pd.DataFrame({"library": libraries})
        libs_df.to_csv(output_dir / "all_visible_libraries.csv", index=False)
        logger.info("Visible libraries: %d", len(libraries))

        target_libs = _match_libraries(libraries, patterns)
        if target_libs.empty:
            logger.warning("No candidate libraries matched. See all_visible_libraries.csv")
            summary = {
                "run_timestamp_utc": datetime.now(UTC).isoformat(),
                "n_libraries_visible": len(libraries),
                "n_candidate_libraries": 0,
                "n_candidate_tables": 0,
                "patterns": patterns,
                "elapsed_sec": round(time.time() - t0, 3),
            }
            target_libs.to_csv(output_dir / "candidate_libraries.csv", index=False)
            pd.DataFrame(columns=["library", "table"]).to_csv(
                output_dir / "candidate_tables.csv", index=False
            )
            pd.DataFrame().to_csv(output_dir / "candidate_schema.csv", index=False)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        table_rows: list[dict[str, str]] = []
        schema_frames: list[pd.DataFrame] = []
        failure_rows: list[dict[str, str]] = []
        described_count_by_library: dict[str, int] = {}
        table_count_by_library: dict[str, int] = {}

        for library in target_libs["library"].tolist():
            logger.info("Listing tables for library=%s", library)
            try:
                tables = sorted(_safe_list_tables(conn, library))
            except Exception as exc:
                logger.exception("Failed listing tables for %s", library)
                failure_rows.append({"library": library, "table": "*LIST_TABLES*", "error": repr(exc)})
                tables = []

            table_count_by_library[library] = len(tables)
            for table in tables:
                table_rows.append({"library": library, "table": table})

            selected, selection_reason = _select_tables_for_description(
                tables, table_keywords, describe_level, max_describe_per_library
            )
            described_count_by_library[library] = len(selected)
            logger.info(
                "library=%s tables=%d selected_for_description=%d reason=%s",
                library,
                len(tables),
                len(selected),
                selection_reason,
            )

            for idx, table in enumerate(selected, start=1):
                logger.info("Describe %s.%s (%d/%d)", library, table, idx, len(selected))
                try:
                    desc = _safe_describe_table(conn, library, table)
                    if desc.empty:
                        failure_rows.append(
                            {"library": library, "table": table, "error": "empty describe_table result"}
                        )
                    else:
                        schema_frames.append(desc)
                except Exception as exc:
                    logger.warning("Failed describe_table for %s.%s: %r", library, table, exc)
                    failure_rows.append({"library": library, "table": table, "error": repr(exc)})

        tables_df = pd.DataFrame(table_rows, columns=["library", "table"])
        failures_df = pd.DataFrame(failure_rows, columns=["library", "table", "error"])
        schema_df = pd.concat(schema_frames, ignore_index=True) if schema_frames else pd.DataFrame()

        target_libs = target_libs.copy()
        target_libs["n_tables"] = target_libs["library"].map(table_count_by_library).fillna(0).astype(int)
        target_libs["n_described_tables"] = (
            target_libs["library"].map(described_count_by_library).fillna(0).astype(int)
        )

        target_libs.to_csv(output_dir / "candidate_libraries.csv", index=False)
        tables_df.to_csv(output_dir / "candidate_tables.csv", index=False)
        failures_df.to_csv(output_dir / "describe_failures.csv", index=False)
        schema_df.to_csv(output_dir / "candidate_schema.csv", index=False)

        summary = {
            "run_timestamp_utc": datetime.now(UTC).isoformat(),
            "n_libraries_visible": len(libraries),
            "n_candidate_libraries": int(target_libs.shape[0]),
            "n_candidate_tables": int(tables_df.shape[0]),
            "n_schema_rows": int(schema_df.shape[0]),
            "n_failures": int(failures_df.shape[0]),
            "patterns": patterns,
            "table_keywords": table_keywords,
            "describe_level": describe_level,
            "max_describe_per_library": max_describe_per_library,
            "elapsed_sec": round(time.time() - t0, 3),
            "table_count_by_library": table_count_by_library,
            "described_count_by_library": described_count_by_library,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_markdown_inventory(
            output_dir / "schema_inventory.md",
            summary,
            target_libs,
            tables_df,
            schema_df,
            failures_df,
        )
        logger.info("WRDS schema discovery complete")
        logger.info("Summary: %s", summary)
        return summary

    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover WRDS libraries, tables, and candidate schemas.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default="data/manifests/wrds_schema")
    parser.add_argument("--describe-level", choices=["config", "none", "likely", "all"], default="config")
    parser.add_argument("--max-describe-per-library", type=int, default=-1)
    parser.add_argument("--library-pattern", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()
    summary = discover_wrds_schema(
        config_path=(root / args.config).resolve(),
        output_dir=(root / args.output_dir).resolve(),
        describe_level=args.describe_level,
        max_describe_per_library=args.max_describe_per_library,
        library_patterns=args.library_pattern,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
