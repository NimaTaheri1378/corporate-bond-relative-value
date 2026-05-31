#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PARAM_READ_COLUMNS = [
    "model",
    "rmse",
    "mae",
    "score",
    "unstable",
    "n_issues",
    "n_issue_date_rows",
    "gross_volume",
    "maturity_span",
    "fit_warning",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def parquet_footer(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "file_size_bytes": 0,
        "schema_sha256": "",
        "error": "",
    }

    if not path.exists():
        rec["error"] = "missing_file"
        return rec

    try:
        pf = pq.ParquetFile(path)
        schema_text = str(pf.schema_arrow)
        rec["rows"] = int(pf.metadata.num_rows)
        rec["file_size_bytes"] = int(path.stat().st_size)
        rec["schema_sha256"] = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
        return rec
    except Exception as exc:
        rec["error"] = repr(exc)
        return rec


def read_params_summary(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "param_model_counts": {},
        "param_unstable_curves": 0,
        "param_warning_counts": {},
        "param_rmse_count": 0,
        "param_rmse_median": None,
        "param_rmse_p90": None,
        "param_rmse_p99": None,
        "param_mae_median": None,
        "param_n_issues_median": None,
        "param_maturity_span_median": None,
        "param_error": "",
    }

    try:
        if not path.exists():
            rec["param_error"] = "missing_file"
            return rec

        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows == 0:
            return rec

        available = set(pf.schema_arrow.names)
        selected = [c for c in PARAM_READ_COLUMNS if c in available]
        if not selected:
            rec["param_error"] = "no_selected_param_columns"
            return rec

        df = pf.read(columns=selected).to_pandas()

        if "model" in df.columns:
            rec["param_model_counts"] = {
                str(k): int(v)
                for k, v in df["model"].astype("string").fillna("<NA>").value_counts().items()
            }

        if "unstable" in df.columns:
            rec["param_unstable_curves"] = int(df["unstable"].fillna(False).astype(bool).sum())

        if "fit_warning" in df.columns:
            rec["param_warning_counts"] = {
                str(k): int(v)
                for k, v in df["fit_warning"].astype("string").fillna("").value_counts().items()
                if str(k)
            }

        for col, out_key in [
            ("rmse", "param_rmse"),
            ("mae", "param_mae"),
            ("n_issues", "param_n_issues"),
            ("maturity_span", "param_maturity_span"),
        ]:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            if col == "rmse":
                rec["param_rmse_count"] = int(len(s))
                rec["param_rmse_median"] = round(float(s.median()), 6)
                rec["param_rmse_p90"] = round(float(s.quantile(0.90)), 6)
                rec["param_rmse_p99"] = round(float(s.quantile(0.99)), 6)
            elif col == "mae":
                rec["param_mae_median"] = round(float(s.median()), 6)
            elif col == "n_issues":
                rec["param_n_issues_median"] = round(float(s.median()), 6)
            elif col == "maturity_span":
                rec["param_maturity_span_median"] = round(float(s.median()), 6)

        return rec

    except Exception as exc:
        rec["param_error"] = repr(exc)
        return rec


def validate_one(row: dict[str, Any]) -> dict[str, Any]:
    params_path = Path(str(row["params_output_path"]))
    residuals_path = Path(str(row["residuals_output_path"]))

    expected_curves = int(float(row.get("curves_fit", 0) or 0))
    expected_residuals = int(float(row.get("residual_rows", 0) or 0))

    pfoot = parquet_footer(params_path)
    rfoot = parquet_footer(residuals_path)
    psum = read_params_summary(params_path)

    rec: dict[str, Any] = {
        "universe": str(row.get("universe", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "params_output_path": str(params_path),
        "residuals_output_path": str(residuals_path),
        "expected_curves_fit": expected_curves,
        "expected_residual_rows": expected_residuals,
        "params_exists": pfoot["exists"],
        "residuals_exists": rfoot["exists"],
        "params_footer_rows": pfoot["rows"],
        "residuals_footer_rows": rfoot["rows"],
        "params_file_size_bytes": pfoot["file_size_bytes"],
        "residuals_file_size_bytes": rfoot["file_size_bytes"],
        "params_schema_sha256": pfoot["schema_sha256"],
        "residuals_schema_sha256": rfoot["schema_sha256"],
        "params_error": pfoot["error"],
        "residuals_error": rfoot["error"],
        "params_row_match": pfoot["rows"] == expected_curves,
        "residuals_row_match": rfoot["rows"] == expected_residuals,
        "ok": False,
    }

    rec.update(psum)

    rec["ok"] = bool(
        rec["params_exists"]
        and rec["residuals_exists"]
        and rec["params_error"] == ""
        and rec["residuals_error"] == ""
        and rec["param_error"] == ""
        and rec["params_row_match"]
        and rec["residuals_row_match"]
    )

    return rec


def merge_counter_dicts(series: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for item in series.dropna():
        if isinstance(item, dict):
            d = item
        else:
            try:
                d = json.loads(str(item).replace("'", '"'))
            except Exception:
                continue

        for k, v in d.items():
            counter[str(k)] += int(v)

    return dict(counter.most_common())


def weighted_median(values: pd.Series, weights: pd.Series | None = None) -> float | None:
    x = pd.to_numeric(values, errors="coerce")
    if weights is None:
        y = x.dropna()
        return None if y.empty else round(float(y.median()), 6)

    w = pd.to_numeric(weights, errors="coerce")
    m = x.notna() & w.notna() & (w > 0)

    if not m.any():
        y = x.dropna()
        return None if y.empty else round(float(y.median()), 6)

    xx = x.loc[m].to_numpy(dtype=float)
    ww = w.loc[m].to_numpy(dtype=float)

    order = xx.argsort()
    xx = xx[order]
    ww = ww[order]

    cdf = ww.cumsum() / ww.sum()
    return round(float(xx[cdf >= 0.5][0]), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 06F validate full curve-fit outputs. No WRDS.")
    parser.add_argument("--workspace", default="/home/nt612/github/Corporate Bond Relative-Value")
    parser.add_argument("--universe", default="core_public")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.workspace).expanduser().resolve()
    run_id = utc_stamp()

    manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_fit_v1_{args.universe}_manifest.csv"
    )
    nonempty_manifest_path = (
        root
        / "data"
        / "manifests"
        / "processed"
        / f"curve_fit_v1_{args.universe}_nonempty_manifest.csv"
    )
    step06e_summary_path = root / "artifacts" / "tables" / "step06e_curve_fit_residuals_summary.json"
    step06e_detail_path = root / "artifacts" / "tables" / "step06e_curve_fit_residuals_partition_summary.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 06E manifest: {manifest_path}")
    if not step06e_summary_path.exists():
        raise FileNotFoundError(f"Missing Step 06E summary: {step06e_summary_path}")

    manifest = pd.read_csv(manifest_path)
    step06e_summary = json.loads(step06e_summary_path.read_text())

    table_dir = root / "artifacts" / "tables"
    manifest_dir = root / "data" / "manifests" / "processed"
    log_dir = root / "run_logs"

    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    results: list[dict[str, Any]] = []

    print(f"run_id={run_id}")
    print(f"workspace={root}")
    print(f"universe={args.universe}")
    print(f"partitions_to_validate={len(rows)}")
    print(f"workers={args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(validate_one, row) for row in rows]

        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)

            if i == 1 or i % 100 == 0 or i == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                p_rows = sum(int(r.get("params_footer_rows") or 0) for r in results)
                r_rows = sum(int(r.get("residuals_footer_rows") or 0) for r in results)
                print(
                    f"progress {i}/{len(futures)} ok={ok} params_rows={p_rows:,} residual_rows={r_rows:,}",
                    flush=True,
                )

    detail = pd.DataFrame(results).sort_values(["start_date", "end_date", "params_output_path"]).reset_index(drop=True)

    expected_curves = int(pd.to_numeric(manifest["curves_fit"], errors="coerce").fillna(0).sum())
    expected_residuals = int(pd.to_numeric(manifest["residual_rows"], errors="coerce").fillna(0).sum())

    params_footer_rows = int(pd.to_numeric(detail["params_footer_rows"], errors="coerce").fillna(0).sum())
    residuals_footer_rows = int(pd.to_numeric(detail["residuals_footer_rows"], errors="coerce").fillna(0).sum())

    params_missing = int((~detail["params_exists"].astype(bool)).sum())
    residuals_missing = int((~detail["residuals_exists"].astype(bool)).sum())
    params_mismatch = int((~detail["params_row_match"].astype(bool)).sum())
    residuals_mismatch = int((~detail["residuals_row_match"].astype(bool)).sum())

    param_errors = int(
        detail["params_error"].fillna("").astype(str).ne("").sum()
        + detail["param_error"].fillna("").astype(str).ne("").sum()
    )
    residual_errors = int(detail["residuals_error"].fillna("").astype(str).ne("").sum())

    model_counts = merge_counter_dicts(detail["param_model_counts"])
    warning_counts = merge_counter_dicts(detail["param_warning_counts"])

    unstable_curves = int(pd.to_numeric(detail["param_unstable_curves"], errors="coerce").fillna(0).sum())

    curve_manifest_validated = manifest.copy()
    extras = detail.loc[
        :,
        [
            "params_output_path",
            "params_footer_rows",
            "residuals_footer_rows",
            "params_file_size_bytes",
            "residuals_file_size_bytes",
            "params_schema_sha256",
            "residuals_schema_sha256",
            "param_unstable_curves",
            "param_rmse_median",
            "param_rmse_p90",
            "param_rmse_p99",
        ],
    ].copy()

    curve_manifest_validated = curve_manifest_validated.merge(extras, on="params_output_path", how="left")

    validated_manifest_path = manifest_dir / f"curve_fit_v1_{args.universe}_validated_manifest.csv"
    validated_nonempty_path = manifest_dir / f"curve_fit_v1_{args.universe}_validated_nonempty_manifest.csv"

    curve_manifest_validated.to_csv(validated_manifest_path, index=False)
    curve_manifest_validated.loc[
        pd.to_numeric(curve_manifest_validated["params_footer_rows"], errors="coerce").fillna(0) > 0
    ].to_csv(validated_nonempty_path, index=False)

    detail_path = table_dir / f"step06f_curve_fit_{args.universe}_validation_detail.csv"
    validation_path = table_dir / f"step06f_curve_fit_{args.universe}_validation.json"

    detail.to_csv(detail_path, index=False)

    summary = {
        "ok": (
            params_missing == 0
            and residuals_missing == 0
            and params_mismatch == 0
            and residuals_mismatch == 0
            and param_errors == 0
            and residual_errors == 0
            and params_footer_rows == expected_curves
            and residuals_footer_rows == expected_residuals
        ),
        "run_id": run_id,
        "workspace": str(root),
        "universe": args.universe,
        "partitions_validated": int(len(detail)),
        "partitions_ok": int(detail["ok"].astype(bool).sum()),
        "params_missing_files": params_missing,
        "residuals_missing_files": residuals_missing,
        "params_row_mismatches": params_mismatch,
        "residuals_row_mismatches": residuals_mismatch,
        "param_read_errors": param_errors,
        "residual_read_errors": residual_errors,
        "expected_curves_from_manifest": expected_curves,
        "params_footer_rows": params_footer_rows,
        "params_footer_minus_manifest": int(params_footer_rows - expected_curves),
        "expected_residual_rows_from_manifest": expected_residuals,
        "residuals_footer_rows": residuals_footer_rows,
        "residuals_footer_minus_manifest": int(residuals_footer_rows - expected_residuals),
        "model_counts_from_params": {str(k): int(v) for k, v in model_counts.items()},
        "unstable_curves_from_params": unstable_curves,
        "unstable_curve_pct_from_params": None if params_footer_rows == 0 else round(100.0 * unstable_curves / params_footer_rows, 6),
        "fit_warning_counts_from_params": {str(k): int(v) for k, v in warning_counts.items()},
        "rmse_median_by_partition_weighted": weighted_median(detail["param_rmse_median"], detail["params_footer_rows"]),
        "rmse_p90_median_by_partition": weighted_median(detail["param_rmse_p90"], detail["params_footer_rows"]),
        "rmse_p99_median_by_partition": weighted_median(detail["param_rmse_p99"], detail["params_footer_rows"]),
        "params_schema_fingerprints": {str(k): int(v) for k, v in detail["params_schema_sha256"].fillna("MISSING").value_counts().items()},
        "residuals_schema_fingerprints": {str(k): int(v) for k, v in detail["residuals_schema_sha256"].fillna("MISSING").value_counts().items()},
        "source_step06e_summary": str(step06e_summary_path),
        "source_step06e_detail": str(step06e_detail_path),
        "source_step06e_summary_ok": bool(step06e_summary.get("ok")),
        "validated_manifest": str(validated_manifest_path),
        "validated_nonempty_manifest": str(validated_nonempty_path),
        "detail_path": str(detail_path),
        "note": "Local-only curve fit validation. Recomputes model mix from parameter parquet because resumed partitions were skipped in Step 06E summary.",
    }

    write_json(validation_path, summary)

    bundle = log_dir / f"step06f_curve_fit_validation_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            validation_path,
            detail_path,
            validated_manifest_path,
            validated_nonempty_path,
            step06e_summary_path,
        ]:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION={validation_path}")
    print(f"DETAIL={detail_path}")
    print(f"VALIDATED_MANIFEST={validated_manifest_path}")
    print(f"BUNDLE={bundle}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
