#!/usr/bin/env python
from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    root = Path("/home/nt612/github/Corporate Bond Relative-Value").resolve()
    run_id = utc_stamp()

    detail_path = root / "artifacts/tables/step04g_clean_output_footer_validation_detail.csv"
    validation_path = root / "artifacts/tables/step04g_clean_output_footer_validation.json"

    if not detail_path.exists():
        raise FileNotFoundError(detail_path)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    validation = json.loads(validation_path.read_text())
    if not validation.get("ok"):
        raise RuntimeError(f"Step 04G validation is not OK: {validation_path}")

    detail = pd.read_csv(detail_path)

    manifest_dir = root / "data/manifests/processed"
    table_dir = root / "artifacts/tables"
    log_dir = root / "run_logs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def build_universe_manifest(universe: str) -> pd.DataFrame:
        if universe == "extended_regular":
            path_col = "extended_output_path"
            row_col = "extended_footer_rows"
            schema_col = "extended_schema_sha256"
            size_col = "extended_file_size_bytes"
        elif universe == "core_public":
            path_col = "core_public_output_path"
            row_col = "core_public_footer_rows"
            schema_col = "core_public_schema_sha256"
            size_col = "core_public_file_size_bytes"
        else:
            raise ValueError(universe)

        out = pd.DataFrame(
            {
                "universe": universe,
                "start_date": detail["start_date"],
                "end_date": detail["end_date"],
                "source_stage": detail["source_stage"],
                "raw_rows": pd.to_numeric(detail["raw_rows"], errors="coerce").fillna(0).astype("int64"),
                "clean_rows": pd.to_numeric(detail[row_col], errors="coerce").fillna(0).astype("int64"),
                "file_size_bytes": pd.to_numeric(detail[size_col], errors="coerce").fillna(0).astype("int64"),
                "schema_sha256": detail[schema_col].astype("string"),
                "output_path": detail[path_col].astype("string"),
            }
        )

        out["is_empty_partition"] = out["clean_rows"].eq(0)
        out["year"] = pd.to_datetime(out["start_date"]).dt.year
        out = out.sort_values(["start_date", "end_date", "output_path"]).reset_index(drop=True)
        return out

    extended = build_universe_manifest("extended_regular")
    core = build_universe_manifest("core_public")

    extended_path = manifest_dir / "trace_clean_v1_extended_regular_manifest.csv"
    core_path = manifest_dir / "trace_clean_v1_core_public_manifest.csv"
    extended_nonempty_path = manifest_dir / "trace_clean_v1_extended_regular_nonempty_manifest.csv"
    core_nonempty_path = manifest_dir / "trace_clean_v1_core_public_nonempty_manifest.csv"

    extended.to_csv(extended_path, index=False)
    core.to_csv(core_path, index=False)
    extended.loc[~extended["is_empty_partition"]].to_csv(extended_nonempty_path, index=False)
    core.loc[~core["is_empty_partition"]].to_csv(core_nonempty_path, index=False)

    yearly = (
        pd.concat([extended, core], ignore_index=True)
        .groupby(["universe", "year"], as_index=False)
        .agg(
            partitions=("output_path", "count"),
            nonempty_partitions=("is_empty_partition", lambda x: int((~x).sum())),
            clean_rows=("clean_rows", "sum"),
            raw_rows=("raw_rows", "sum"),
            file_size_bytes=("file_size_bytes", "sum"),
        )
    )
    yearly["retention_pct_vs_raw"] = (100.0 * yearly["clean_rows"] / yearly["raw_rows"]).round(6)

    yearly_path = table_dir / "step04h_clean_trace_yearly_summary.csv"
    yearly.to_csv(yearly_path, index=False)

    summary = {
        "ok": True,
        "run_id": run_id,
        "workspace": str(root),
        "extended_regular": {
            "partitions": int(len(extended)),
            "nonempty_partitions": int((~extended["is_empty_partition"]).sum()),
            "empty_partitions": int(extended["is_empty_partition"].sum()),
            "clean_rows": int(extended["clean_rows"].sum()),
            "schema_fingerprints": {str(k): int(v) for k, v in extended["schema_sha256"].value_counts().items()},
            "manifest": str(extended_path),
            "nonempty_manifest": str(extended_nonempty_path),
        },
        "core_public": {
            "partitions": int(len(core)),
            "nonempty_partitions": int((~core["is_empty_partition"]).sum()),
            "empty_partitions": int(core["is_empty_partition"].sum()),
            "clean_rows": int(core["clean_rows"].sum()),
            "schema_fingerprints": {str(k): int(v) for k, v in core["schema_sha256"].value_counts().items()},
            "manifest": str(core_path),
            "nonempty_manifest": str(core_nonempty_path),
        },
        "yearly_summary": str(yearly_path),
        "source_validation": str(validation_path),
    }

    # Guardrails from Step 04F/04G.
    if summary["extended_regular"]["clean_rows"] != 291_886_230:
        summary["ok"] = False
    if summary["core_public"]["clean_rows"] != 263_929_435:
        summary["ok"] = False
    if summary["extended_regular"]["partitions"] != 1914:
        summary["ok"] = False
    if summary["core_public"]["partitions"] != 1914:
        summary["ok"] = False

    summary_path = table_dir / "step04h_clean_trace_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bundle = log_dir / f"step04h_clean_trace_manifest_bundle_{run_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in [
            summary_path,
            yearly_path,
            extended_path,
            core_path,
            extended_nonempty_path,
            core_nonempty_path,
            validation_path,
        ]:
            tar.add(p, arcname=str(p.relative_to(root)))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BUNDLE={bundle}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
