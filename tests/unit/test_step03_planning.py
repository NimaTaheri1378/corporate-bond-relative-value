from pathlib import Path
import json

import pandas as pd

from corpbond_rv.data.extraction_plan import build_extraction_tasks


def _write_contracts(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_step03_trace_pilot_uses_last_months(tmp_path: Path):
    contracts = tmp_path / "contracts.json"
    profiles = tmp_path / "profiles.csv"
    _write_contracts(
        contracts,
        {
            "tables": {
                "trace_enhanced_clean": {
                    "role": "primary_clean_trace_trades",
                    "priority": 1,
                    "library": "lib",
                    "table": "tbl",
                    "date_column": "trd_exctn_dt",
                    "partition": "month",
                    "columns": ["cusip_id", "trd_exctn_dt"],
                    "quality_filters": [],
                    "enabled_by_default": True,
                }
            }
        },
    )
    pd.DataFrame(
        [
            {
                "contract": "trace_enhanced_clean",
                "n_rows": 100,
                "min_date": "2024-01-15",
                "max_date": "2024-05-20",
            }
        ]
    ).to_csv(profiles, index=False)
    tasks = build_extraction_tasks(
        contracts,
        profiles,
        tmp_path / "raw",
        phase="trace",
        trace_pilot_months=2,
    )
    assert len(tasks) == 2
    assert tasks[0].start_date == "2024-04-01"
    assert tasks[1].start_date == "2024-05-01"


def test_step03_core_uses_quarter_windows_and_static(tmp_path: Path):
    contracts = tmp_path / "contracts.json"
    profiles = tmp_path / "profiles.csv"
    _write_contracts(
        contracts,
        {
            "tables": {
                "bondret_monthly": {
                    "role": "primary_monthly_return_labels",
                    "priority": 1,
                    "library": "lib",
                    "table": "bondret",
                    "date_column": "date",
                    "partition": "year",
                    "columns": ["date", "cusip"],
                    "quality_filters": [],
                    "enabled_by_default": True,
                },
                "fisd_issue_issuer": {
                    "role": "primary_security_master",
                    "priority": 1,
                    "library": "fisd",
                    "table": "issue_issuer",
                    "date_column": None,
                    "partition": "none",
                    "columns": ["issue_id"],
                    "quality_filters": [],
                    "enabled_by_default": True,
                },
            }
        },
    )
    pd.DataFrame(
        [{"contract": "bondret_monthly", "n_rows": 100, "min_date": "2020-07-31", "max_date": "2021-02-28"}]
    ).to_csv(profiles, index=False)
    tasks = build_extraction_tasks(contracts, profiles, tmp_path / "raw", phase="core")
    names = [task.contract_name for task in tasks]
    assert names.count("fisd_issue_issuer") == 1
    assert names.count("bondret_monthly") == 3
    assert [t.start_date for t in tasks if t.contract_name == "bondret_monthly"] == [
        "2020-07-01",
        "2020-10-01",
        "2021-01-01",
    ]
