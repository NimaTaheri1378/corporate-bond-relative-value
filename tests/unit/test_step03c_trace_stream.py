from __future__ import annotations

from datetime import date

import pytest

from corpbond_rv.data.trace_stream_extract import (
    TraceContract,
    TraceWindowTask,
    choose_columns,
    count_sql,
    make_windows,
    select_sql,
    stream_trace_to_parquet,
    window_label,
)


def test_select_sql_is_bounded_and_orderless():
    task = TraceWindowTask(
        contract_name="trace_enhanced_clean",
        library="wrdsapps_bondret",
        table="trace_enhanced_clean",
        date_column="trd_exctn_dt",
        columns=("cusip_id", "trd_exctn_dt", "rptd_pr", "entrd_vol_qt"),
        quality_filters=("rptd_pr > 0", "entrd_vol_qt is not null"),
        start_date="2025-03-03",
        end_date="2025-03-04",
        output_path="x.parquet",
    )
    sql = select_sql(task).lower()
    assert "wrdsapps_bondret.trace_enhanced_clean" in sql
    assert "trd_exctn_dt >= date '2025-03-03'" in sql
    assert "trd_exctn_dt < date '2025-03-04'" in sql
    assert "order by" not in sql
    assert "limit" not in sql
    assert "count(*)" in count_sql(task).lower()


def test_week_windows_and_label_are_stable():
    windows = make_windows("2025-03-01", "2025-03-17", "week")
    assert windows == [
        ("2025-03-01", "2025-03-08"),
        ("2025-03-08", "2025-03-15"),
        ("2025-03-15", "2025-03-17"),
    ]
    assert window_label("2025-03-01", "2025-03-08") == "20250301_to_20250308"


def test_choose_columns_lean_preserves_required_fields():
    c = TraceContract(
        name="trace_enhanced_clean",
        library="wrdsapps_bondret",
        table="trace_enhanced_clean",
        date_column="trd_exctn_dt",
        columns=("cusip_id", "trd_exctn_dt", "rptd_pr", "entrd_vol_qt", "unused_col"),
        quality_filters=(),
    )
    cols = choose_columns(c, "lean")
    assert "cusip_id" in cols
    assert "trd_exctn_dt" in cols
    assert "rptd_pr" in cols
    assert "entrd_vol_qt" in cols
    assert "unused_col" not in cols


def test_stream_trace_to_parquet_does_not_require_cursor_description(monkeypatch, tmp_path):
    """WRDS named cursors may expose description=None; task.columns are the source of truth."""
    pq = pytest.importorskip("pyarrow.parquet")

    class FakeCursor:
        description = None
        itersize = 0

        def __init__(self):
            self._batches = [
                [
                    ("123456AA1", date(2025, 1, 31), 99.75, 1_000_000),
                    ("123456AA2", date(2025, 1, 31), 101.25, 2_000_000),
                ],
                [],
            ]

        def execute(self, sql):
            self.sql = sql

        def fetchmany(self, n):
            return self._batches.pop(0)

        def close(self):
            pass

    class FakeConn:
        def cursor(self, name=None):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr("corpbond_rv.data.trace_stream_extract.wrds_connect", lambda *args, **kwargs: FakeConn())

    task = TraceWindowTask(
        contract_name="trace_enhanced_clean",
        library="wrdsapps_bondret",
        table="trace_enhanced_clean",
        date_column="trd_exctn_dt",
        columns=("cusip_id", "trd_exctn_dt", "rptd_pr", "entrd_vol_qt"),
        quality_filters=(),
        start_date="2025-01-31",
        end_date="2025-02-01",
        output_path=str(tmp_path / "trace_day.parquet"),
        validate_counts=False,
        chunk_rows=10,
        compression="zstd",
    )

    result = stream_trace_to_parquet(task)

    assert result["ok"] is True
    assert result["n_rows"] == 2
    assert result["n_chunks"] == 1
    assert result["returncode"] == 0
    assert pq.ParquetFile(task.output_path).metadata.num_rows == 2



def test_trace_arrow_schema_stabilizes_numeric_mixed_chunks():
    pa = pytest.importorskip("pyarrow")
    from corpbond_rv.data.trace_stream_extract import trace_arrow_schema

    cols = ("cusip_id", "trd_exctn_dt", "days_to_sttl_ct", "orig_msg_seq_nb", "rptd_pr")
    schema = trace_arrow_schema(cols)

    names = [f.name for f in schema]
    types = {f.name: f.type for f in schema}
    assert names == list(cols)
    assert pa.types.is_floating(types["days_to_sttl_ct"])
    assert pa.types.is_floating(types["orig_msg_seq_nb"])
    assert pa.types.is_floating(types["rptd_pr"])
    assert pa.types.is_timestamp(types["trd_exctn_dt"])
    assert pa.types.is_string(types["cusip_id"])

    import pandas as pd

    chunk_int = pd.DataFrame(
        {
            "cusip_id": ["A", "B"],
            "trd_exctn_dt": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "days_to_sttl_ct": [1, 2],
            "orig_msg_seq_nb": [None, 123],
            "rptd_pr": [99.5, 101.25],
        }
    )
    chunk_float = pd.DataFrame(
        {
            "cusip_id": ["C"],
            "trd_exctn_dt": pd.to_datetime(["2020-01-02"]),
            "days_to_sttl_ct": [3.5],
            "orig_msg_seq_nb": [None],
            "rptd_pr": [100],
        }
    )

    t1 = pa.Table.from_pandas(chunk_int, preserve_index=False).select(list(cols)).cast(schema, safe=False)
    t2 = pa.Table.from_pandas(chunk_float, preserve_index=False).select(list(cols)).cast(schema, safe=False)

    assert t1.schema == schema
    assert t2.schema == schema

