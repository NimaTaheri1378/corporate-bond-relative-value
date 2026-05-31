import json
from pathlib import Path

from corpbond_rv.data.extraction_plan import windows_for_granularity
from corpbond_rv.data.wrds_extract import ExtractTask


def test_quarter_windows_cover_date_range():
    windows = windows_for_granularity("2020-02-15", "2020-08-10", "quarter")
    assert [w.start for w in windows] == ["2020-02-15", "2020-04-01", "2020-07-01"]
    assert windows[-1].end == "2020-08-10"


def test_extract_task_json_round_trip(tmp_path: Path):
    task = ExtractTask(
        contract_name="demo",
        library="lib",
        table="tab",
        role="test",
        date_column="date",
        partition="quarter",
        columns=("a", "b"),
        quality_filters=("a is not null",),
        output_path=str(tmp_path / "part.parquet"),
        start_date="2020-01-01",
        end_date="2020-04-01",
        limit=10,
    )
    payload = task.to_json_mapping()
    restored = ExtractTask.from_mapping(json.loads(json.dumps(payload)))
    assert restored == task
