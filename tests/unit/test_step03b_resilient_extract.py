from pathlib import Path

import pandas as pd
import pytest

from corpbond_rv.data.wrds_extract import ExtractTask, run_tasks


def test_run_tasks_checkpoint_skip_existing(tmp_path: Path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "existing.parquet"
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(out, index=False)
    task = ExtractTask(
        contract_name="dummy",
        library="lib",
        table="table",
        role="test",
        date_column=None,
        partition="none",
        columns=("x",),
        quality_filters=(),
        output_path=str(out),
    )
    manifest = tmp_path / "manifest.csv"
    df = run_tasks([task], manifest_path=manifest, workers=1, engine="sequential")
    assert manifest.exists()
    assert bool(df.loc[0, "ok"])
    assert bool(df.loc[0, "skipped"])
    assert int(df.loc[0, "n_rows"]) == 3
