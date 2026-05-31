from corpbond_rv.data.table_contracts import load_contracts
from corpbond_rv.data.wrds_sql import build_select_sql
from corpbond_rv.utils.paths import project_root


def test_step02_contracts_load_and_generate_sql():
    root = project_root()
    contracts = load_contracts(root / "configs" / "table_contracts.json")
    assert "trace_enhanced_clean" in contracts
    assert contracts["trace_enhanced_clean"].fqname == "wrdsapps_bondret.trace_enhanced_clean"
    sql = build_select_sql(contracts["bondret_monthly"], start_date="2020-01-01", end_date="2021-01-01", limit=10)
    assert "wrdsapps_bondret.bondret" in sql
    assert "date >= '2020-01-01'" in sql
    assert "limit 10" in sql
