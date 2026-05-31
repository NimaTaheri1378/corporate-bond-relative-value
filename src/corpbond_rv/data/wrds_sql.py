from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from corpbond_rv.data.table_contracts import TableContract


def sql_literal(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return "'" + value.isoformat() + "'"
    safe = str(value).replace("'", "''")
    return f"'{safe}'"


def fq_table(contract: TableContract) -> str:
    # WRDS libraries/tables discovered here are lower-case identifiers.
    return f"{contract.library}.{contract.table}"


def select_list(contract: TableContract) -> str:
    return ", ".join(contract.columns) if contract.columns else "*"


def build_where(
    contract: TableContract,
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    extra_filters: Iterable[str] | None = None,
) -> str:
    filters: list[str] = []
    if contract.quality_filters:
        filters.extend(contract.quality_filters)
    if extra_filters:
        filters.extend([f for f in extra_filters if str(f).strip()])
    if contract.date_column and start_date:
        filters.append(f"{contract.date_column} >= {sql_literal(start_date)}")
    if contract.date_column and end_date:
        filters.append(f"{contract.date_column} < {sql_literal(end_date)}")
    if not filters:
        return ""
    return " where " + " and ".join(f"({f})" for f in filters)


def build_select_sql(
    contract: TableContract,
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    limit: int | None = None,
    extra_filters: Iterable[str] | None = None,
) -> str:
    sql = f"select {select_list(contract)} from {fq_table(contract)}"
    sql += build_where(contract, start_date=start_date, end_date=end_date, extra_filters=extra_filters)
    if limit is not None and limit > 0:
        sql += f" limit {int(limit)}"
    return sql


def build_profile_sql(contract: TableContract) -> str:
    if contract.date_column:
        return (
            f"select count(*)::bigint as n_rows, "
            f"min({contract.date_column}) as min_date, max({contract.date_column}) as max_date "
            f"from {fq_table(contract)}"
        )
    return f"select count(*)::bigint as n_rows from {fq_table(contract)}"


def build_sample_sql(contract: TableContract, n: int = 5) -> str:
    if contract.date_column:
        return (
            f"select {select_list(contract)} from {fq_table(contract)} "
            f"where {contract.date_column} is not null "
            f"order by {contract.date_column} desc limit {int(n)}"
        )
    return f"select {select_list(contract)} from {fq_table(contract)} limit {int(n)}"


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str

    @property
    def label(self) -> str:
        return f"{self.start}_to_{self.end}".replace("-", "")


def yearly_windows(start: str, end: str) -> list[DateWindow]:
    s_year = int(start[:4])
    e_year = int(end[:4])
    out: list[DateWindow] = []
    for year in range(s_year, e_year + 1):
        a = max(f"{year:04d}-01-01", start)
        b = min(f"{year + 1:04d}-01-01", end)
        if a < b:
            out.append(DateWindow(a, b))
    return out


def monthly_windows(start: str, end: str) -> list[DateWindow]:
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out: list[DateWindow] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        if m == 12:
            ny, nm = y + 1, 1
        else:
            ny, nm = y, m + 1
        a = max(f"{y:04d}-{m:02d}-01", start)
        b = min(f"{ny:04d}-{nm:02d}-01", end)
        if a < b:
            out.append(DateWindow(a, b))
        y, m = ny, nm
    return out


def windows_for_partition(start: str, end: str, partition: str) -> list[DateWindow]:
    if partition == "month":
        return monthly_windows(start, end)
    return yearly_windows(start, end)
