from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from corpbond_rv.data.table_contracts import TableContract, load_contracts
from corpbond_rv.data.wrds_extract import ExtractTask

TRACE_CONTRACTS = {"trace_enhanced_clean", "trace_standard_clean"}
# Do not single-shot dated tables. Even when row counts are moderate,
# partitioned pulls are safer on shared clusters and make reruns resumable.
SINGLE_SHOT_DATED_CORE_CONTRACTS: set[str] = set()
# These are larger dated panels. Use smaller windows than annual pulls to
# avoid giant in-memory DataFrames and to make resume very granular.
CORE_WINDOWED_CONTRACTS = {
    "bondret_monthly",
    "contrib_bond_returns",
    "fang_bond_firm_link",
}
CORE_CONTRACTS = {
    "bondret_monthly",
    "contrib_bond_returns",
    "fisd_issue_issuer",
    "fisd_coupon_info",
    "fisd_rating_hist",
    "fisd_amount_outstanding",
    "crsp_bond_link",
    "fang_bond_firm_link",
}


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str

    @property
    def label(self) -> str:
        return f"{self.start}_to_{self.end}".replace("-", "")


@dataclass(frozen=True)
class ContractBounds:
    min_date: str | None
    max_date: str | None
    n_rows: int | None


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _date_str(d: date | None) -> str | None:
    return None if d is None else d.isoformat()


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)


def _floor_for_window(d: date, window: str) -> date:
    if window == "month":
        return date(d.year, d.month, 1)
    if window == "quarter":
        q_month = ((d.month - 1) // 3) * 3 + 1
        return date(d.year, q_month, 1)
    return date(d.year, 1, 1)


def _ceil_exclusive_for_window(d: date, window: str) -> date:
    # The end bound is exclusive; one day after max observed date is safe
    # and avoids accidentally requesting months beyond available data.
    return d + timedelta(days=1)


def windows_for_granularity(start: str, end: str, granularity: str) -> list[DateWindow]:
    if granularity not in {"month", "quarter", "year"}:
        raise ValueError(f"Unsupported window granularity: {granularity}")
    s = _parse_date(start)
    e = _parse_date(end)
    if s is None or e is None or not (s < e):
        raise ValueError(f"Invalid date window: {start} to {end}")
    step = {"month": 1, "quarter": 3, "year": 12}[granularity]
    cur = _floor_for_window(s, granularity)
    out: list[DateWindow] = []
    while cur < e:
        nxt = _add_months(cur, step)
        a = max(cur, s)
        b = min(nxt, e)
        if a < b:
            out.append(DateWindow(a.isoformat(), b.isoformat()))
        cur = nxt
    return out


def load_profile_bounds(profile_csv: str | Path) -> dict[str, ContractBounds]:
    path = Path(profile_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, ContractBounds] = {}
    for _, row in df.iterrows():
        name = str(row.get("contract", row.get("contract_name", ""))).strip()
        if not name:
            continue
        n_rows = None
        try:
            value = row.get("n_rows")
            if value is not None and not pd.isna(value):
                n_rows = int(value)
        except Exception:
            n_rows = None
        out[name] = ContractBounds(
            min_date=_date_str(_parse_date(row.get("min_date"))),
            max_date=_date_str(_parse_date(row.get("max_date"))),
            n_rows=n_rows,
        )
    return out


def names_for_phase(
    phase: str,
    contracts: dict[str, TableContract],
    *,
    include_disabled: bool = False,
    include_fallback_trace: bool = False,
) -> set[str]:
    available = set(contracts)
    if not include_disabled:
        available = {name for name in available if contracts[name].enabled_by_default}
    if phase == "core":
        return available.intersection(CORE_CONTRACTS)
    if phase == "trace":
        names = {"trace_enhanced_clean"}
        if include_fallback_trace:
            names.add("trace_standard_clean")
        return available.intersection(names)
    if phase in {"all", "plan"}:
        names = set(CORE_CONTRACTS).union({"trace_enhanced_clean"})
        if include_fallback_trace:
            names.add("trace_standard_clean")
        return available.intersection(names)
    raise ValueError(f"Unknown phase: {phase}")


def default_date_range_for_contract(
    contract: TableContract,
    bounds: dict[str, ContractBounds],
    *,
    window_granularity: str,
    override_start: str | None = None,
    override_end: str | None = None,
) -> tuple[str | None, str | None]:
    if not contract.date_column or contract.partition == "none":
        return None, None
    b = bounds.get(contract.name)
    start_d = _parse_date(override_start) if override_start else _parse_date(b.min_date if b else None)
    if override_end:
        end_d = _parse_date(override_end)
    else:
        max_d = _parse_date(b.max_date if b else None)
        end_d = _ceil_exclusive_for_window(max_d, window_granularity) if max_d else None
    if start_d is None or end_d is None:
        raise ValueError(
            f"No date bounds for dated contract {contract.name}. Pass explicit start/end "
            "or refresh data/manifests/wrds_profiles/table_profiles.csv."
        )
    start_d = _floor_for_window(start_d, window_granularity)
    if not (start_d < end_d):
        raise ValueError(f"Invalid date range for {contract.name}: {start_d} to {end_d}")
    return start_d.isoformat(), end_d.isoformat()


def build_extraction_tasks(
    contracts_path: str | Path,
    profile_csv: str | Path,
    output_root: str | Path,
    *,
    phase: str,
    include_disabled: bool = False,
    include_fallback_trace: bool = False,
    trace_pilot_months: int = 0,
    trace_start_date: str | None = None,
    trace_end_date: str | None = None,
    core_start_date: str | None = None,
    core_end_date: str | None = None,
    core_window: str = "quarter",
    trace_window: str = "month",
    limit: int | None = None,
) -> list[ExtractTask]:
    contracts = load_contracts(contracts_path)
    bounds = load_profile_bounds(profile_csv)
    selected = names_for_phase(
        phase,
        contracts,
        include_disabled=include_disabled,
        include_fallback_trace=include_fallback_trace,
    )
    root = Path(output_root)
    tasks: list[ExtractTask] = []
    for name in sorted(selected, key=lambda n: (contracts[n].priority, n)):
        contract = contracts[name]
        is_trace = name in TRACE_CONTRACTS
        single_shot = (
            not contract.date_column
            or contract.partition == "none"
            or (name in SINGLE_SHOT_DATED_CORE_CONTRACTS and not is_trace)
        )
        if single_shot:
            out_path = root / contract.name / "full" / "part.parquet"
            tasks.append(ExtractTask.from_contract(contract, output_path=out_path, limit=limit))
            continue

        window_granularity = trace_window if is_trace else core_window
        if name in {"fang_bond_firm_link", "fisd_amount_outstanding", "fisd_rating_hist"} and not is_trace:
            # Historical link/rating/amount tables are naturally yearly.
            # Yearly windows keep memory low without creating excessive quarter tasks.
            window_granularity = "year"

        start, end = default_date_range_for_contract(
            contract,
            bounds,
            window_granularity=window_granularity,
            override_start=trace_start_date if is_trace and trace_start_date else core_start_date,
            override_end=trace_end_date if is_trace and trace_end_date else core_end_date,
        )
        windows = windows_for_granularity(start, end, window_granularity)
        if is_trace and trace_pilot_months and trace_pilot_months > 0:
            windows = windows[-int(trace_pilot_months):]
        for window in windows:
            out_path = (
                root
                / contract.name
                / f"{contract.date_column}={window.label}"
                / "part.parquet"
            )
            tasks.append(
                ExtractTask.from_contract(
                    contract,
                    output_path=out_path,
                    start_date=window.start,
                    end_date=window.end,
                    limit=limit,
                )
            )
    return tasks


def tasks_to_frame(tasks: Iterable[ExtractTask]) -> pd.DataFrame:
    rows = [task.to_record() for task in tasks]
    if not rows:
        return pd.DataFrame()
    cols = [
        "contract_name", "role", "library", "table", "date_column", "partition",
        "start_date", "end_date", "limit", "output_path", "quality_filters", "columns",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)
