
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

TRACE_LEAN_PREFERRED_COLUMNS = [
    "cusip_id",
    "bond_sym_id",
    "company_symbol",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "trd_rpt_dt",
    "trd_rpt_tm",
    "msg_seq_nb",
    "trc_st",
    "scrty_type_cd",
    "wis_fl",
    "cmsn_trd",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_sign_cd",
    "yld_pt",
    "asof_cd",
    "days_to_sttl_ct",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "rpt_side_cd",
    "buy_cmsn_rt",
    "buy_cpcty_cd",
    "sell_cmsn_rt",
    "sell_cpcty_cd",
    "spcl_trd_fl",
    "trdg_mkt_cd",
    "dissem_fl",
    "orig_msg_seq_nb",
    "stlmnt_dt",
    "trd_mod_3",
    "trd_mod_4",
    "rptg_party_type",
    "ats_indicator",
]

DATE_COLUMNS = {"trd_exctn_dt", "trd_rpt_dt", "stlmnt_dt", "orig_dis_dt"}
TIME_COLUMNS = {"trd_exctn_tm", "trd_rpt_tm"}
NUMERIC_COLUMNS = {
    "msg_seq_nb",
    "orig_msg_seq_nb",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_pt",
    "days_to_sttl_ct",
    "buy_cmsn_rt",
    "sell_cmsn_rt",
}

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class TraceContract:
    name: str
    library: str
    table: str
    date_column: str
    columns: tuple[str, ...]
    quality_filters: tuple[str, ...]

    @property
    def fqname(self) -> str:
        return f"{quote_ident(self.library)}.{quote_ident(self.table)}"


@dataclass(frozen=True)
class TraceWindowTask:
    contract_name: str
    library: str
    table: str
    date_column: str
    columns: tuple[str, ...]
    quality_filters: tuple[str, ...]
    start_date: str
    end_date: str
    output_path: str
    limit: int | None = None
    chunk_rows: int = 75_000
    compression: str = "zstd"
    overwrite: bool = False
    validate_counts: bool = True
    statement_timeout_ms: int = 1_800_000

    @property
    def contract(self) -> TraceContract:
        return TraceContract(
            name=self.contract_name,
            library=self.library,
            table=self.table,
            date_column=self.date_column,
            columns=self.columns,
            quality_filters=self.quality_filters,
        )

    def to_json(self) -> dict[str, Any]:
        rec = asdict(self)
        rec["columns"] = list(self.columns)
        rec["quality_filters"] = list(self.quality_filters)
        return rec

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TraceWindowTask":
        return cls(
            contract_name=str(payload["contract_name"]),
            library=str(payload["library"]),
            table=str(payload["table"]),
            date_column=str(payload["date_column"]),
            columns=tuple(map(str, payload["columns"])),
            quality_filters=tuple(map(str, payload.get("quality_filters", []))),
            start_date=str(payload["start_date"]),
            end_date=str(payload["end_date"]),
            output_path=str(payload["output_path"]),
            limit=None if payload.get("limit") in (None, "", "None") else int(payload["limit"]),
            chunk_rows=int(payload.get("chunk_rows", 75_000)),
            compression=str(payload.get("compression", "zstd")),
            overwrite=bool(payload.get("overwrite", False)),
            validate_counts=bool(payload.get("validate_counts", True)),
            statement_timeout_ms=int(payload.get("statement_timeout_ms", 1_800_000)),
        )


def quote_ident(value: str) -> str:
    if not IDENT_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def validate_date_string(value: str) -> str:
    if not DATE_RE.match(value):
        raise ValueError(f"Expected YYYY-MM-DD date, got {value!r}")
    datetime.strptime(value, "%Y-%m-%d")
    return value


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    text = str(value).strip()[:10]
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def read_table_contracts(path: str | Path) -> dict[str, TraceContract]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = raw.get("tables", raw)
    if not isinstance(tables, dict):
        raise ValueError("configs/table_contracts.json must contain a table dictionary")
    out: dict[str, TraceContract] = {}
    for name, cfg in tables.items():
        if "trace" not in str(name).lower():
            continue
        date_column = cfg.get("date_column")
        if not date_column:
            continue
        out[str(name)] = TraceContract(
            name=str(name),
            library=str(cfg["library"]),
            table=str(cfg["table"]),
            date_column=str(date_column),
            columns=tuple(map(str, cfg.get("columns", []))),
            quality_filters=tuple(map(str, cfg.get("quality_filters", []))),
        )
    return out


def choose_columns(contract: TraceContract, mode: str) -> tuple[str, ...]:
    available = list(contract.columns)
    if mode == "full":
        return tuple(available)
    if mode == "lean":
        chosen = [c for c in TRACE_LEAN_PREFERRED_COLUMNS if c in available]
        if contract.date_column not in chosen:
            chosen.insert(0, contract.date_column)
        for required in ["cusip_id", "rptd_pr", "entrd_vol_qt"]:
            if required in available and required not in chosen:
                chosen.append(required)
        return tuple(chosen)
    raise ValueError("columns_mode must be 'lean' or 'full'")


def where_clause(contract: TraceContract, start_date: str, end_date: str) -> str:
    start_date = validate_date_string(start_date)
    end_date = validate_date_string(end_date)
    date_col = quote_ident(contract.date_column)
    conditions = [
        f"{date_col} >= DATE '{start_date}'",
        f"{date_col} < DATE '{end_date}'",
    ]
    for f in contract.quality_filters:
        text = str(f).strip()
        if text:
            conditions.append(f"({text})")
    return " and ".join(conditions)


def select_sql(task: TraceWindowTask) -> str:
    contract = task.contract
    cols = ", ".join(quote_ident(c) for c in task.columns)
    sql = f"select {cols} from {contract.fqname} where {where_clause(contract, task.start_date, task.end_date)}"
    if task.limit is not None and int(task.limit) > 0:
        sql += f" limit {int(task.limit)}"
    return sql


def count_sql(task: TraceWindowTask) -> str:
    contract = task.contract
    return f"select count(*)::bigint as n from {contract.fqname} where {where_clause(contract, task.start_date, task.end_date)}"


def daily_counts_sql(contract: TraceContract, start_date: str, end_date: str) -> str:
    start_date = validate_date_string(start_date)
    end_date = validate_date_string(end_date)
    date_col = quote_ident(contract.date_column)
    conditions = [
        f"{date_col} >= DATE '{start_date}'",
        f"{date_col} < DATE '{end_date}'",
    ]
    for f in contract.quality_filters:
        text = str(f).strip()
        if text:
            conditions.append(f"({text})")
    where = " and ".join(conditions)
    return (
        f"select {date_col}::date as trade_date, count(*)::bigint as n_rows "
        f"from {contract.fqname} where {where} group by 1 order by n_rows desc, trade_date desc"
    )


def _parse_pgpass_username(pgpass: Path) -> str | None:
    if not pgpass.exists():
        return None
    try:
        lines = pgpass.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        host, port, db, user, _password = parts[:5]
        if "wrds" in host.lower() or db == "wrds" or port == "9737":
            return user
    return None


def wrds_connect(statement_timeout_ms: int = 1_800_000):
    import psycopg2

    host = os.getenv("WRDS_HOST", "wrds-pgdata.wharton.upenn.edu")
    port = int(os.getenv("WRDS_PORT", "9737"))
    dbname = os.getenv("WRDS_DB", "wrds")
    user = os.getenv("WRDS_USERNAME") or os.getenv("PGUSER") or _parse_pgpass_username(Path.home() / ".pgpass") or os.getenv("USER")
    if not user:
        raise RuntimeError("Could not infer WRDS username. Set WRDS_USERNAME or PGUSER.")
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        sslmode=os.getenv("WRDS_SSLMODE", "require"),
        connect_timeout=int(os.getenv("WRDS_CONNECT_TIMEOUT", "60")),
        application_name="corp_bond_rv_trace_stream",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"set statement_timeout to {int(statement_timeout_ms)}")
            cur.execute("set idle_in_transaction_session_timeout to 0")
    except Exception:
        pass
    return conn


def scalar_count(task: TraceWindowTask) -> int:
    conn = wrds_connect(task.statement_timeout_ms)
    try:
        with conn.cursor() as cur:
            cur.execute(count_sql(task))
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()


def daily_counts(contract: TraceContract, *, start_date: str, end_date: str, statement_timeout_ms: int = 1_800_000) -> pd.DataFrame:
    conn = wrds_connect(statement_timeout_ms)
    try:
        with conn.cursor() as cur:
            cur.execute(daily_counts_sql(contract, start_date, end_date))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def normalize_trace_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c in DATE_COLUMNS or c.endswith("_dt") or c == "maturity":
            out[c] = pd.to_datetime(out[c], errors="coerce")
        elif c in TIME_COLUMNS or c.endswith("_tm"):
            out[c] = out[c].map(lambda x: None if pd.isna(x) else str(x)).astype("string")
        elif c in NUMERIC_COLUMNS:
            # Always use float64 at the raw TRACE layer. Several WRDS windows contain
            # integer-only chunks followed by chunks with NULL/decimal values; letting
            # pandas/Arrow infer per chunk caused Step 03e Parquet append failures.
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        elif out[c].dtype == object:
            out[c] = out[c].map(lambda x: None if pd.isna(x) else str(x)).astype("string")
    return out



def trace_arrow_schema(columns: Iterable[str]):
    """Return a stable Arrow schema for streamed TRACE chunks.

    This intentionally stores raw TRACE numeric columns as float64 to avoid per-chunk
    int64/double/null inference drift. Cleaning layers may cast identifiers/counts back
    to nullable integer types after raw extraction is complete.
    """
    import pyarrow as pa

    fields = []
    for c in columns:
        name = str(c)
        if name in DATE_COLUMNS or name.endswith("_dt") or name == "maturity":
            fields.append(pa.field(name, pa.timestamp("ns")))
        elif name in NUMERIC_COLUMNS:
            fields.append(pa.field(name, pa.float64()))
        else:
            fields.append(pa.field(name, pa.string()))
    return pa.schema(fields)

def write_empty_parquet(path: Path, columns: Iterable[str], compression: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    schema = trace_arrow_schema(columns)
    arrays = [pa.array([], type=field.type) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, tmp, compression=compression, write_statistics=True)
    tmp.replace(path)


def parquet_n_rows(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return None
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        return None


def stream_trace_to_parquet(task: TraceWindowTask) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.time()
    out_path = Path(task.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path = out_path.with_suffix(".sql")
    sql = select_sql(task)
    sql_path.write_text(sql + "\n", encoding="utf-8")

    base = {
        "contract_name": task.contract_name,
        "library": task.library,
        "table": task.table,
        "start_date": task.start_date,
        "end_date": task.end_date,
        "output_path": str(out_path),
        "limit": task.limit,
        "chunk_rows": task.chunk_rows,
        "compression": task.compression,
        "ok": False,
        "skipped": False,
        "n_rows": 0,
        "expected_rows": None,
        "count_match": None,
        "file_size_bytes": 0,
        "elapsed_sec": None,
        "rows_per_sec": None,
        "n_chunks": 0,
        "error": "",
        "returncode": None,
    }

    if out_path.exists() and out_path.stat().st_size > 0 and not task.overwrite:
        # existing output count validation for resume/scale retries
        rows = parquet_n_rows(out_path)
        base.update({
            "ok": True,
            "skipped": True,
            "n_rows": int(rows or 0),
            "file_size_bytes": int(out_path.stat().st_size),
            "elapsed_sec": round(time.time() - started, 3),
            "returncode": 0,
        })
        if task.validate_counts:
            expected_rows = scalar_count(task)
            base["expected_rows"] = int(expected_rows)
            base["count_match"] = bool(int(rows or 0) == int(expected_rows))
            base["ok"] = bool(base["count_match"] and int(rows or 0) >= 0)
            if not base["count_match"]:
                base["error"] = f"existing parquet row count {int(rows or 0)} != expected WRDS count {int(expected_rows)}"
        return base

    expected_rows = None
    if task.validate_counts:
        expected_rows = scalar_count(task)
        base["expected_rows"] = int(expected_rows)
        print(
            json.dumps({"event": "expected_rows", "start_date": task.start_date, "end_date": task.end_date, "expected_rows": int(expected_rows)}),
            flush=True,
        )

    tmp = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()

    writer = None
    total_rows = 0
    n_chunks = 0
    conn = None
    cursor = None
    try:
        conn = wrds_connect(task.statement_timeout_ms)
        cursor_name = f"trace_stream_{os.getpid()}_{uuid.uuid4().hex[:10]}"
        cursor = conn.cursor(name=cursor_name)
        cursor.itersize = max(1_000, int(task.chunk_rows))
        cursor.execute(sql)
        # WRDS/PostgreSQL named cursors may expose cursor.description as None before
        # rows are fetched. We control the SELECT list, so task.columns is the stable
        # schema source and avoids a metadata-dependent pilot/scale failure.
        colnames = list(task.columns)
        if not colnames:
            desc = cursor.description
            if desc is None:
                raise RuntimeError("cursor.description is None and task.columns is empty")
            colnames = [d[0] for d in desc]
        arrow_schema = trace_arrow_schema(colnames)

        while True:
            rows = cursor.fetchmany(int(task.chunk_rows))
            if not rows:
                break
            if len(rows[0]) != len(colnames):
                raise RuntimeError(
                    f"TRACE row width {len(rows[0])} does not match selected column count {len(colnames)}"
                )
            df = pd.DataFrame.from_records(rows, columns=colnames)
            df = normalize_trace_chunk(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = table.select(colnames).cast(arrow_schema, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp,
                    arrow_schema,
                    compression=task.compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table)
            total_rows += int(table.num_rows)
            n_chunks += 1
            if n_chunks == 1 or n_chunks % 10 == 0:
                print(
                    json.dumps({"event": "chunk_written", "start_date": task.start_date, "end_date": task.end_date, "chunks": n_chunks, "rows": total_rows}),
                    flush=True,
                )
            del df, table, rows

        if writer is not None:
            writer.close()
            writer = None
            tmp.replace(out_path)
        else:
            write_empty_parquet(out_path, colnames, task.compression)

        elapsed = time.time() - started
        actual_rows = parquet_n_rows(out_path)
        n_rows = int(actual_rows if actual_rows is not None else total_rows)
        count_match = None
        if expected_rows is not None and task.limit is None:
            count_match = int(expected_rows) == n_rows
        elif expected_rows is not None and task.limit is not None:
            count_match = n_rows <= int(task.limit) and n_rows <= int(expected_rows)

        base.update({
            "ok": bool(count_match is not False),
            "n_rows": n_rows,
            "expected_rows": expected_rows,
            "count_match": count_match,
            "file_size_bytes": int(out_path.stat().st_size),
            "elapsed_sec": round(elapsed, 3),
            "rows_per_sec": round(n_rows / elapsed, 2) if elapsed > 0 else None,
            "n_chunks": n_chunks,
            "returncode": 0 if count_match is not False else 3,
            "error": "" if count_match is not False else "row-count validation failed",
        })
        return base
    except BaseException as exc:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        elapsed = time.time() - started
        base.update({
            "ok": False,
            "expected_rows": expected_rows,
            "elapsed_sec": round(elapsed, 3),
            "error": repr(exc),
            "returncode": 2,
        })
        return base
    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def make_windows(start_date: str, end_date: str, window: str) -> list[tuple[str, str]]:
    s = parse_date(start_date)
    e = parse_date(end_date)
    if s is None or e is None or not s < e:
        raise ValueError(f"Invalid range {start_date} to {end_date}")
    if window == "day":
        step_days = 1
    elif window == "week":
        step_days = 7
    elif window == "month":
        # month windows are handled separately to keep exact month boundaries
        out = []
        cur = date(s.year, s.month, 1)
        while cur < e:
            y = cur.year + (cur.month // 12)
            m = cur.month % 12 + 1
            nxt = date(y, m, 1)
            a, b = max(cur, s), min(nxt, e)
            if a < b:
                out.append((a.isoformat(), b.isoformat()))
            cur = nxt
        return out
    else:
        raise ValueError("window must be day, week, or month")
    out = []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=step_days), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


def window_label(start_date: str, end_date: str) -> str:
    return f"{start_date}_to_{end_date}".replace("-", "")


def load_profile_bounds(profile_csv: str | Path, contract_name: str) -> tuple[str, str, int | None]:
    df = pd.read_csv(profile_csv)
    row = df.loc[(df.get("contract", df.get("contract_name")).astype(str) == contract_name)]
    if row.empty:
        raise ValueError(f"No profile row found for {contract_name} in {profile_csv}")
    r = row.iloc[0]
    min_date = parse_date(str(r.get("min_date")))
    max_date = parse_date(str(r.get("max_date")))
    if min_date is None or max_date is None:
        raise ValueError(f"Profile row for {contract_name} lacks min/max dates")
    n_rows = None
    try:
        n_rows = int(r.get("n_rows"))
    except Exception:
        pass
    return min_date.isoformat(), (max_date + timedelta(days=1)).isoformat(), n_rows


def build_tasks(
    *,
    contract: TraceContract,
    columns: tuple[str, ...],
    windows: list[tuple[str, str]],
    output_root: str | Path,
    chunk_rows: int,
    compression: str,
    overwrite: bool,
    validate_counts: bool,
    statement_timeout_ms: int,
    limit: int | None = None,
) -> list[TraceWindowTask]:
    root = Path(output_root)
    tasks = []
    for start, end in windows:
        out_path = root / contract.name / f"{contract.date_column}={window_label(start, end)}" / "part.parquet"
        tasks.append(
            TraceWindowTask(
                contract_name=contract.name,
                library=contract.library,
                table=contract.table,
                date_column=contract.date_column,
                columns=columns,
                quality_filters=contract.quality_filters,
                start_date=start,
                end_date=end,
                output_path=str(out_path),
                limit=limit,
                chunk_rows=chunk_rows,
                compression=compression,
                overwrite=overwrite,
                validate_counts=validate_counts,
                statement_timeout_ms=statement_timeout_ms,
            )
        )
    return tasks


def choose_pilot_windows(
    *,
    contract: TraceContract,
    profile_csv: str | Path,
    pilot_days: int,
    lookback_days: int,
    statement_timeout_ms: int,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    _min_date, end_exclusive, _n = load_profile_bounds(profile_csv, contract.name)
    max_d = parse_date(end_exclusive) - timedelta(days=1)  # type: ignore[operator]
    start = max_d - timedelta(days=int(lookback_days))
    counts = daily_counts(
        contract,
        start_date=start.isoformat(),
        end_date=(max_d + timedelta(days=1)).isoformat(),
        statement_timeout_ms=statement_timeout_ms,
    )
    if counts.empty:
        fallback = []
        cur = max_d
        while len(fallback) < int(pilot_days) and cur >= start:
            if cur.weekday() < 5:
                fallback.append((cur.isoformat(), (cur + timedelta(days=1)).isoformat()))
            cur -= timedelta(days=1)
        return counts, fallback
    counts["trade_date"] = pd.to_datetime(counts["trade_date"]).dt.date
    selected = counts.sort_values(["n_rows", "trade_date"], ascending=[False, False]).head(int(pilot_days))
    selected = selected.sort_values("trade_date")
    windows = [
        (d.isoformat(), (d + timedelta(days=1)).isoformat())
        for d in selected["trade_date"].tolist()
    ]
    return counts, windows


def _slug(task: TraceWindowTask, idx: int) -> str:
    s = f"{idx:04d}_{task.contract_name}_{task.start_date}_to_{task.end_date}"
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", s).strip("_")[:180]


def run_task_child(task_path: str | Path, result_path: str | Path) -> int:
    payload = json.loads(Path(task_path).read_text(encoding="utf-8"))
    task = TraceWindowTask.from_json(payload)
    result = stream_trace_to_parquet(task)
    rp = Path(result_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_name(f".{rp.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(rp)
    return 0 if bool(result.get("ok")) else int(result.get("returncode") or 2)


def _dead_child_result(task: TraceWindowTask, *, returncode: int | None, elapsed: float, stderr_path: Path) -> dict[str, Any]:
    err_tail = ""
    try:
        if stderr_path.exists():
            err_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except Exception:
        pass
    sig = None
    if returncode is not None and returncode < 0:
        try:
            sig = signal.Signals(-returncode).name
        except Exception:
            sig = f"SIGNAL_{-returncode}"
    return {
        "contract_name": task.contract_name,
        "library": task.library,
        "table": task.table,
        "start_date": task.start_date,
        "end_date": task.end_date,
        "output_path": task.output_path,
        "limit": task.limit,
        "chunk_rows": task.chunk_rows,
        "compression": task.compression,
        "ok": False,
        "skipped": False,
        "n_rows": 0,
        "expected_rows": None,
        "count_match": False,
        "file_size_bytes": 0,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": None,
        "n_chunks": 0,
        "error": f"child exited without result json; returncode={returncode}; signal={sig}; stderr_tail={err_tail}",
        "returncode": returncode,
    }


def write_manifest(manifest_path: str | Path, results: list[dict[str, Any]]) -> pd.DataFrame:
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    sort_cols = [c for c in ["contract_name", "start_date", "end_date", "output_path"] if c in df.columns]
    if sort_cols and not df.empty:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(manifest, index=False)
    manifest.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return df


def run_subprocess_tasks(
    tasks: list[TraceWindowTask],
    *,
    manifest_path: str | Path,
    workers: int,
    progress_every: int,
) -> pd.DataFrame:
    manifest = Path(manifest_path)
    payload_dir = manifest.parent / f"{manifest.stem}_task_payloads"
    result_dir = manifest.parent / f"{manifest.stem}_task_results"
    log_dir = manifest.parent / f"{manifest.stem}_child_logs"
    payload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    # Give each WRDS child a modest Arrow budget; workers should handle parallelism.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("ARROW_NUM_THREADS", "2")

    active: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    next_idx = 0
    total = len(tasks)
    workers = max(1, int(workers))

    def launch(idx: int) -> None:
        task = tasks[idx]
        slug = _slug(task, idx + 1)
        task_json = payload_dir / f"{slug}.task.json"
        result_json = result_dir / f"{slug}.result.json"
        stdout_path = log_dir / f"{slug}.stdout.log"
        stderr_path = log_dir / f"{slug}.stderr.log"
        task_json.write_text(json.dumps(task.to_json(), indent=2) + "\n", encoding="utf-8")
        out_f = stdout_path.open("w", encoding="utf-8")
        err_f = stderr_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "corpbond_rv.data.trace_stream_extract",
            "--child-task",
            str(task_json),
            "--child-result",
            str(result_json),
        ]
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True, env=env)
        active[idx] = {
            "proc": proc,
            "task": task,
            "started": time.time(),
            "result_json": result_json,
            "stdout_file": out_f,
            "stderr_file": err_f,
            "stderr_path": stderr_path,
        }

    while next_idx < total and len(active) < workers:
        launch(next_idx)
        next_idx += 1

    while active:
        finished = []
        for idx, meta in list(active.items()):
            proc: subprocess.Popen = meta["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            finished.append(idx)
            for key in ["stdout_file", "stderr_file"]:
                try:
                    meta[key].close()
                except Exception:
                    pass
            task: TraceWindowTask = meta["task"]
            result_json: Path = meta["result_json"]
            elapsed = time.time() - float(meta["started"])
            if result_json.exists():
                try:
                    result = json.loads(result_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    result = _dead_child_result(task, returncode=rc, elapsed=elapsed, stderr_path=meta["stderr_path"])
                    result["error"] = f"could not parse result json: {exc!r}; {result.get('error')}"
            else:
                result = _dead_child_result(task, returncode=rc, elapsed=elapsed, stderr_path=meta["stderr_path"])
            result["returncode"] = rc
            if rc not in (0, None) and bool(result.get("ok")):
                result["ok"] = False
                result["error"] = f"child returncode={rc} despite ok result"
            results.append(result)
            df = write_manifest(manifest, results)
            i = len(results)
            if progress_every and (i == 1 or i % progress_every == 0 or i == total):
                ok = int(df.get("ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
                failed = int((~df.get("ok", pd.Series(dtype=bool)).fillna(False)).sum()) if not df.empty else 0
                rows = int(pd.to_numeric(df.get("n_rows", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()) if not df.empty else 0
                print(f"progress {i}/{total} ok={ok} failed={failed} rows={rows:,}", flush=True)
        for idx in finished:
            active.pop(idx, None)
        while next_idx < total and len(active) < workers:
            launch(next_idx)
            next_idx += 1
        if not finished:
            time.sleep(0.25)

    return write_manifest(manifest, results)


def validate_manifest(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"ok": False, "error": "empty manifest", "n_tasks": 0}
    ok = df["ok"].fillna(False).astype(bool)
    n_rows = pd.to_numeric(df.get("n_rows", 0), errors="coerce").fillna(0)
    expected = pd.to_numeric(df.get("expected_rows", 0), errors="coerce")
    result = {
        "ok": bool(ok.all() and n_rows.sum() > 0),
        "n_tasks": int(len(df)),
        "ok_tasks": int(ok.sum()),
        "failed_tasks": int((~ok).sum()),
        "rows": int(n_rows.sum()),
        "expected_rows": int(expected.fillna(0).sum()) if "expected_rows" in df.columns else None,
        "file_size_bytes": int(pd.to_numeric(df.get("file_size_bytes", 0), errors="coerce").fillna(0).sum()),
        "mean_rows_per_sec": float(pd.to_numeric(df.get("rows_per_sec", 0), errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna().mean()) if "rows_per_sec" in df.columns else None,
    }
    if "count_match" in df.columns:
        cm = df["count_match"].dropna()
        result["count_matches"] = int(cm.astype(bool).sum())
        result["count_mismatches"] = int((~cm.astype(bool)).sum())
        if int((~cm.astype(bool)).sum()) > 0:
            result["ok"] = False
    return result


def make_pilot_artifacts(df: pd.DataFrame, counts: pd.DataFrame | None, output_dir: str | Path) -> None:
    out = Path(output_dir)
    tables_dir = out / "artifacts" / "tables"
    fig_dir = out / "artifacts" / "figures_interactive"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "step03c_trace_pilot_manifest.csv", index=False)
    if counts is not None and not counts.empty:
        counts.to_csv(tables_dir / "step03c_trace_recent_daily_counts.csv", index=False)
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        if counts is not None and not counts.empty:
            c = counts.copy()
            c["trade_date"] = pd.to_datetime(c["trade_date"])
            c = c.sort_values("trade_date")
            fig.add_trace(go.Bar(x=c["trade_date"], y=c["n_rows"], name="recent daily rows"))
            fig.update_layout(
                title="TRACE Enhanced recent daily row counts used for pilot selection",
                xaxis_title="Trade date",
                yaxis_title="Rows passing raw quality filters",
                height=520,
            )
            fig.write_html(fig_dir / "step03c_trace_recent_daily_counts.html", include_plotlyjs="cdn")
        fig2 = go.Figure()
        d = df.copy()
        d["window"] = d["start_date"].astype(str) + " → " + d["end_date"].astype(str)
        fig2.add_trace(go.Bar(x=d["window"], y=d["n_rows"], name="extracted rows"))
        if "expected_rows" in d.columns:
            fig2.add_trace(go.Scatter(x=d["window"], y=d["expected_rows"], name="expected count", mode="markers"))
        fig2.update_layout(
            title="TRACE streaming pilot extraction validation",
            xaxis_title="Pilot window",
            yaxis_title="Rows",
            height=520,
        )
        fig2.write_html(fig_dir / "step03c_trace_pilot_validation.html", include_plotlyjs="cdn")
    except Exception as exc:
        (fig_dir / "step03c_plotly_skipped.txt").write_text(f"Plotly artifacts skipped: {exc!r}\n", encoding="utf-8")


def write_recommendation(
    *,
    workspace: str | Path,
    manifest_path: str | Path,
    validation: dict[str, Any],
    contract_name: str,
    full_start: str,
    full_end: str,
    scale_window: str,
    scale_workers: int,
    chunk_rows: int,
) -> Path:
    workspace = Path(workspace)
    p = workspace / "data" / "manifests" / "extractions" / "step03c_trace_scale_recommendation.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if validation.get("ok") else "FAIL"
    text = f"""# Step 03c TRACE streaming pilot recommendation

Pilot status: **{status}**

Manifest: `{manifest_path}`

Validation summary:

```json
{json.dumps(validation, indent=2)}
```

If and only if this pilot is accepted, run the full TRACE scale once with:

```bash
cd \"{workspace}\"
python scripts/run_step03c_trace.py scale \\
  --contract {contract_name} \\
  --columns-mode lean \\
  --window {scale_window} \\
  --workers {scale_workers} \\
  --chunk-rows {chunk_rows} \\
  --start-date {full_start} \\
  --end-date {full_end} \\
  --output-root data/raw/wrds/v1 \\
  --manifest-dir data/manifests/extractions \\
  --validate-counts
```

Do not run the scale command repeatedly. The extractor skips non-empty Parquet partitions by default, but the intended workflow is pilot validation first and exactly one scale-up pass.
"""
    p.write_text(text, encoding="utf-8")
    return p


def command_pilot(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    contracts = read_table_contracts(workspace / args.contracts)
    contract = contracts[args.contract]
    columns = choose_columns(contract, args.columns_mode)
    full_start, full_end, _profile_rows = load_profile_bounds(workspace / args.profiles, args.contract)

    counts, windows = choose_pilot_windows(
        contract=contract,
        profile_csv=workspace / args.profiles,
        pilot_days=args.pilot_days,
        lookback_days=args.lookback_days,
        statement_timeout_ms=args.statement_timeout_ms,
    )
    if not windows:
        raise RuntimeError("Could not choose non-empty pilot windows")

    run_id = args.run_id or datetime.now(UTC).strftime("pilot_%Y%m%dT%H%M%SZ")
    manifest_path = workspace / args.manifest_dir / f"step03c_trace_{run_id}.csv"
    output_root = workspace / args.output_root
    tasks = build_tasks(
        contract=contract,
        columns=columns,
        windows=windows,
        output_root=output_root,
        chunk_rows=args.chunk_rows,
        compression=args.compression,
        overwrite=args.overwrite,
        validate_counts=args.validate_counts,
        statement_timeout_ms=args.statement_timeout_ms,
        limit=args.limit,
    )
    plan_path = manifest_path.with_name(f"{manifest_path.stem}_plan.csv")
    pd.DataFrame([t.to_json() for t in tasks]).to_csv(plan_path, index=False)
    print(json.dumps({
        "mode": "pilot",
        "run_id": run_id,
        "contract": args.contract,
        "columns_mode": args.columns_mode,
        "n_columns": len(columns),
        "windows": windows,
        "workers": args.workers,
        "chunk_rows": args.chunk_rows,
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
    }, indent=2))
    df = run_subprocess_tasks(tasks, manifest_path=manifest_path, workers=args.workers, progress_every=args.progress_every)
    validation = validate_manifest(df)
    validation_path = manifest_path.with_name(f"{manifest_path.stem}_validation.json")
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8")
    make_pilot_artifacts(df, counts, workspace)
    rec = write_recommendation(
        workspace=workspace,
        manifest_path=manifest_path,
        validation=validation,
        contract_name=args.contract,
        full_start=full_start,
        full_end=full_end,
        scale_window=args.scale_window,
        scale_workers=args.scale_workers,
        chunk_rows=args.scale_chunk_rows or args.chunk_rows,
    )
    print(json.dumps({"validation": validation, "validation_path": str(validation_path), "recommendation": str(rec)}, indent=2, default=str))
    return 0 if validation.get("ok") else 1


def command_scale(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    contracts = read_table_contracts(workspace / args.contracts)
    contract = contracts[args.contract]
    columns = choose_columns(contract, args.columns_mode)
    full_start, full_end, _profile_rows = load_profile_bounds(workspace / args.profiles, args.contract)
    start = args.start_date or full_start
    end = args.end_date or full_end
    windows = make_windows(start, end, args.window)
    run_id = args.run_id or datetime.now(UTC).strftime("scale_%Y%m%dT%H%M%SZ")
    manifest_path = workspace / args.manifest_dir / f"step03c_trace_{run_id}.csv"
    tasks = build_tasks(
        contract=contract,
        columns=columns,
        windows=windows,
        output_root=workspace / args.output_root,
        chunk_rows=args.chunk_rows,
        compression=args.compression,
        overwrite=args.overwrite,
        validate_counts=args.validate_counts,
        statement_timeout_ms=args.statement_timeout_ms,
        limit=args.limit,
    )
    plan_path = manifest_path.with_name(f"{manifest_path.stem}_plan.csv")
    pd.DataFrame([t.to_json() for t in tasks]).to_csv(plan_path, index=False)
    print(json.dumps({
        "mode": "scale",
        "run_id": run_id,
        "contract": args.contract,
        "window": args.window,
        "start_date": start,
        "end_date": end,
        "n_tasks": len(tasks),
        "workers": args.workers,
        "chunk_rows": args.chunk_rows,
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
    }, indent=2))
    df = run_subprocess_tasks(tasks, manifest_path=manifest_path, workers=args.workers, progress_every=args.progress_every)
    validation = validate_manifest(df)
    validation_path = manifest_path.with_name(f"{manifest_path.stem}_validation.json")
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "validation_path": str(validation_path)}, indent=2, default=str))
    return 0 if validation.get("ok") else 1


def command_plan(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    contracts = read_table_contracts(workspace / args.contracts)
    contract = contracts[args.contract]
    columns = choose_columns(contract, args.columns_mode)
    full_start, full_end, profile_rows = load_profile_bounds(workspace / args.profiles, args.contract)
    start = args.start_date or full_start
    end = args.end_date or full_end
    windows = make_windows(start, end, args.window)
    out = Path(args.output) if args.output else workspace / args.manifest_dir / f"step03c_trace_scale_plan_{args.window}.csv"
    rows = [
        {
            "contract_name": contract.name,
            "start_date": a,
            "end_date": b,
            "n_columns": len(columns),
            "output_path": str(Path(args.output_root) / contract.name / f"{contract.date_column}={window_label(a,b)}" / "part.parquet"),
        }
        for a, b in windows
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({
        "mode": "plan",
        "contract": contract.name,
        "profile_rows": profile_rows,
        "start_date": start,
        "end_date": end,
        "window": args.window,
        "n_tasks": len(windows),
        "n_columns": len(columns),
        "plan_path": str(out),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Streaming TRACE extractor for Step 03c pilot/scale workflow.")
    p.add_argument("mode", choices=["pilot", "scale", "plan"])
    p.add_argument("--workspace", default=".")
    p.add_argument("--contracts", default="configs/table_contracts.json")
    p.add_argument("--profiles", default="data/manifests/wrds_profiles/table_profiles.csv")
    p.add_argument("--manifest-dir", default="data/manifests/extractions")
    p.add_argument("--output-root", default="data/raw/wrds/v1")
    p.add_argument("--contract", default="trace_enhanced_clean")
    p.add_argument("--columns-mode", choices=["lean", "full"], default="lean")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--chunk-rows", type=int, default=75_000)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--statement-timeout-ms", type=int, default=1_800_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-counts", action="store_true")
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--run-id", default=None)
    p.add_argument("--pilot-days", type=int, default=3)
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument("--window", choices=["day", "week", "month"], default="week")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--scale-window", choices=["day", "week", "month"], default="week")
    p.add_argument("--scale-workers", type=int, default=6)
    p.add_argument("--scale-chunk-rows", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--child-task", default=None)
    p.add_argument("--child-result", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Child mode is intentionally hidden from help; it avoids import-time side effects in parent tasks.
    if "--child-task" in argv:
        pa = build_parser()
        ns = pa.parse_args(["pilot", *argv])
        if not ns.child_task or not ns.child_result:
            raise SystemExit("--child-task and --child-result are required")
        return run_task_child(ns.child_task, ns.child_result)
    args = build_parser().parse_args(argv)
    if args.mode == "pilot":
        return command_pilot(args)
    if args.mode == "scale":
        return command_scale(args)
    if args.mode == "plan":
        return command_plan(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
