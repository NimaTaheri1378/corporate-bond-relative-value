from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from corpbond_rv.data.table_contracts import TableContract
from corpbond_rv.data.wrds_sql import build_select_sql


@dataclass(frozen=True)
class ExtractTask:
    contract_name: str
    library: str
    table: str
    role: str
    date_column: str | None
    partition: str
    columns: tuple[str, ...]
    quality_filters: tuple[str, ...]
    output_path: str
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = None

    @classmethod
    def from_contract(
        cls,
        contract: TableContract,
        *,
        output_path: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> "ExtractTask":
        return cls(
            contract_name=contract.name,
            library=contract.library,
            table=contract.table,
            role=contract.role,
            date_column=contract.date_column,
            partition=contract.partition,
            columns=tuple(contract.columns),
            quality_filters=tuple(contract.quality_filters),
            output_path=str(output_path),
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ExtractTask":
        return cls(
            contract_name=str(payload["contract_name"]),
            library=str(payload["library"]),
            table=str(payload["table"]),
            role=str(payload.get("role", "")),
            date_column=payload.get("date_column"),
            partition=str(payload.get("partition", "none")),
            columns=tuple(map(str, payload.get("columns", []))),
            quality_filters=tuple(map(str, payload.get("quality_filters", []))),
            output_path=str(payload["output_path"]),
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            limit=None if payload.get("limit") in (None, "", "nan") else int(payload.get("limit")),
        )

    def to_contract(self) -> TableContract:
        return TableContract(
            name=self.contract_name,
            role=self.role,
            priority=1,
            library=self.library,
            table=self.table,
            date_column=self.date_column,
            partition=self.partition,
            columns=tuple(self.columns),
            quality_filters=tuple(self.quality_filters),
            enabled_by_default=True,
        )

    def to_json_mapping(self) -> dict[str, object]:
        rec = asdict(self)
        rec["columns"] = list(self.columns)
        rec["quality_filters"] = list(self.quality_filters)
        return rec

    def to_record(self) -> dict[str, object]:
        rec = asdict(self)
        rec["columns"] = ",".join(self.columns)
        rec["quality_filters"] = " AND ".join(self.quality_filters)
        return rec


def parquet_n_rows(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        return None


def stable_sql_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False, compression=compression)
    tmp.replace(path)


def _safe_wrds_connection():
    import contextlib
    import io
    import wrds

    with contextlib.redirect_stdout(io.StringIO()):
        return wrds.Connection()


def _task_base_record(task: ExtractTask) -> dict[str, object]:
    out: dict[str, object] = task.to_record()
    out.update(
        {
            "ok": False,
            "skipped": False,
            "n_rows": 0,
            "file_size_bytes": 0,
            "elapsed_sec": None,
            "attempts": 0,
            "error": "",
            "sql_sha256": "",
            "returncode": None,
        }
    )
    return out


def run_extract_task(
    task: ExtractTask,
    *,
    compression: str = "zstd",
    overwrite: bool = False,
    retries: int = 2,
    retry_sleep_seconds: float = 10.0,
) -> dict[str, object]:
    started = time.time()
    out = _task_base_record(task)
    path = Path(task.output_path)
    contract = task.to_contract()
    sql = build_select_sql(
        contract,
        start_date=task.start_date,
        end_date=task.end_date,
        limit=task.limit,
    )
    out["sql_sha256"] = stable_sql_hash(sql)
    sql_path = path.with_suffix(".sql")
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sql + "\n", encoding="utf-8")

    if path.exists() and path.stat().st_size > 0 and not overwrite:
        rows = parquet_n_rows(path)
        out.update(
            {
                "ok": True,
                "skipped": True,
                "n_rows": int(rows or 0),
                "file_size_bytes": int(path.stat().st_size),
                "elapsed_sec": round(time.time() - started, 3),
                "returncode": 0,
            }
        )
        return out

    last_error = ""
    for attempt in range(1, int(retries) + 2):
        out["attempts"] = attempt
        conn = None
        try:
            conn = _safe_wrds_connection()
            df = conn.raw_sql(sql)
            _write_parquet_atomic(df, path, compression=compression)
            rows = parquet_n_rows(path)
            out.update(
                {
                    "ok": True,
                    "skipped": False,
                    "n_rows": int(rows if rows is not None else len(df)),
                    "file_size_bytes": int(path.stat().st_size),
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": "",
                    "returncode": 0,
                }
            )
            return out
        except BaseException as exc:  # pragma: no cover - WRDS environment only
            last_error = repr(exc)
            out["error"] = last_error
            if attempt <= int(retries):
                time.sleep(float(retry_sleep_seconds) * attempt)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    out["elapsed_sec"] = round(time.time() - started, 3)
    out["error"] = last_error
    out["returncode"] = 2
    return out


def _sanitize_slug(text: str, max_len: int = 150) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return clean[:max_len] or "task"


def _task_slug(task: ExtractTask, idx: int) -> str:
    label = f"{idx:04d}_{task.contract_name}"
    if task.start_date or task.end_date:
        label += f"_{task.start_date or 'start'}_to_{task.end_date or 'end'}"
    else:
        label += "_full"
    return _sanitize_slug(label)


def _write_manifest(df: pd.DataFrame, manifest: Path, results: list[dict[str, object]]) -> None:
    sort_cols = [c for c in ["contract_name", "start_date", "end_date", "output_path"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(manifest, index=False)
    df.to_csv(manifest.with_suffix(".partial.csv"), index=False)
    manifest.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")


def _dead_child_result(task: ExtractTask, *, returncode: int | None, stderr_path: Path, elapsed_sec: float) -> dict[str, object]:
    out = _task_base_record(task)
    stderr_tail = ""
    try:
        if stderr_path.exists():
            text = stderr_path.read_text(encoding="utf-8", errors="replace")
            stderr_tail = text[-4000:]
    except Exception:
        stderr_tail = ""
    out.update(
        {
            "elapsed_sec": round(elapsed_sec, 3),
            "returncode": returncode,
            "error": f"child process exited without result json; returncode={returncode}; stderr_tail={stderr_tail}",
        }
    )
    return out


def run_tasks_sequential(
    tasks: list[ExtractTask],
    *,
    manifest: Path,
    compression: str,
    overwrite: bool,
    retries: int,
    retry_sleep_seconds: float,
    progress_every: int,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    for i, task in enumerate(tasks, start=1):
        res = run_extract_task(
            task,
            compression=compression,
            overwrite=overwrite,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        results.append(res)
        df = pd.DataFrame(results)
        _write_manifest(df, manifest, results)
        if progress_every and (i == 1 or i % progress_every == 0 or i == len(tasks)):
            ok = sum(bool(r.get("ok")) for r in results)
            skipped = sum(bool(r.get("skipped")) for r in results)
            rows = sum(int(r.get("n_rows") or 0) for r in results)
            print(f"progress {i}/{len(tasks)} ok={ok} skipped={skipped} rows={rows:,}", flush=True)
    return pd.DataFrame(results)


def run_tasks_subprocess(
    tasks: list[ExtractTask],
    *,
    manifest: Path,
    workers: int,
    compression: str,
    overwrite: bool,
    retries: int,
    retry_sleep_seconds: float,
    progress_every: int,
) -> pd.DataFrame:
    payload_dir = manifest.parent / f"{manifest.stem}_task_payloads"
    child_log_dir = manifest.parent / f"{manifest.stem}_child_logs"
    payload_dir.mkdir(parents=True, exist_ok=True)
    child_log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    max_workers = max(1, int(workers))
    next_idx = 0
    active: dict[int, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    total = len(tasks)

    def launch(idx: int) -> None:
        task = tasks[idx]
        slug = _task_slug(task, idx + 1)
        task_json = payload_dir / f"{slug}.task.json"
        result_json = payload_dir / f"{slug}.result.json"
        stdout_path = child_log_dir / f"{slug}.stdout.log"
        stderr_path = child_log_dir / f"{slug}.stderr.log"
        task_json.write_text(json.dumps(task.to_json_mapping(), indent=2) + "\n", encoding="utf-8")
        out_f = stdout_path.open("w", encoding="utf-8")
        err_f = stderr_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "corpbond_rv.data.wrds_extract",
            "--child-extract",
            "--task-json",
            str(task_json),
            "--result-json",
            str(result_json),
            "--compression",
            compression,
            "--retries",
            str(int(retries)),
            "--retry-sleep-seconds",
            str(float(retry_sleep_seconds)),
        ]
        if overwrite:
            cmd.append("--overwrite")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True, env=env)
        active[idx] = {
            "proc": proc,
            "task": task,
            "result_json": result_json,
            "stdout_file": out_f,
            "stderr_file": err_f,
            "stderr_path": stderr_path,
            "started": time.time(),
        }

    while next_idx < total and len(active) < max_workers:
        launch(next_idx)
        next_idx += 1

    while active:
        finished: list[int] = []
        for idx, meta in list(active.items()):
            proc: subprocess.Popen = meta["proc"]  # type: ignore[assignment]
            rc = proc.poll()
            if rc is None:
                continue
            finished.append(idx)
            for key in ["stdout_file", "stderr_file"]:
                try:
                    meta[key].close()  # type: ignore[index, union-attr]
                except Exception:
                    pass
            task: ExtractTask = meta["task"]  # type: ignore[assignment]
            result_json: Path = meta["result_json"]  # type: ignore[assignment]
            elapsed = time.time() - float(meta["started"])
            if result_json.exists():
                try:
                    res = json.loads(result_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    res = _dead_child_result(
                        task,
                        returncode=rc,
                        stderr_path=meta["stderr_path"],  # type: ignore[arg-type]
                        elapsed_sec=elapsed,
                    )
                    res["error"] = f"could not parse child result json: {exc!r}; {res.get('error')}"
            else:
                res = _dead_child_result(
                    task,
                    returncode=rc,
                    stderr_path=meta["stderr_path"],  # type: ignore[arg-type]
                    elapsed_sec=elapsed,
                )
            if rc not in (0, None) and bool(res.get("ok")):
                res["ok"] = False
                res["error"] = f"child returncode={rc} despite ok result"
            res["returncode"] = rc
            results.append(res)
            df = pd.DataFrame(results)
            _write_manifest(df, manifest, results)
            i = len(results)
            if progress_every and (i == 1 or i % progress_every == 0 or i == total):
                ok = sum(bool(r.get("ok")) for r in results)
                skipped = sum(bool(r.get("skipped")) for r in results)
                rows = sum(int(r.get("n_rows") or 0) for r in results)
                failed = sum(not bool(r.get("ok")) for r in results)
                print(
                    f"progress {i}/{total} ok={ok} failed={failed} skipped={skipped} rows={rows:,}",
                    flush=True,
                )
        for idx in finished:
            active.pop(idx, None)
        while next_idx < total and len(active) < max_workers:
            launch(next_idx)
            next_idx += 1
        if not finished:
            time.sleep(0.25)

    df = pd.DataFrame(results)
    _write_manifest(df, manifest, results)
    return df


def run_tasks(
    tasks: list[ExtractTask],
    *,
    manifest_path: str | Path,
    workers: int,
    compression: str = "zstd",
    overwrite: bool = False,
    retries: int = 2,
    retry_sleep_seconds: float = 10.0,
    sequential: bool = False,
    engine: str = "subprocess",
    executor: str | None = None,
    progress_every: int = 5,
) -> pd.DataFrame:
    if executor is not None:
        engine = executor
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not tasks:
        df = pd.DataFrame()
        _write_manifest(df, manifest, [])
        return df

    if sequential or workers <= 1 or engine in {"sequential", "thread"}:
        df = run_tasks_sequential(
            tasks,
            manifest=manifest,
            compression=compression,
            overwrite=overwrite,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
            progress_every=progress_every,
        )
        df["executor"] = engine
        _write_manifest(df, manifest, df.to_dict("records"))
        return df
    if engine != "subprocess":
        raise ValueError("Step 03b supports engine='subprocess', 'thread', or 'sequential'.")
    df = run_tasks_subprocess(
        tasks,
        manifest=manifest,
        workers=workers,
        compression=compression,
        overwrite=overwrite,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
        progress_every=progress_every,
    )
    df["executor"] = engine
    _write_manifest(df, manifest, df.to_dict("records"))
    return df


def default_workers(value: int | None = None, *, cap: int = 8, fallback: int = 4) -> int:
    if value is not None and int(value) > 0:
        return max(1, min(int(value), int(cap)))
    env = os.getenv("WRDS_PARALLEL_WORKERS") or os.getenv("RV_WRDS_WORKERS")
    if env:
        try:
            return max(1, min(int(env), int(cap)))
        except ValueError:
            pass
    return max(1, min(int(fallback), int(cap)))


def _read_task(path: str | Path) -> ExtractTask:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExtractTask.from_mapping(payload)


def child_extract_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one WRDS extraction task child process.")
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    task = _read_task(args.task_json)
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    res = run_extract_task(
        task,
        compression=args.compression,
        overwrite=args.overwrite,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    tmp = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(result_path)
    return 0 if bool(res.get("ok")) else 2


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--child-extract":
        return child_extract_main(argv[1:])
    parser = argparse.ArgumentParser(description="WRDS extraction utility module.")
    parser.add_argument("--child-extract", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if args.child_extract:
        return child_extract_main(rest)
    parser.error("This module is intended to be used through scripts/run_step03_extract.py")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
