from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TableContract:
    name: str
    role: str
    priority: int
    library: str
    table: str
    date_column: str | None = None
    partition: str = "none"
    description: str = ""
    columns: tuple[str, ...] = field(default_factory=tuple)
    required_columns: tuple[str, ...] = field(default_factory=tuple)
    quality_filters: tuple[str, ...] = field(default_factory=tuple)
    enabled_by_default: bool = True

    @property
    def fqname(self) -> str:
        return f"{self.library}.{self.table}"

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "TableContract":
        return cls(
            name=name,
            role=str(data.get("role", "")),
            priority=int(data.get("priority", 999)),
            library=str(data["library"]),
            table=str(data["table"]),
            date_column=data.get("date_column"),
            partition=str(data.get("partition", "none")),
            description=str(data.get("description", "")),
            columns=tuple(map(str, data.get("columns", []))),
            required_columns=tuple(map(str, data.get("required_columns", []))),
            quality_filters=tuple(map(str, data.get("quality_filters", []))),
            enabled_by_default=bool(data.get("enabled_by_default", True)),
        )


def load_contract_payload(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"PyYAML is required to read {p}; use configs/table_contracts.json instead."
        ) from exc
    return yaml.safe_load(text)


def load_contracts(path: str | Path) -> dict[str, TableContract]:
    payload = load_contract_payload(path)
    tables = payload.get("tables", {})
    contracts = {
        name: TableContract.from_mapping(name, spec)
        for name, spec in tables.items()
    }
    return dict(sorted(contracts.items(), key=lambda kv: (kv[1].priority, kv[0])))


def enabled_contracts(path: str | Path) -> dict[str, TableContract]:
    return {k: v for k, v in load_contracts(path).items() if v.enabled_by_default}


def load_schema_inventory(schema_csv: str | Path) -> pd.DataFrame:
    p = Path(schema_csv)
    if not p.exists():
        raise FileNotFoundError(f"Schema inventory not found: {p}")
    schema = pd.read_csv(p)
    required = {"library", "table", "name"}
    missing = required.difference(schema.columns)
    if missing:
        raise ValueError(f"Schema inventory missing columns: {sorted(missing)}")
    return schema


def validate_contracts_against_schema(
    contracts: dict[str, TableContract], schema_csv: str | Path
) -> pd.DataFrame:
    schema = load_schema_inventory(schema_csv)
    rows: list[dict[str, object]] = []
    for contract in contracts.values():
        sub = schema[(schema["library"] == contract.library) & (schema["table"] == contract.table)]
        available = set(sub["name"].astype(str))
        requested = set(contract.columns)
        required = set(contract.required_columns)
        missing_requested = sorted(requested.difference(available))
        missing_required = sorted(required.difference(available))
        rows.append(
            {
                "contract": contract.name,
                "role": contract.role,
                "fqname": contract.fqname,
                "table_exists": bool(len(sub)),
                "n_available_columns": int(len(available)),
                "n_requested_columns": int(len(requested)),
                "missing_requested_columns": ",".join(missing_requested),
                "missing_required_columns": ",".join(missing_required),
                "valid": bool(len(sub)) and not missing_required,
            }
        )
    return pd.DataFrame(rows)
