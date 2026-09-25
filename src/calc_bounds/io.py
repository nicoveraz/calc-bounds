"""JSONL read/write for pydantic models."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    with path.open() as f:
        return [model.model_validate_json(line) for line in f if line.strip()]
