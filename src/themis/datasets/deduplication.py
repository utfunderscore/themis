"""Reusable chat dataset deduplication."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import polars as pl


def deduplicate_chats(source: Path | pl.LazyFrame, destination: Path) -> Path:
    """Keep the first record for each exact content string, without ordering rows.

    The destination is replaced atomically, so invalid input or a failed write
    leaves any existing output untouched.
    """
    if isinstance(source, Path) and source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix="deduplicate-", suffix=".jsonl"
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        if isinstance(source, Path):
            # Polars coerces mistyped content and skips blank lines, so check first.
            has_rows = _validate_chats(source)
            # Infer from every row so fields that first appear late are kept.
            chats = pl.scan_ndjson(source, infer_schema_length=None) if has_rows else None
        else:
            chats = source
        if chats is not None:
            deduplicated = chats.unique(subset=["content"], keep="first")
            _ = deduplicated.sink_ndjson(temporary_path)
        _ = temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    with destination.open("rb") as output:
        total = sum(1 for _ in output)
    print(f"Deduplicated dataset contains {total:,} rows", flush=True)
    return destination


def _validate_chats(source: Path) -> bool:
    """Raise ``ValueError`` unless every line is an object with string content."""
    has_rows = False
    with source.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            row = cast(object, json.loads(line))  # JSONDecodeError is a ValueError.
            if not isinstance(row, dict) or not isinstance(row.get("content"), str):  # pyright: ignore[reportUnknownMemberType]
                raise ValueError(f"{source}:{number}: expected an object with string content")
            has_rows = True
    return has_rows
