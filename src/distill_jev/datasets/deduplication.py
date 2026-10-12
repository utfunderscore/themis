"""Reusable chat dataset deduplication."""

from pathlib import Path

import polars as pl


def deduplicate_chats(source: Path | pl.LazyFrame, destination: Path) -> Path:
    """Keep the first record for each exact content string, without ordering rows."""
    if isinstance(source, Path) and source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    chats = pl.scan_ndjson(source) if isinstance(source, Path) else source
    deduplicated = chats.unique(  # pyright: ignore[reportUnknownMemberType]
        subset=["content"], keep="first"
    )
    _ = deduplicated.sink_ndjson(destination)
    total = pl.scan_ndjson(destination).select(pl.len()).collect().item()
    print(f"Deduplicated dataset contains {total:,} rows", flush=True)
    return destination
