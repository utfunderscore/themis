"""Download the Minecraft Server Chat dataset as JSONL."""

from __future__ import annotations

import json
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import polars as pl

DATASET_URL = (
    "https://huggingface.co/datasets/declip/Minecraft-Server-Chat/resolve/main/"
    "clean.json?download=true"
)


def load(path: Path) -> pl.LazyFrame:
    """Lazily load cached chats using the common three-column schema."""
    return pl.scan_ndjson(
        path, schema={"username": pl.String, "content": pl.String, "date": pl.String}
    )


def download(output_path: Path, force: bool = False) -> pl.LazyFrame:
    """Cache JSONL unless already present, then return lazy chats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if force or not output_path.exists():
        with cast(HTTPResponse, urlopen(DATASET_URL)) as response:
            records = cast(object, json.load(response))
        if not isinstance(records, list):
            raise ValueError("Expected the dataset to contain a JSON array")
        with output_path.open("w", encoding="utf-8") as output:
            for record in cast(list[object], records):
                _ = output.write(json.dumps(record, ensure_ascii=False) + "\n")

    return load(output_path)
