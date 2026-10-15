"""Download the 2025 2b2t archive and cache only its chats.csv file."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import polars as pl

DATASET_URL = "https://cdn.2b2t.vc/vcdb-2025.zip"


def load(path: Path) -> pl.LazyFrame:
    """Lazily load cached CSV chats using the common three-column schema."""
    return pl.scan_csv(path, infer_schema=False, empty_string_is_null=False).select(  # pyright: ignore[reportUnknownMemberType]
        pl.col("player_name").alias("username"),
        pl.col("chat").alias("content"),
        pl.col("time").alias("date"),
    )


def download(output_path: Path = Path("/cache/chats.csv"), force: bool = False) -> pl.LazyFrame:
    """Return lazy chats, downloading the cached CSV if missing or forced.

    The full ZIP is streamed to disk, but only chats.csv is extracted. Temporary
    files are removed even on failure, and an existing CSV is replaced only after
    successful extraction. The destination directory must be writable.
    """
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        return load(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="2b2t-", dir=output_path.parent) as directory:
        archive_path = Path(directory) / "dataset.zip"
        request = Request(DATASET_URL, headers={"User-Agent": "themis/0.1 (dataset downloader)"})
        with urlopen(request, timeout=60) as response, archive_path.open("wb") as archive:
            shutil.copyfileobj(response, archive, length=1024 * 1024)

        with ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).name == "chats.csv"
            ]
            if len(members) != 1:
                raise ValueError(
                    f"Expected exactly one chats.csv in the archive; found {len(members)}"
                )
            csv_path = Path(directory) / "chats.csv"
            with archive.open(members[0]) as source, csv_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)

        csv_path.replace(output_path)

    return load(output_path)
