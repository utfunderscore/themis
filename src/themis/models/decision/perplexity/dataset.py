"""Preparation, local sharding, and identity for decision inference."""

import json
from collections.abc import Iterable
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .batching import DEFAULT_BATCH_SIZE, DEFAULT_PADDED_TOKEN_BUDGET, MAX_INPUT_TOKENS
from .policy import MODEL_ID, MODEL_REVISION, NUMERICAL_POLICY, PIPELINE_VERSION
from .questions import validate_question

PROCESSING_WINDOW = 128


def prepare_dataset(source: str | Path, destination: str | Path, *, question: dict[str, Any],
                    text_field: str = "content", native_fields: Iterable[str] = ()) -> Path:
    """Write native decision rows while retaining each original record in ``source``.

    ``state`` and ``question`` are the exact fields supplied to native inference.
    ``native_fields`` can opt source fields (such as image inputs) into that native
    row. The ``source`` envelope keeps dataset metadata and prior scores available
    in the scored output without making them part of the shared inference API.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("source and destination must differ")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a nonempty string")
    native_fields = tuple(native_fields)
    reserved = {"state", "question", "source", "decision"}
    if (any(not isinstance(field, str) or not field for field in native_fields)
            or len(set(native_fields)) != len(native_fields)
            or set(native_fields) & reserved):
        raise ValueError("native_fields must be distinct non-reserved field names")
    validate_question(question)
    question = json.loads(json.dumps(question, ensure_ascii=False, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="prepare-decisions-", dir=destination.parent) as temporary:
        prepared = Path(temporary) / "prepared.jsonl"
        with source.open("rb") as reader, prepared.open("wb") as writer:
            for number, line in enumerate(reader, 1):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Invalid JSON on row {number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get(text_field), str):
                    raise ValueError(f"Row {number} must have string {text_field!r}")
                if "decision" in record:
                    raise ValueError(f"Row {number} already has a decision")
                missing = [field for field in native_fields if field not in record]
                if missing:
                    raise ValueError(f"Row {number} is missing native field {missing[0]!r}")
                row = {"state": record[text_field], "question": question}
                row.update({field: record[field] for field in native_fields})
                row["source"] = record
                try:
                    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid JSON on row {number}") from exc
                writer.write(payload)
        prepared.replace(destination)
    return destination


def split_dataset(source: str | Path, directory: str | Path, *,
                  rows_per_shard: int = 1024) -> tuple[list[Path], str, int]:
    """Return numbered shard paths, input SHA-256, and row count.

    Use a caller-owned TemporaryDirectory for scratch files and cleanup on failure.
    Original bytes and row order are preserved; memory is bounded by shard size.
    """
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    shards = []
    checksum = sha256()
    total = 0
    with Path(source).open("rb") as reader:
        while lines := list(islice(reader, rows_per_shard)):
            for number, line in enumerate(lines, total + 1):
                record = json.loads(line)
                if not isinstance(record, dict) or "state" not in record or "question" not in record:
                    raise ValueError(f"Row {number} must be a native decision row")
                validate_question(record["question"])
                if "decision" in record:
                    raise ValueError(f"Row {number} already has a decision")
                checksum.update(line)
            shard = Path(directory) / f"{len(shards):06d}.jsonl"
            with shard.open("xb") as writer:
                writer.writelines(lines)
            shards.append(shard)
            total += len(lines)
    return shards, checksum.hexdigest(), total


def dataset_manifest(checksum: str, total_rows: int, *, batch_size: int = DEFAULT_BATCH_SIZE,
                     rows_per_shard: int = 1024) -> bytes:
    """Hash these bytes for the job ID; operational settings do not belong here."""
    return json.dumps({
        "manifest_version": 3,
        "input_sha256": checksum,
        "total_rows": total_rows,
        # Each prepared row carries its native question. Its raw input hash
        # intentionally preserves choice-criteria insertion order.
        "input_format": "native-decision-rows-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_version": PIPELINE_VERSION,
        "numerical_policy": NUMERICAL_POLICY,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "output_field": "decision",
        "batch_size": batch_size,
        "max_padded_tokens": DEFAULT_PADDED_TOKEN_BUDGET,
        "processing_window": PROCESSING_WINDOW,
        "rows_per_shard": rows_per_shard,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
       allow_nan=False).encode("utf-8")
