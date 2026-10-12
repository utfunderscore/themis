"""Local sharding and job identity for resumable decision inference."""

import json
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any

from .batching import DEFAULT_BATCH_SIZE, DEFAULT_PADDED_TOKEN_BUDGET, MAX_INPUT_TOKENS
from .policy import MODEL_ID, MODEL_REVISION, NUMERICAL_POLICY, PIPELINE_VERSION

PROCESSING_WINDOW = 128


def split_dataset(source: str | Path, directory: str | Path, *,
                  text_field: str = "content", rows_per_shard: int = 1024,
                  ) -> tuple[list[Path], str, int]:
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
                if not isinstance(record[text_field], str):
                    raise ValueError(f"Row {number} must have string {text_field!r}")
                if "decision" in record:
                    raise ValueError(f"Row {number} already has a decision")
                checksum.update(line)
            shard = Path(directory) / f"{len(shards):06d}.jsonl"
            with shard.open("xb") as writer:
                writer.writelines(lines)
            shards.append(shard)
            total += len(lines)
    return shards, checksum.hexdigest(), total


def dataset_manifest(checksum: str, total_rows: int, *, question: dict[str, Any],
                     text_field: str = "content", batch_size: int = DEFAULT_BATCH_SIZE,
                     rows_per_shard: int = 1024) -> bytes:
    """Hash these bytes for the job ID; operational settings do not belong here."""
    question = dict(question)
    if question["type"] == "choice":
        # Candidate order determines the meaning of the probability vector.
        question["criteria"] = list(question["criteria"].items())
    return json.dumps({
        "manifest_version": 2,
        "input_sha256": checksum,
        "total_rows": total_rows,
        "question": question,
        "question_encoding": "choice-criteria-ordered-pairs-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_version": PIPELINE_VERSION,
        "numerical_policy": NUMERICAL_POLICY,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "text_field": text_field,
        "output_field": "decision",
        "batch_size": batch_size,
        "max_padded_tokens": DEFAULT_PADDED_TOKEN_BUDGET,
        "processing_window": PROCESSING_WINDOW,
        "rows_per_shard": rows_per_shard,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
       allow_nan=False).encode("utf-8")
