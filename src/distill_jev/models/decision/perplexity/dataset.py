"""Local contract and preparation for resumable Perplexity datasets.

The future paid orchestrator will expose::

    predict_dataset(input_path, output_path, *, question, text_field='content',
                    batch_size=32, rows_per_shard=1024, max_containers=1,
                    progress_interval=30) -> Path

It is deliberately not implemented here. Preparation never scores or publishes
an output. Each input object survives unchanged; only the future scorer may add
``decision`` (native probabilities, native answer, and pinned provenance).
Decision state is the selected string field; other record fields, including
images, moderation and sampling metadata, are not inference arguments.
Whole rendered inputs must later be checked against the native token limit;
this CPU-only phase validates JSON/structure, not tokenizer lengths.
"""

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from .batching import (
    DEFAULT_BATCH_SIZE, DEFAULT_PADDED_TOKEN_BUDGET, MAX_INPUT_TOKENS, plan_batches,
)
from .policy import MODEL_ID, MODEL_REVISION, NUMERICAL_POLICY, PIPELINE_VERSION
from .questions import validate_question

MANIFEST_VERSION = 1
OUTPUT_FIELD = "decision"
DEFAULT_PROCESSING_WINDOW = 128


@dataclass(frozen=True)
class DatasetSettings:
    """Semantic and operational settings; construction validates both locally."""

    text_field: str = "content"
    batch_size: int = DEFAULT_BATCH_SIZE
    rows_per_shard: int = 1024
    max_containers: int = 1
    progress_interval: float = 30
    max_padded_tokens: int = DEFAULT_PADDED_TOKEN_BUDGET
    processing_window: int = DEFAULT_PROCESSING_WINDOW

    def __post_init__(self) -> None:
        if not isinstance(self.text_field, str) or not self.text_field or self.text_field == OUTPUT_FIELD:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("text_field must be a nonempty string other than 'decision'")
        _ = plan_batches([], self.batch_size, max_padded_tokens=self.max_padded_tokens)
        for name, value in (("rows_per_shard", self.rows_per_shard),
                            ("max_containers", self.max_containers),
                            ("processing_window", self.processing_window)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.processing_window > 128:
            raise ValueError("processing_window must not exceed the 128-row RPC limit")
        if (type(self.progress_interval) not in (int, float)
                or not math.isfinite(self.progress_interval) or self.progress_interval <= 0):
            raise ValueError("progress_interval must be positive and finite")


def validate_dataset_paths(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    """Protect source bytes, including symlink and existing hardlink aliases.

    Does not create directories or touch the eventual output.
    """
    source, destination = Path(input_path), Path(output_path)
    if (source.resolve() == destination.resolve()
            or (destination.exists() and source.samefile(destination))):
        raise ValueError("input_path and output_path must differ")
    return source, destination


def decision_input(record: dict[str, object], question: dict[str, object], *,
                   text_field: str = "content") -> dict[str, object]:
    """Map a validated original record to text-only native decision state."""
    return {"state": record[text_field], "question": question}


@dataclass(frozen=True)
class DatasetShard:
    path: Path
    start_row: int  # Zero-based source position; JSONL errors use one-based rows.
    row_count: int


@dataclass(frozen=True)
class PreparedDataset:
    shards: tuple[DatasetShard, ...]
    input_sha256: str
    total_rows: int
    manifest: bytes

    @property
    def job_id(self) -> str:
        return sha256(self.manifest).hexdigest()


def dataset_manifest(checksum: str, total_rows: int, *, question: dict[str, object],
                     settings: DatasetSettings) -> bytes:
    """Versioned canonical semantic identity, independent of run operations.

    Choice criteria become ordered pairs, so sort_keys cannot erase candidate
    order. Decode this representation back to an insertion-ordered dict before
    native rendering. Score criteria already are ordered; noul uses the pinned
    native boolean probability convention. Extra question fields are retained.
    """
    validate_question(question)
    if (not isinstance(checksum, str) or len(checksum) != 64  # pyright: ignore[reportUnnecessaryIsInstance]
            or any(char not in "0123456789abcdef" for char in checksum)):
        raise ValueError("checksum must be a lowercase SHA256 hex digest")
    if type(total_rows) is not int or total_rows < 0:
        raise ValueError("total_rows must be a nonnegative integer")
    encoded_question = dict(question)
    if question["type"] == "choice":
        encoded_question["criteria"] = list(cast(dict[str, object], question["criteria"]).items())
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "input_sha256": checksum,
        "total_rows": total_rows,
        "question": encoded_question,
        "question_encoding": "choice-criteria-ordered-pairs-v1",
        "probability_interpretation": "pinned-native-question-candidate-order",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_version": PIPELINE_VERSION,
        "numerical_policy": NUMERICAL_POLICY,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "text_field": settings.text_field,
        "output_field": OUTPUT_FIELD,
        "batch_size": settings.batch_size,
        "max_padded_tokens": settings.max_padded_tokens,
        "processing_window": settings.processing_window,
        "rows_per_shard": settings.rows_per_shard,
    }
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate JSON field {key!r}")
        record[key] = value
    return record


def prepare_dataset(input_path: str | Path, directory: str | Path, *,
                    question: dict[str, object], settings: DatasetSettings | None = None,
                    output_path: str | Path | None = None) -> PreparedDataset:
    """Stream-validate the complete input and write raw-byte local shards.

    ``directory`` must be an existing empty caller-owned scratch directory.
    On validation failure only files created here are removed. One row is held
    at a time (plus O(shard count) descriptors); no dataset-sized row list.
    Blank lines, duplicate JSON keys, nonfinite numbers, nonobjects, nonstring
    text, and preexisting decision fields are errors, never skipped. Original
    bytes (including whitespace, CRLF and a missing final newline) are hashed
    and preserved. Shard positions provide source row identity without adding
    internal fields to user records. Empty input has no shards and a manifest.
    """
    settings = settings if settings is not None else DatasetSettings()
    validate_question(question)
    # Snapshot JSON values before reading a potentially large input.
    question = cast(dict[str, object], json.loads(json.dumps(question, allow_nan=False)))
    source = Path(input_path)
    if output_path is not None:
        _ = validate_dataset_paths(source, output_path)
    directory = Path(directory)
    if not directory.is_dir() or any(directory.iterdir()):
        raise ValueError("directory must be an existing empty scratch directory")
    # A future publication must not target any preparation-owned path.
    if output_path is not None and Path(output_path).resolve().is_relative_to(directory.resolve()):
        raise ValueError("output_path must be outside the scratch directory")
    checksum = sha256()
    total_rows = 0
    shards: list[DatasetShard] = []
    created: list[Path] = []
    writer = None
    try:
        with source.open("rb") as reader:
            for raw in reader:
                row_number = total_rows + 1
                try:
                    record = cast(object, json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                                                     object_pairs_hook=_unique_object))
                    # Also catch finite JSON literals that overflow to inf.
                    _ = json.dumps(record, allow_nan=False)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid JSON on row {row_number}: {exc}") from exc
                if not isinstance(record, dict) or not isinstance(
                    cast(dict[str, object], record).get(settings.text_field), str
                ):
                    raise ValueError(f"Row {row_number} must have a string {settings.text_field!r} field")
                if OUTPUT_FIELD in record:
                    raise ValueError(f"Row {row_number} already has a {OUTPUT_FIELD} field")
                if total_rows % settings.rows_per_shard == 0:
                    if writer is not None:
                        writer.close()
                    path = directory / f"{len(shards):06d}.jsonl"
                    writer = path.open("xb")
                    created.append(path)
                    shards.append(DatasetShard(path, total_rows, 0))
                assert writer is not None  # The first validated row opens a shard.
                _ = writer.write(raw)
                checksum.update(raw)
                total_rows += 1
                last = shards[-1]
                shards[-1] = DatasetShard(last.path, last.start_row, last.row_count + 1)
        manifest = dataset_manifest(checksum.hexdigest(), total_rows, question=question, settings=settings)
        return PreparedDataset(tuple(shards), checksum.hexdigest(), total_rows, manifest)
    except BaseException:
        if writer is not None:
            writer.close()
        for path in created:
            path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
