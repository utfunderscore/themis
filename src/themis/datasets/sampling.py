"""Reproducible guided sampling of scored JSONL datasets, without remote work."""

import hashlib
import heapq
import json
import math
import os
import random
import struct
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

SAMPLING_VERSION = 1
# Source row, byte offset, byte length, category index, top-two margin, text length.
_INDEX = struct.Struct("!QQQIdQ")


@dataclass(frozen=True, slots=True)
class _Row:
    index: int
    offset: int
    byte_length: int
    category: int
    margin: float
    characters: int


def _index_rows(path: Path) -> Iterator[_Row]:
    with path.open("rb") as reader:
        while data := reader.read(_INDEX.size):
            if len(data) != _INDEX.size:
                raise RuntimeError("Incomplete temporary sampling index")
            yield _Row(*_INDEX.unpack(data))


def _keep_best(heap: list[tuple[float, int, _Row]], row: _Row, priority: float, count: int) -> None:
    if count == 0:
        return
    entry = (priority, -row.index, row)
    if len(heap) < count:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def _source_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as reader:
        while block := reader.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def _record_details(line: bytes, index: int, text_field: str, score_field: str,
                    metadata_field: str) -> tuple[dict[str, Any], str, float, int]:
    try:
        record = json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON at source row {index}") from exc
    if not isinstance(record, dict) or not isinstance(record.get(text_field), str):
        raise ValueError(f"Source row {index} requires a string {text_field!r}")
    if metadata_field in record:
        raise ValueError(f"Source row {index} already contains reserved field {metadata_field!r}")
    scores = record.get(score_field)
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"Source row {index} requires nonempty {score_field!r} probabilities")
    winner, best, second = "", -1.0, 0.0
    for label, value in scores.items():
        if (not isinstance(label, str) or not label or type(value) not in (float, int)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid probability in source row {index}")
        # Strict > preserves the score object's insertion order for argmax ties.
        if value > best:
            winner, second, best = label, max(best, 0.0), float(value)
        elif value > second:
            second = float(value)
    if best <= 0:
        raise ValueError(f"Source row {index} has no positive probability mass")
    return record, winner, best - second, len(record[text_field])


def guided_sample_dataset(
    source: str | Path, destination: str | Path, quotas: Mapping[str, int], *,
    seed: int = 42, ambiguous_fraction: float = 0.2, long_fraction: float = 0.1,
    text_field: str = "content", score_field: str = "moderation", metadata_field: str = "sampling",
) -> Path:
    """Select exact winning-label quotas with disjoint ambiguity/length/random tiers.

    Per category: first the smallest top-two probability gaps, then the longest
    remaining texts (Unicode character count), then a uniform seeded reservoir
    from the remainder. Fractional tier quotas are floored; random gets the rest.
    Ranking ties prefer earlier source rows. Sampling is without replacement by
    source row, not a second content-deduplication pass.

    Every original field/score is retained, with added namespaced source indices
    and selection metadata. Output follows original source order. Insufficient
    categories fail rather than silently redistribute quotas. A compact on-disk
    index avoids reparsing the full JSONL for every tier; memory scales with sample
    size, not source size. No libraries/models/remote services are required.

    Reuse requires matching configuration and SHA-256 checks of both source and
    output. Existing unrelated, stale or incomplete artifacts are never replaced.
    A .manifest.json sidecar records counts, hashes, seed and sampling policy.
    """
    source, destination = Path(source), Path(destination)
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    if source.resolve() in (destination.resolve(), manifest_path.resolve()):
        raise ValueError("source and output artifacts must differ")
    if (not quotas or any(not isinstance(label, str) or not label or type(count) is not int or count < 0
                          for label, count in quotas.items()) or sum(quotas.values()) == 0):
        raise ValueError("quotas must contain nonnegative integer counts and a positive total")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for name, value in (("ambiguous_fraction", ambiguous_fraction), ("long_fraction", long_fraction)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite fraction between 0 and 1")
    if ambiguous_fraction + long_fraction > 1:
        raise ValueError("ambiguous_fraction + long_fraction must not exceed 1")
    if any(not isinstance(field, str) or not field for field in (text_field, score_field, metadata_field)):
        raise ValueError("Field names must be nonempty strings")
    if len({text_field, score_field, metadata_field}) != 3:
        raise ValueError("Text, score and metadata fields must differ")
    labels = sorted(quotas)
    requested = {label: quotas[label] for label in labels}
    config = {
        "version": SAMPLING_VERSION, "source_path": str(source.resolve()), "quotas": requested,
        "seed": seed, "ambiguous_fraction": ambiguous_fraction, "long_fraction": long_fraction,
        "text_field": text_field, "score_field": score_field, "metadata_field": metadata_field,
        "selection_order": ["ambiguous", "long", "random"],
        "argmax_ties": "score-object insertion order", "ranking_ties": "earlier source row",
        "rounding": "floor ambiguity/length quotas; random takes remainder", "source_indices": "zero-based",
    }
    identity = _source_identity(source)
    if destination.exists() or manifest_path.exists():
        if not destination.is_file() or not manifest_path.is_file():
            raise FileExistsError("Sample/manifest incomplete; refusing to overwrite existing artifacts")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("config") != config:
            raise FileExistsError("Existing sample uses a different sampling configuration")
        if saved.get("source_sha256") != _sha256(source) or identity != _source_identity(source):
            raise ValueError("Existing sample source changed; choose a new destination")
        if saved.get("output_sha256") != _sha256(destination):
            raise ValueError("Existing sample checksum mismatch; refusing to reuse it")
        print(f"Reusing verified guided sample: {destination}", flush=True)
        return destination

    targets = {
        label: {"ambiguous": math.floor(count * ambiguous_fraction), "long": math.floor(count * long_fraction)}
        for label, count in requested.items()
    }
    for label, count in requested.items():
        targets[label]["random"] = count - sum(targets[label].values())
    margin_heaps: dict[str, list[tuple[float, int, _Row]]] = {label: [] for label in labels}
    length_heaps: dict[str, list[tuple[float, int, _Row]]] = {label: [] for label in labels}
    label_ids = {label: index for index, label in enumerate(labels)}
    available: Counter[str] = Counter()
    checksum = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = last_report = monotonic()
    print(f"Guided sampling {source}: requested {sum(requested.values()):,} rows, seed={seed}", flush=True)
    with TemporaryDirectory(prefix="guided-sample-", dir=destination.parent) as directory:
        temporary = Path(directory)
        index_path = temporary / "rows.index"
        total, offset = 0, 0
        with source.open("rb") as reader, index_path.open("wb") as index_writer:
            for index, line in enumerate(reader):
                checksum.update(line)
                _, label, margin, characters = _record_details(line, index, text_field, score_field, metadata_field)
                available[label] += 1
                if requested.get(label, 0):
                    row = _Row(index, offset, len(line), label_ids[label], margin, characters)
                    index_writer.write(_INDEX.pack(index, offset, len(line), row.category, margin, characters))
                    _keep_best(margin_heaps[label], row, -margin, targets[label]["ambiguous"])
                total += 1
                offset += len(line)
                if monotonic() - last_report >= 10:
                    print(f"Guided sampling: validated {total:,} rows; elapsed={monotonic() - started:.1f}s", flush=True)
                    last_report = monotonic()
        if identity != _source_identity(source):
            raise ValueError("Source changed during guided sampling")
        for label, count in requested.items():
            if available[label] < count:
                raise ValueError(f"Category {label!r} has {available[label]} rows, but quota requests {count}")
        selected: dict[int, tuple[_Row, str]] = {
            row.index: (row, "ambiguous") for heap in margin_heaps.values() for _, _, row in heap
        }
        print("Guided sampling: selecting longest remaining chats", flush=True)
        for row in _index_rows(index_path):
            if row.index not in selected:
                label = labels[row.category]
                _keep_best(length_heaps[label], row, row.characters, targets[label]["long"])
        selected.update({row.index: (row, "long") for heap in length_heaps.values() for _, _, row in heap})
        reservoirs: dict[str, list[_Row]] = {label: [] for label in labels}
        seen: Counter[str] = Counter()
        generators = {label: random.Random(f"{seed}:{label}") for label in labels}
        print("Guided sampling: reservoir-sampling remaining chats", flush=True)
        for row in _index_rows(index_path):
            if row.index in selected:
                continue
            label = labels[row.category]
            count = targets[label]["random"]
            if count == 0:
                continue
            seen[label] += 1
            reservoir = reservoirs[label]
            if len(reservoir) < count:
                reservoir.append(row)
            else:
                slot = generators[label].randrange(seen[label])
                if slot < count:
                    reservoir[slot] = row
        selected.update({row.index: (row, "random") for reservoir in reservoirs.values() for row in reservoir})
        if len(selected) != sum(requested.values()):
            raise RuntimeError("Guided sampling did not fill every quota")
        output_path = temporary / "sample.jsonl"
        output_checksum = hashlib.sha256()
        actual = {label: Counter() for label in labels}
        with source.open("rb") as reader, output_path.open("wb") as writer:
            for index in sorted(selected):
                row, method = selected[index]
                reader.seek(row.offset)
                original = reader.read(row.byte_length)
                record, label, margin, characters = _record_details(original, index, text_field, score_field, metadata_field)
                if (label != labels[row.category] or margin != row.margin or characters != row.characters):
                    raise ValueError("Source changed during guided sampling")
                record[metadata_field] = {
                    "source_row_index": index, "category": label, "method": method,
                    "top_two_margin": margin, "content_characters": characters,
                }
                payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                writer.write(payload)
                output_checksum.update(payload)
                actual[label][method] += 1
        if identity != _source_identity(source):
            raise ValueError("Source changed during guided sampling")
        if any(dict(actual[label]) != {method: count for method, count in targets[label].items() if count}
               for label in labels):
            raise RuntimeError("Guided sampling tier counts do not match requested quotas")
        manifest = {
            "config": config, "source_sha256": checksum.hexdigest(), "source_bytes": offset,
            "source_rows": total, "available_categories": dict(available), "sample_rows": len(selected),
            "selected_tiers": {label: dict(actual[label]) for label in labels},
            "output_sha256": output_checksum.hexdigest(),
        }
        temporary_manifest = temporary / "manifest.json"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
        # Atomic no-clobber publication on the destination filesystem. If the
        # second link fails, remove only the first link created by this call.
        os.link(output_path, destination)
        try:
            os.link(temporary_manifest, manifest_path)
        except BaseException:
            if destination.exists() and destination.samefile(output_path):
                destination.unlink()
            raise
    print(f"Saved {len(selected):,} guided rows to {destination}; elapsed={monotonic() - started:.1f}s", flush=True)
    return destination
