"""Sharded inference with Volume results and per-shard Modal Dict checkpoints."""

# decisions.py imports _predict_file lazily inside the Modal worker.
# pyright: reportImportCycles=false

import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from uuid import uuid4

import modal

from .batching import DEFAULT_BATCH_SIZE, plan_batches
from .dataset import PROCESSING_WINDOW, dataset_manifest, split_dataset
from .decisions import Decisions, app, completed, data_volume, download_model
from .progress import RUN_STARTED_AT_ENV, log


def _predict_file(input_path, batch_size, predict, *, data_root=Path("/data")):
    """Commit a closed result before atomically publishing its completion entry."""
    if saved := completed.get(input_path):
        return saved
    data_volume.reload()
    source = data_root / input_path
    # Distinct attempts never overwrite an already published result.
    result = source.with_name(f"{source.stem}.{uuid4().hex}.scores.jsonl")
    count = 0
    checksum = sha256()
    with source.open(encoding="utf-8") as reader, result.open("wb") as writer:
        while lines := list(islice(reader, PROCESSING_WINDOW)):
            records = [json.loads(line) for line in lines]
            predictions = predict(records, batch_size)
            for row, prediction in zip(records, predictions, strict=True):
                row["decision"] = prediction
                payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
                writer.write(payload)
                checksum.update(payload)
                count += 1
    data_volume.commit()
    saved = {"result_path": result.relative_to(data_root).as_posix(),
             "rows": count, "sha256": checksum.hexdigest()}
    completed[input_path] = saved
    return saved


def predict_dataset(input_path: str | Path, output_path: str | Path, *,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    rows_per_shard: int = 1024, max_containers: int = 1) -> Path:
    """Score JSONL in source order, resuming completed shards on subsequent calls.

    Uses one coordinator per job. Checkpoints and remote files are retained after
    success; Dict entries expire after seven days without access. Changing the
    destination or worker limit does not change the job identity. Empty inputs
    and download-only resumes do not start inference.
    """
    started_at = time()
    plan_batches([], batch_size)
    if max_containers < 1:
        raise ValueError("max_containers must be positive")
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("input_path and output_path must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="decisions-", dir=destination.parent) as temp:
        shards, checksum, total = split_dataset(source, temp, rows_per_shard=rows_per_shard)
        manifest = dataset_manifest(checksum, total, batch_size=batch_size,
                                     rows_per_shard=rows_per_shard)
        job_path = f"jobs/{sha256(manifest).hexdigest()}"
        paths = [f"{job_path}/{shard.name}" for shard in shards]
        with ThreadPoolExecutor(max_workers=8) as pool:
            checkpoints = dict(zip(paths, pool.map(completed.get, paths), strict=True))
        pending = [path for path in paths if checkpoints[path] is None]
        log(f"Job {job_path}: {total:,} rows; {len(paths) - len(pending)}/{len(paths)} shards reused",
            started_at=started_at)
        if pending:
            with data_volume.batch_upload(force=True) as upload:
                for shard, path in zip(shards, paths, strict=True):
                    if checkpoints[path] is None:
                        upload.put_file(shard, path)
            with modal.enable_output(), app.run():
                download_model.remote()
                # Modal types app.cls as the bare class; at runtime it is a modal.Cls.
                worker = Decisions.with_options(  # pyright: ignore[reportAttributeAccessIssue]
                    max_containers=max_containers,
                    env={RUN_STARTED_AT_ENV: str(started_at)},
                )()
                for count, _ in enumerate(worker.predict_file.map(
                    pending, kwargs={"batch_size": batch_size}, order_outputs=False,
                ), 1):
                    log(f"Completed {count}/{len(pending)} submitted shards", started_at=started_at)
            # Read the durable entries, rather than treating RPC success as a checkpoint.
            with ThreadPoolExecutor(max_workers=8) as pool:
                checkpoints.update(zip(pending, pool.map(completed.get, pending), strict=True))
        result = Path(temp) / "result.jsonl"
        with result.open("wb") as writer:
            for index, path in enumerate(paths):
                saved = checkpoints[path]
                expected = min(rows_per_shard, total - index * rows_per_shard)
                if saved is None or saved["rows"] != expected:
                    raise ValueError(f"Incomplete shard: {path}")
                checksum = sha256()
                for chunk in data_volume.read_file(saved["result_path"]):
                    writer.write(chunk)
                    checksum.update(chunk)
                if checksum.hexdigest() != saved["sha256"]:
                    raise ValueError(f"Result checksum mismatch: {path}")
        result.replace(destination)
    log(f"Published {total:,} rows: {destination}", started_at=started_at)
    return destination
