"""KoalaAI text moderation on Modal, with weights stored in a Volume."""

import fcntl
import json
import math
import os
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from threading import Event, Thread
from time import monotonic
from typing import Protocol, cast

import modal

from themis._model_types import HubModule, Model, Tokenizer, TorchModule, TransformersModule


class Progress(Protocol):
    """Counters shared by Modal workers (or a plain dictionary in local tests)."""

    def __setitem__(self, key: str, value: int, /) -> None: ...
    def values(self) -> Iterable[int]: ...


MODEL_ID = "KoalaAI/Text-Moderation"
MODEL_REVISION = "838ade0edb66dcffc5532d08ff6ed5c899abfb5c"
PIPELINE_VERSION = 2  # FP16 weights, length bucketing, FP32 softmax.
MODEL_DIR = f"/models/{MODEL_REVISION}"

app = modal.App("themis-moderation")
volume = modal.Volume.from_name("themis-moderation-models", create_if_missing=True)
data_volume = modal.Volume.from_name("themis-moderation-data", create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface-hub==0.36.0"
)
inference_image = download_image.pip_install(
    "torch==2.8.0", "transformers==4.57.1"
)


@app.function(image=download_image, volumes={"/models": volume}, timeout=900)
def download_model() -> str:
    """Download the pinned weights on CPU. Run once before inference/deployment."""
    hub = cast(HubModule, cast(object, import_module("huggingface_hub")))

    _ = hub.snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    volume.commit()
    return MODEL_DIR


@app.cls(
    image=inference_image,
    gpu="T4",
    volumes={"/models": volume, "/data": data_volume},
    scaledown_window=300,
    timeout=3600,
    max_containers=4,
)
class Moderation:
    # Modal invokes load before methods; these are not constructor attributes.
    # Its enter decorator's generic return type is not fully annotated.
    @modal.enter()  # pyright: ignore[reportUnknownMemberType]
    def load(self) -> None:
        transformers = cast(TransformersModule, cast(object, import_module("transformers")))

        self.tokenizer: Tokenizer = transformers.AutoTokenizer.from_pretrained(  # pyright: ignore[reportUninitializedInstanceVariable]
            MODEL_DIR, local_files_only=True
        )
        self.model: Model = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUninitializedInstanceVariable]
            MODEL_DIR, local_files_only=True, use_safetensors=True
        ).half().to("cuda").eval()

    @modal.method()
    def predict(self, texts: list[str], batch_size: int = 128) -> list[dict[str, float]]:
        """Return nine label probabilities per text, preserving input order.

        Texts are truncated to 512 tokens (including special tokens).
        """
        torch = cast(TorchModule, cast(object, import_module("torch")))

        # Modal RPC arguments still need runtime validation despite annotations.
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("texts must be a list of strings")
        if type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be an integer between 1 and 128")

        results: list[dict[str, float]] = []
        with torch.inference_mode():
            for offset in range(0, len(texts), 10_000):
                window = texts[offset : offset + 10_000]
                encoded = self.tokenizer(window, truncation=True, max_length=512)
                indices = sorted(range(len(window)), key=lambda i: len(encoded["input_ids"][i]))
                ordered: list[dict[str, float]] = [{} for _ in window]
                for start in range(0, len(indices), batch_size):
                    batch = indices[start : start + batch_size]
                    inputs = self.tokenizer.pad(
                        [{key: value[i] for key, value in encoded.items()} for i in batch],
                        padding=True, return_tensors="pt",
                    ).to("cuda")
                    scores = self.model(**inputs).logits.float().softmax(dim=-1).cpu().tolist()
                    for position, row in zip(batch, scores, strict=True):
                        ordered[position] = {
                            self.model.config.id2label[index]: score
                            for index, score in enumerate(row)
                        }
                results.extend(ordered)
        return results

    @modal.method()
    def predict_file(
        self, input_path: str, text_field: str, batch_size: int, progress: Progress
    ) -> str:
        """Stream a Volume shard to a result file; RPC carries only paths."""
        return _predict_file(input_path, text_field, batch_size, progress, self.predict.local)


def _predict_file(
    input_path: str,
    text_field: str,
    batch_size: int,
    progress: Progress,
    predict: Callable[[list[str], int], list[dict[str, float]]],
    *,
    data_root: Path | None = None,
) -> str:
    """Worker body kept importable for tests without private Modal introspection."""
    data_volume.reload()
    if data_root is None:
        data_root = Path("/data")
    source = data_root / input_path.lstrip("/")
    destination = source.with_suffix(".scores.jsonl")
    completed = 0
    last_report = monotonic()
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        while lines := list(islice(reader, 10_000)):
            # Shards have already been validated by _split_dataset.
            records = [cast(dict[str, object], json.loads(line)) for line in lines]
            scores = predict([cast(str, record[text_field]) for record in records], batch_size)
            for record, probabilities in zip(records, scores, strict=True):
                record["moderation"] = probabilities
                _ = writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += len(records)
            if monotonic() - last_report >= 5:
                progress[input_path] = completed
                last_report = monotonic()
    data_volume.commit()
    # Scores must be closed and committed before publishing their marker.
    with source.with_suffix(".complete.json").open("w", encoding="utf-8") as marker:
        json.dump({"rows": completed}, marker)
    data_volume.commit()
    progress[input_path] = completed
    return destination.relative_to(data_root).as_posix()


def _split_dataset(
    source: Path, directory: Path, text_field: str, rows_per_shard: int
) -> tuple[list[Path], str, int]:
    """Validate and split JSONL, hashing original bytes in the same pass."""
    shards: list[Path] = []
    checksum = sha256()
    with source.open("rb") as reader:
        line_number = 0
        while first := reader.readline():
            shard = directory / f"{len(shards):06d}.jsonl"
            with shard.open("w", encoding="utf-8") as writer:
                for line in chain([first], islice(reader, rows_per_shard - 1)):
                    checksum.update(line)
                    line_number += 1
                    line = line.decode("utf-8")
                    try:
                        record = cast(object, json.loads(line))
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSON on row {line_number}") from exc
                    if not isinstance(record, dict) or not isinstance(
                        cast(dict[str, object], record).get(text_field), str
                    ):
                        raise ValueError(f"Row {line_number} must have a string {text_field!r} field")
                    if "moderation" in record:
                        raise ValueError(f"Row {line_number} already has a moderation field")
                    _ = writer.write(line.rstrip("\r\n") + "\n")
            shards.append(shard)
            if len(shards) % 50 == 0:
                print(f"Prepared {line_number:,} rows ({len(shards):,} shards)...", flush=True)
    return shards, checksum.hexdigest(), line_number


def _job_manifest(
    checksum: str, total_rows: int, text_field: str, batch_size: int, rows_per_shard: int
) -> bytes:
    """Canonical bytes define both the remote manifest and the job identity."""
    manifest = {
        "input_sha256": checksum,
        "total_rows": total_rows,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "text_field": text_field,
        "batch_size": batch_size,
        "rows_per_shard": rows_per_shard,
        "pipeline_version": PIPELINE_VERSION,
    }
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@contextmanager
def _job_lock(job_id: str) -> Generator[None]:
    """Serialize same-job callers on this machine, including publication/cleanup."""
    directory = Path(gettempdir()) / f"themis-moderation-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    # Keep the inode after unlock: deleting lock files can let waiters race.
    with (directory / f"{job_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_checkpoint(path: str) -> object:
    """Missing files mean unfinished work; other remote errors must propagate."""
    try:
        return cast(object, json.loads(b"".join(data_volume.read_file(path))))
    except FileNotFoundError:
        return None


@contextmanager
def _report_progress(
    progress: Progress, total: int, interval: float, reused_rows: int = 0
) -> Generator[None]:
    """Report wall time and interval throughput independently of shard completion."""
    started = previous_time = monotonic()
    previous_rows = reused_rows
    stopped = Event()

    def report() -> None:
        nonlocal previous_time, previous_rows
        completed = reused_rows + sum(progress.values())
        now = monotonic()
        rate = (completed - previous_rows) / max(now - previous_time, 1e-9)
        print(
            f"Inference elapsed: {now - started:,.1f}s | "
            + f"Rows: {completed:,}/{total:,} | {rate:,.1f} rows/s since last output",
            flush=True,
        )
        previous_time, previous_rows = now, completed

    def monitor() -> None:
        while not stopped.wait(interval):
            try:
                report()
            except Exception as exc:
                print(f"Progress update unavailable: {exc}", flush=True)

    print(f"Inference elapsed: 0.0s | Rows: {reused_rows:,}/{total:,} | 0.0 rows/s since last output", flush=True)
    thread = Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        try:
            report()
        except Exception as exc:
            print(f"Progress update unavailable: {exc}", flush=True)


def _moderate_shards(
    *,
    shards: list[Path],
    job_path: str,
    output_path: Path,
    text_field: str,
    batch_size: int,
    rows_per_shard: int,
    progress_interval: float,
    total_rows: int,
    manifest: bytes,
) -> None:
    """Resume committed shards, then combine all results in original order."""
    # Give each local shard a matching path in this job's remote Volume folder.
    remote_paths = [f"{job_path}/{shard.name}" for shard in shards]
    manifest_path = f"{job_path}/manifest.json"
    print(f"Checking job manifest: {job_path}", flush=True)
    existing = _read_checkpoint(manifest_path)
    if existing is not None and existing != json.loads(manifest):
        raise ValueError("Remote moderation manifest does not match this job")
    markers: dict[str, object] = {}
    if existing is not None:
        print("Listing existing checkpoints...", flush=True)
        # One listing avoids an RPC for every nonexistent marker. Normalize
        # leading slashes because Volume entries may use relative paths.
        present = {entry.path.lstrip("/") for entry in data_volume.listdir(job_path)}
        marker_paths = [
            str(Path(path).with_suffix(".complete.json")) for path in remote_paths
            if str(Path(path).with_suffix(".complete.json")).lstrip("/") in present
        ]
        if marker_paths:
            print(f"Reading {len(marker_paths):,} checkpoints with up to 8 concurrent requests...", flush=True)
            with ThreadPoolExecutor(max_workers=8) as executor:
                for count, (path, marker) in enumerate(zip(marker_paths, executor.map(_read_checkpoint, marker_paths), strict=True), 1):
                    markers[path] = marker
                    if count % 50 == 0 or count == len(marker_paths):
                        print(f"Checkpoints read: {count:,}/{len(marker_paths):,}", flush=True)
    else:
        print("New job: skipping shard checkpoint checks.", flush=True)
    unfinished: list[tuple[Path, str]] = []
    reused_rows = 0
    for index, (shard, remote_path) in enumerate(zip(shards, remote_paths, strict=True)):
        marker = markers.get(str(Path(remote_path).with_suffix(".complete.json")))
        expected_rows = min(rows_per_shard, total_rows - index * rows_per_shard)
        if marker is None:
            unfinished.append((shard, remote_path))
        elif existing is None or marker != {"rows": expected_rows}:
            raise ValueError(f"Invalid moderation checkpoint for {remote_path}")
        else:
            reused_rows += expected_rows

    if unfinished:
        print(f"Reusing {reused_rows:,} rows; uploading {len(unfinished):,} unfinished shards...", flush=True)
        # Overwrite abandoned inputs; score files without markers are rerun.
        with data_volume.batch_upload(force=True) as upload:
            # Modal's put_file return type is unannotated; its result is unused.
            upload.put_file(BytesIO(manifest), manifest_path)  # pyright: ignore[reportUnknownMemberType]
            for shard, remote_path in unfinished:
                upload.put_file(shard, remote_path)  # pyright: ignore[reportUnknownMemberType]

        # Only unfinished jobs start an app or initialize model/GPU workers.
        with app.run():
            print("Upload complete. Preparing cached model weights...", flush=True)
            _ = download_model.remote()
            print(f"Submitting {len(unfinished):,} shards for GPU inference...", flush=True)
            with modal.Dict.ephemeral() as progress, _report_progress(
                progress, total_rows, progress_interval, reused_rows
            ):
                _ = list(
                    Moderation().predict_file.map(
                        [remote_path for _, remote_path in unfinished],
                        kwargs={
                            "text_field": text_field,
                            "batch_size": batch_size,
                            "progress": progress,
                        },
                        order_outputs=True,
                    )
                )

    # Download-only resumes do not need Modal inference setup. Deterministic
    # paths keep reused and newly scored shards in the original dataset order.
    print(f"Downloading {len(remote_paths):,} result shards...", flush=True)
    downloaded_bytes = 0
    with output_path.open("wb") as writer:
        for count, remote_path in enumerate(remote_paths, 1):
            result_path = str(Path(remote_path).with_suffix(".scores.jsonl"))
            for chunk in data_volume.read_file(result_path):
                _ = writer.write(chunk)
                downloaded_bytes += len(chunk)
            if count % 10 == 0 or count == len(remote_paths):
                print(f"Downloaded {count:,}/{len(remote_paths):,} shards ({downloaded_bytes / 2**20:,.1f} MiB)", flush=True)


def moderate_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str = "content",
    batch_size: int = 128,
    rows_per_shard: int = 10_000,
    progress_interval: float = 30,
) -> Path:
    """Upload JSONL, run GPU inference, and download annotated JSONL in row order.

    Automatically resumes checkpoints for identical input bytes and scoring
    settings. Remote files survive failure and are deleted after publication.
    Same-job calls are serialized locally; simultaneous calls from different
    machines are unsupported. Local temporary disk holds a sharded input copy.
    """
    # Reject unsupported batch/shard sizes and invalid progress timing up front.
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be an integer between 1 and 128")
    if type(rows_per_shard) is not int or not 1 <= rows_per_shard <= 10_000:
        raise ValueError("rows_per_shard must be an integer between 1 and 10000")
    if not math.isfinite(progress_interval) or progress_interval <= 0:
        raise ValueError("progress_interval must be positive and finite")
    # Normalize paths and prevent the output from overwriting the input dataset.
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input_path and output_path must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep local working files beside the destination so the final replacement
    # stays on the same filesystem. They are cleaned up even if processing fails.
    with TemporaryDirectory(prefix="moderation-", dir=destination.parent) as temp:
        directory = Path(temp)
        # Validate every JSONL row and split the dataset into bounded-size files.
        print(f"Validating and splitting input: {source}", flush=True)
        shards, checksum, total_rows = _split_dataset(source, directory, text_field, rows_per_shard)
        print(f"Prepared {total_rows:,} rows in {len(shards):,} shards.", flush=True)
        manifest = _job_manifest(checksum, total_rows, text_field, batch_size, rows_per_shard)
        job_id = sha256(manifest).hexdigest()
        job_path = f"/jobs/{job_id}"
        # Write to a temporary result so failures leave existing output intact.
        pending = directory / "result.jsonl"
        print(f"Waiting for local job lock: {job_id}", flush=True)
        with _job_lock(job_id):
            if shards:
                # Upload, score, and download shards in their original order.
                _moderate_shards(
                    shards=shards,
                    job_path=job_path,
                    output_path=pending,
                    text_field=text_field,
                    batch_size=batch_size,
                    rows_per_shard=rows_per_shard,
                    progress_interval=progress_interval,
                    total_rows=total_rows,
                    manifest=manifest,
                )
            else:
                # An empty dataset produces an empty file without starting Modal.
                pending.touch()

            # Publish only complete results, then remove remote job files.
            # If processing failed above, checkpoints remain available for retry.
            _ = pending.replace(destination)
            print(f"Published results: {destination}", flush=True)
            if shards:
                print("Removing completed remote job files...", flush=True)
                data_volume.remove_file(job_path, recursive=True)
            print("Moderation complete.", flush=True)
    return destination
