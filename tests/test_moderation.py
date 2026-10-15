import json
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import Protocol, cast, override
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import modal

from themis import moderation
from themis.moderation import _job_manifest, _report_progress, _split_dataset, moderate_dataset

# White-box tests intentionally exercise internal checkpoint/progress helpers.
# pyright: reportPrivateUsage=false


class MockCall(Protocol):
    @property
    def args(self) -> tuple[object, ...]: ...


class MockHandle(Protocol):
    """Narrow view of unittest's deliberately Any-based dynamic mock objects."""
    side_effect: object
    return_value: "MockHandle"
    call_args: MockCall | None

    def __getattr__(self, name: str) -> "MockHandle": ...
    def reset_mock(self) -> None: ...
    def assert_not_called(self) -> None: ...


class MemoryVolume:
    """Small Volume fake; uploads and checkpoints persist across calls."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.removals: list[str] = []
        self.fail_download: bool = False
        self.reads: list[str] = []
        self.listings: list[str] = []

    def listdir(self, path: str) -> list[SimpleNamespace]:
        self.listings.append(path)
        return [SimpleNamespace(path=key.lstrip("/")) for key in self.files if key.startswith(path + "/")]

    def read_file(self, path: str) -> Iterator[bytes]:
        self.reads.append(path)
        if self.fail_download and path.endswith(".scores.jsonl"):
            raise RuntimeError("download failed")
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    @contextmanager
    def batch_upload(self, force: bool = False) -> Generator["MemoryVolume"]:
        assert force
        yield self

    def put_file(self, source: Path | BytesIO, path: str) -> None:
        self.uploads.append(path)
        self.files[path] = source.read() if isinstance(source, BytesIO) else source.read_bytes()

    def remove_file(self, path: str, recursive: bool = False) -> None:
        assert recursive
        self.removals.append(path)
        self.files = {key: value for key, value in self.files.items() if not key.startswith(path + "/")}


class ResumeTests(unittest.TestCase):
    # unittest calls setUp before each test rather than constructor initialization.
    stack: ExitStack  # pyright: ignore[reportUninitializedInstanceVariable]
    directory: Path  # pyright: ignore[reportUninitializedInstanceVariable]
    source: Path  # pyright: ignore[reportUninitializedInstanceVariable]
    output: Path  # pyright: ignore[reportUninitializedInstanceVariable]
    volume: MemoryVolume  # pyright: ignore[reportUninitializedInstanceVariable]
    app: MockHandle  # pyright: ignore[reportUninitializedInstanceVariable]
    download: MockHandle  # pyright: ignore[reportUninitializedInstanceVariable]
    workers: MockHandle  # pyright: ignore[reportUninitializedInstanceVariable]
    report: MockHandle  # pyright: ignore[reportUninitializedInstanceVariable]
    processed: list[str]  # pyright: ignore[reportUninitializedInstanceVariable]
    fail_shard: str | None  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(TemporaryDirectory()))
        self.source = self.directory / "input.jsonl"
        self.output = self.directory / "output.jsonl"
        _ = self.source.write_text("".join(json.dumps({"id": i, "content": str(i)}) + "\n" for i in range(5)))
        _ = self.output.write_text("existing output")
        self.volume = MemoryVolume()
        _ = self.stack.enter_context(patch.object(moderation, "data_volume", self.volume))
        self.app = cast(MockHandle, self.stack.enter_context(patch.object(moderation, "app")))
        self.download = cast(MockHandle, self.stack.enter_context(patch.object(moderation, "download_model")))
        self.workers = cast(MockHandle, self.stack.enter_context(patch.object(moderation, "Moderation")))
        self.report = cast(MockHandle, self.stack.enter_context(patch.object(moderation, "_report_progress")))
        _ = self.stack.enter_context(patch.object(modal.Dict, "ephemeral", return_value=MagicMock()))
        self.processed = []
        self.fail_shard = None
        self.workers.return_value.predict_file.map.side_effect = self.score

    def score(
        self, paths: list[str], *, kwargs: Mapping[str, object], order_outputs: bool
    ) -> Iterator[str]:
        self.assertTrue(order_outputs)
        self.assertEqual(kwargs["batch_size"], 128)
        for path in paths:
            self.processed.append(Path(path).name)
            result_path = str(Path(path).with_suffix(".scores.jsonl"))
            if Path(path).name == self.fail_shard:
                # An interrupted worker may have a score file but no marker.
                self.volume.files[result_path] = b"partial result\n"
                raise RuntimeError("inference failed")
            rows = [cast(dict[str, object], json.loads(line)) for line in self.volume.files[path].splitlines()]
            for row in rows:
                row["moderation"] = {"OK": 1.0}
            self.volume.files[result_path] = b"".join(
                (json.dumps(row) + "\n").encode() for row in rows
            )
            self.volume.files[str(Path(path).with_suffix(".complete.json"))] = json.dumps({"rows": len(rows)}).encode()
            yield result_path

    def run_job(self, *, progress_interval: float = 30) -> Path:
        return moderate_dataset(self.source, self.output, rows_per_shard=2, progress_interval=progress_interval)

    def assert_ordered_output(self) -> None:
        rows = [cast(dict[str, object], json.loads(line)) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], list(range(5)))
        self.assertTrue(all(row["moderation"] == {"OK": 1.0} for row in rows))

    def test_failed_retry_skips_completed_and_overwrites_unmarked_scores(self) -> None:
        self.fail_shard = "000001.jsonl"
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            _ = self.run_job()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertFalse(self.volume.removals)
        self.assertTrue(any(path.endswith("000000.complete.json") for path in self.volume.files))
        self.volume.uploads.clear()
        self.processed.clear()
        self.fail_shard = None
        _ = self.run_job(progress_interval=1)
        self.assertEqual(self.processed, ["000001.jsonl", "000002.jsonl"])
        self.assertFalse(any(path.endswith("000000.jsonl") for path in self.volume.uploads))
        assert self.report.call_args is not None
        self.assertEqual(self.report.call_args.args[1:], (5, 1, 2))
        self.assert_ordered_output()
        self.assertFalse(self.volume.files)
        self.assertEqual(len(self.volume.removals), 1)
        # Successful jobs remove checkpoints; the next call starts fresh.
        self.processed.clear()
        _ = self.run_job()
        self.assertEqual(self.processed, ["000000.jsonl", "000001.jsonl", "000002.jsonl"])

    def test_fresh_job_skips_checkpoint_listing_and_reads(self) -> None:
        _ = self.run_job()
        self.assertEqual(self.volume.listings, [])
        self.assertFalse(any(path.endswith(".complete.json") for path in self.volume.reads))

    def test_resume_reads_only_listed_markers(self) -> None:
        self.fail_shard = "000001.jsonl"
        with self.assertRaises(RuntimeError):
            self.run_job()
        self.volume.reads.clear()
        self.fail_shard = None
        self.run_job()
        self.assertEqual(len(self.volume.listings), 1)
        marker_reads = [path for path in self.volume.reads if path.endswith(".complete.json")]
        self.assertEqual(len(marker_reads), 1)
        self.assertTrue(marker_reads[0].endswith("000000.complete.json"))

    def test_download_only_resume_ignores_destination_and_progress_interval(self) -> None:
        self.volume.fail_download = True
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            _ = self.run_job()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertFalse(self.volume.removals)
        self.assertEqual(sum(path.endswith(".complete.json") for path in self.volume.files), 3)
        self.volume.fail_download = False
        self.app.reset_mock()
        self.download.reset_mock()
        self.workers.reset_mock()
        self.volume.uploads.clear()
        self.output = self.directory / "other-output.jsonl"
        _ = self.run_job(progress_interval=1)
        self.app.run.assert_not_called()
        self.download.remote.assert_not_called()
        self.workers.assert_not_called()
        self.assertFalse(self.volume.uploads)
        self.assert_ordered_output()

    def test_middle_checkpoint_is_merged_in_order(self) -> None:
        self.volume.fail_download = True
        with self.assertRaises(RuntimeError):
            _ = self.run_job()
        for path in list(self.volume.files):
            if path.endswith(("000000.complete.json", "000002.complete.json")):
                del self.volume.files[path]
        self.processed.clear()
        self.volume.fail_download = False
        _ = self.run_job()
        self.assertEqual(self.processed, ["000000.jsonl", "000002.jsonl"])
        self.assert_ordered_output()

    def test_publication_failure_retains_checkpoints(self) -> None:
        with patch.object(Path, "replace", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                _ = self.run_job()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertFalse(self.volume.removals)
        self.assertEqual(sum(path.endswith(".complete.json") for path in self.volume.files), 3)

    def test_remote_errors_are_not_treated_as_missing_checkpoints(self) -> None:
        with patch.object(self.volume, "read_file", side_effect=RuntimeError("network failed")):
            with self.assertRaisesRegex(RuntimeError, "network failed"):
                _ = self.run_job()
        self.assertEqual(self.output.read_text(), "existing output")
        self.download.remote.assert_not_called()

    def test_invalid_manifest_or_marker_preserves_checkpoints(self) -> None:
        self.volume.fail_download = True
        with self.assertRaises(RuntimeError):
            _ = self.run_job()
        manifest_path = next(path for path in self.volume.files if path.endswith("manifest.json"))
        original = self.volume.files[manifest_path]
        self.volume.files[manifest_path] = b'{"wrong": "manifest"}'
        with self.assertRaisesRegex(ValueError, "manifest"):
            _ = self.run_job()
        self.volume.files[manifest_path] = original
        marker_path = next(path for path in self.volume.files if path.endswith(".complete.json"))
        self.volume.files[marker_path] = b'{"rows": 999}'
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            _ = self.run_job()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertFalse(self.volume.removals)


class DatasetTests(unittest.TestCase):
    def test_progress_aggregates_workers_and_uses_interval_rate(self) -> None:
        progress = {"shard-a": 8, "shard-b": 4}
        with (
            patch("themis.moderation.monotonic", side_effect=[10, 14, 16]),
            patch("themis.moderation.Thread") as thread,
            patch("builtins.print") as output,
        ):
            with _report_progress(progress, 20, 30):
                # Run one scheduled report, then stop the monitor loop.
                with patch("themis.moderation.Event.wait", side_effect=[False, True]):
                    assert thread.call_args is not None
                    cast(Callable[[], None], thread.call_args.kwargs["target"])()
                progress["shard-a"] = 12
            self.assertIn("Rows: 12/20 | 3.0 rows/s", cast(str, output.call_args_list[1].args[0]))
            self.assertIn("elapsed: 6.0s | Rows: 16/20 | 2.0 rows/s", cast(str, output.call_args_list[2].args[0]))

    def test_shards_preserve_rows_and_final_line_without_newline(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "input.jsonl"
            records = [{"id": i, "content": f"こんにちは {i}"} for i in range(5)]
            _ = source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            shards, checksum, total = _split_dataset(source, directory, "content", 2)
            self.assertEqual(total, 5)
            self.assertEqual(checksum, sha256(source.read_bytes()).hexdigest())
            contents = [[cast(dict[str, object], json.loads(line)) for line in shard.read_text().splitlines()] for shard in shards]
            self.assertEqual([len(rows) for rows in contents], [2, 2, 1])
            self.assertEqual([row for rows in contents for row in rows], records)

    def test_invalid_input_preserves_existing_output(self) -> None:
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input.jsonl"
            output = Path(temp) / "output.jsonl"
            _ = output.write_text("existing output")
            for invalid in ['invalid', '{"content": null}', '{"content": "x", "moderation": {}}']:
                _ = source.write_text('{"content": "valid"}\n' + invalid)
                with self.assertRaisesRegex(ValueError, "[Rr]ow 2"):
                    _ = moderate_dataset(source, output)
                self.assertEqual(output.read_text(), "existing output")

    def test_empty_dataset_downloads_empty_output(self) -> None:
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input.jsonl"
            output = Path(temp) / "output.jsonl"
            source.touch()
            self.assertEqual(moderate_dataset(source, output), output)
            self.assertEqual(output.read_bytes(), b"")

    def test_manifest_identity_is_stable_and_covers_all_scoring_settings(self) -> None:
        arguments: tuple[str, int, str, int, int] = ("checksum", 5, "content", 16, 2)
        baseline = _job_manifest(*arguments)
        self.assertEqual(baseline, _job_manifest(*arguments))
        alternatives: list[tuple[str, int, str, int, int]] = [
            ("new checksum", 5, "content", 16, 2),
            ("checksum", 6, "content", 16, 2),
            ("checksum", 5, "text", 16, 2),
            ("checksum", 5, "content", 32, 2),
            ("checksum", 5, "content", 16, 3),
        ]
        for changed in alternatives:
            self.assertNotEqual(sha256(baseline).digest(), sha256(_job_manifest(*changed)).digest())
        for name in ["MODEL_ID", "MODEL_REVISION", "PIPELINE_VERSION"]:
            with patch.object(moderation, name, "new value"):
                self.assertNotEqual(baseline, _job_manifest(*arguments))

    def test_checksum_uses_original_bytes_including_line_endings(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "input.jsonl"
            _ = source.write_bytes(b'{"content": "x"}\r\n{"content": "y"}')
            shards, checksum, total = _split_dataset(source, directory, "content", 1)
            self.assertEqual(total, 2)
            self.assertEqual(checksum, sha256(source.read_bytes()).hexdigest())
            self.assertEqual(shards[0].read_bytes(), b'{"content": "x"}\n')

    def test_reused_progress_does_not_inflate_interval_throughput(self) -> None:
        with (
            patch.object(moderation, "monotonic", side_effect=[10, 12]),
            patch.object(moderation, "Thread"),
            patch("builtins.print") as output,
        ):
            with _report_progress({"new": 2}, 10, 30, reused_rows=5):
                pass
        self.assertIn("Rows: 5/10", cast(str, output.call_args_list[0].args[0]))
        self.assertIn("Rows: 7/10 | 1.0 rows/s", cast(str, output.call_args_list[1].args[0]))

    def test_same_job_calls_are_serialized_locally(self) -> None:
        attempted, acquired = Event(), Event()

        def second_caller() -> None:
            attempted.set()
            with moderation._job_lock("serialization-test"):
                acquired.set()

        with TemporaryDirectory() as temp, patch.object(moderation, "gettempdir", return_value=temp):
            with moderation._job_lock("serialization-test"):
                thread = Thread(target=second_caller)
                thread.start()
                self.assertTrue(attempted.wait(1))
                self.assertFalse(acquired.wait(0.05))
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertTrue(acquired.is_set())

    def test_worker_commits_scores_before_completion_marker(self) -> None:
        # Exercise the worker body without Modal containers or model loading.
        worker = moderation._predict_file
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "input.jsonl"
            _ = source.write_text('{"content": "x"}\n')
            destination = source.with_suffix(".scores.jsonl")
            marker = source.with_suffix(".complete.json")
            commits: list[bool] = []

            def commit() -> None:
                commits.append(marker.exists())
                result = cast(dict[str, object], json.loads(destination.read_text()))
                self.assertEqual(result["moderation"], {"OK": 1.0})

            volume = MagicMock()
            cast(MagicMock, volume.commit).side_effect = commit

            def predict(_texts: list[str], _batch_size: int) -> list[dict[str, float]]:
                return [{"OK": 1.0}]

            with patch.object(moderation, "data_volume", volume):
                _ = worker("input.jsonl", "content", 16, {}, predict, data_root=directory)
            self.assertEqual(commits, [False, True])
            self.assertEqual(json.loads(marker.read_text()), {"rows": 1})
            marker.unlink()
            cast(MagicMock, volume.commit).side_effect = RuntimeError("commit failed")
            with patch.object(moderation, "data_volume", volume):
                with self.assertRaisesRegex(RuntimeError, "commit failed"):
                    _ = worker("input.jsonl", "content", 16, {}, predict, data_root=directory)
            self.assertFalse(marker.exists())
