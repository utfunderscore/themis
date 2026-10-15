"""Exercise interruption and resume with in-memory Modal substitutes."""

import json
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from themis.models.decision.perplexity import bulk


class MemoryVolume:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.uploads = []
        self.fail_download = False
        self.fail_commit = False

    def batch_upload(self, **kwargs):
        return nullcontext(self)

    def put_file(self, source, path):
        self.files[path] = source.read_bytes() if isinstance(source, Path) else source.read()
        self.uploads.append(path)

    def reload(self):
        for name, content in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("commit failed")
        for path in self.root.rglob("*"):
            if path.is_file():
                self.files[path.relative_to(self.root).as_posix()] = path.read_bytes()

    def read_file(self, path):
        if self.fail_download:
            raise RuntimeError("download failed")
        yield self.files[path]


class BulkTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(TemporaryDirectory()))
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.source, self.output = self.root / "input.jsonl", self.root / "output.jsonl"
        self.source.write_text("".join(json.dumps({
            "state": str(i), "question": {"type": "noul"},
            "source": {"content": str(i), "id": i, "moderation": {"OK": 0.5}},
        }) + "\n" for i in range(5)))
        self.output.write_text("previous result")
        self.volume = MemoryVolume(self.remote)
        self.completed = {}
        stack.enter_context(patch.object(bulk, "data_volume", self.volume))
        stack.enter_context(patch.object(bulk, "completed", self.completed))
        stack.enter_context(patch.object(bulk.modal, "enable_output", return_value=nullcontext()))
        stack.enter_context(patch.object(bulk, "log"))
        self.app = stack.enter_context(patch.object(bulk, "app"))
        self.app.run.return_value = nullcontext()
        self.download = stack.enter_context(patch.object(bulk, "download_model"))
        self.worker = stack.enter_context(patch.object(bulk, "Decisions"))
        self.worker.with_options.return_value.return_value.predict_file.map.side_effect = self.score
        self.processed = []
        self.predicted_rows = []
        self.fail_shard = None

    def score(self, paths, *, kwargs, order_outputs):
        self.assertFalse(order_outputs)
        for path in reversed(paths):
            self.processed.append(Path(path).name)
            if Path(path).name == self.fail_shard:
                raise RuntimeError("inference failed")
            yield bulk._predict_file(path, **kwargs, predict=self.predict, data_root=self.remote)

    def predict(self, rows, batch_size):
        self.predicted_rows.extend(json.loads(json.dumps(row)) for row in rows)
        return [{"answer": row["state"], "probabilities": [0.2, 0.8]} for row in rows]

    def run_job(self, **kwargs):
        return bulk.predict_dataset(self.source, self.output, rows_per_shard=2, **kwargs)

    def assert_results(self):
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["source"]["id"] for row in rows], list(range(5)))
        self.assertEqual([row["decision"]["answer"] for row in rows], list(map(str, range(5))))
        self.assertTrue(all(row["source"]["moderation"] == {"OK": 0.5} for row in rows))

    def test_failure_resumes_only_unfinished_shards_and_restores_order(self):
        self.fail_shard = "000001.jsonl"
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_job()
        self.assertEqual(self.output.read_text(), "previous result")
        self.assertEqual(len(self.completed), 1)
        self.processed.clear()
        self.volume.uploads.clear()
        self.fail_shard = None
        self.run_job(max_containers=2)
        self.assertEqual(self.processed, ["000001.jsonl", "000000.jsonl"])
        self.assertFalse(any(path.endswith("000002.jsonl") for path in self.volume.uploads))
        options = self.worker.with_options.call_args.kwargs
        self.assertEqual(options["max_containers"], 2)
        self.assertGreater(float(options["env"][bulk.RUN_STARTED_AT_ENV]), 0)
        self.assert_results()

    def test_passes_prepared_native_rows_directly_to_predict(self):
        self.run_job()
        self.assertEqual(
            sorted(self.predicted_rows, key=lambda row: row["source"]["id"]),
            [json.loads(line) for line in self.source.read_text().splitlines()],
        )

    def test_download_failure_resumes_without_gpu_or_upload(self):
        self.volume.fail_download = True
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            self.run_job()
        self.assertEqual(self.output.read_text(), "previous result")
        self.assertEqual(len(self.completed), 3)
        self.volume.fail_download = False
        self.volume.uploads.clear()
        self.app.reset_mock()
        self.download.reset_mock()
        self.worker.reset_mock()
        self.output = self.root / "new-output.jsonl"
        self.run_job()
        self.app.run.assert_not_called()
        self.download.remote.assert_not_called()
        self.worker.with_options.assert_not_called()
        self.assertFalse(self.volume.uploads)
        self.assert_results()

    def test_commit_failure_does_not_publish_completion(self):
        self.volume.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.run_job()
        self.assertFalse(self.completed)
        self.assertEqual(self.output.read_text(), "previous result")
        self.volume.fail_commit = False
        self.run_job()
        for saved in self.completed.values():
            self.assertIn(saved["result_path"], self.volume.files)
        self.assert_results()

    def test_worker_reuses_completed_entry_without_predicting(self):
        self.run_job()
        path, saved = next(iter(self.completed.items()))
        def fail(*args):
            raise AssertionError("already completed")
        self.assertEqual(bulk._predict_file(path, 32, fail, data_root=self.remote), saved)

    def test_corrupt_results_and_publication_failure_preserve_output(self):
        self.run_job()
        self.output.write_text("previous result")
        with patch.object(Path, "replace", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.run_job()
        self.assertEqual(self.output.read_text(), "previous result")
        saved = next(iter(self.completed.values()))
        self.volume.files[saved["result_path"]] = b"broken result\n"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_job()
        self.assertEqual(self.output.read_text(), "previous result")
        self.assertEqual(len(self.completed), 3)

    def test_empty_input_and_path_collision_do_not_start_remote_work(self):
        self.source.write_bytes(b"")
        self.run_job()
        self.assertEqual(self.output.read_bytes(), b"")
        self.app.run.assert_not_called()
        self.assertFalse(self.volume.files)
        self.output = self.source
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.run_job()
