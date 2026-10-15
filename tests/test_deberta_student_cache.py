import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from distill_jev.models.deberta.workflow import download_student


class StudentCacheTests(unittest.TestCase):
    def test_download_and_reuse_exact_run(self):
        receipt = {"run_id": "a" * 32, "volume_name": "training",
                   "output_directory": "runs/a/output", "model_directory": "runs/a/output/model"}
        files = {"runs/a/output/model/config.json": b"{}",
                 "runs/a/output/model/model.safetensors": b"weights",
                 "runs/a/output/run_config.json": b'{"max_length": 256}'}
        with TemporaryDirectory() as temporary, patch("modal.Volume.from_name") as factory:
            volume = factory.return_value
            volume.iterdir.return_value = [
                SimpleNamespace(path=path, type=SimpleNamespace(name="FILE"))
                for path in files if "/model/" in path
            ]
            volume.read_file.side_effect = lambda path: iter([files[path]])
            model = download_student(receipt, temporary)
            self.assertEqual((model / "model.safetensors").read_bytes(), b"weights")
            self.assertEqual(json.loads((model.parent / "run_config.json").read_text())["max_length"], 256)
            factory.reset_mock()
            self.assertEqual(download_student(receipt, temporary), model)
            factory.assert_not_called()
            with self.assertRaises(ValueError):
                download_student({**receipt, "volume_name": "different"}, temporary)

    def test_failed_download_is_not_cached(self):
        receipt = {"run_id": "b" * 32, "volume_name": "training",
                   "output_directory": "output", "model_directory": "output/model"}
        with TemporaryDirectory() as temporary, patch("modal.Volume.from_name") as factory:
            factory.return_value.iterdir.side_effect = RuntimeError("download failed")
            with self.assertRaises(RuntimeError):
                download_student(receipt, temporary)
            self.assertFalse((Path(temporary) / receipt["run_id"]).exists())
