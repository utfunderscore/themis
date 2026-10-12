"""Local inference checks; never invoke Modal."""

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from distill_jev.models.decision.perplexity import decisions


class DecisionTests(unittest.TestCase):
    def test_prediction_uses_optimized_worker_and_batch_ceiling(self):
        rows = [{"state": "message", "question": {"type": "noul"}}]
        with (patch.object(decisions.modal, "enable_output", return_value=nullcontext()),
              patch.object(decisions, "app") as app,
              patch.object(decisions, "download_model") as download,
              patch.object(decisions, "Decisions") as worker):
            app.run.return_value = nullcontext()
            worker.return_value.predict.remote.return_value = [{"answer": True}]
            self.assertEqual(decisions.predict_decisions(rows), [{"answer": True}])
            worker.return_value.predict.remote.assert_called_once_with(rows, batch_size=32)
            download.remote.assert_called_once_with()

    def test_invalid_work_sizes_fail_before_remote_run(self):
        with patch.object(decisions, "app") as app:
            for rows, size in (([], 32), ([{}] * 129, 32), ([{}], 0), ([{}], 33)):
                with self.assertRaises(ValueError):
                    decisions.predict_decisions(rows, batch_size=size)
            app.run.assert_not_called()

    def test_complete_checkpoint_skips_hub_and_commit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight").write_bytes(b"weights")
            (root / ".download-complete.json").write_text(json.dumps({
                "revision": decisions.MODEL_REVISION, "files": {"weight": 7},
            }))
            with (patch.object(decisions, "MODEL_DIR", directory),
                  patch.object(decisions, "volume") as volume,
                  patch.object(decisions, "import_module") as imports):
                self.assertEqual(decisions._ensure_checkpoint(), directory)
                imports.assert_not_called()
                volume.commit.assert_not_called()
                (root / "weight").write_bytes(b"bad")
                imports.side_effect = RuntimeError("downloader requested")
                with self.assertRaisesRegex(RuntimeError, "downloader requested"):
                    decisions._ensure_checkpoint()
