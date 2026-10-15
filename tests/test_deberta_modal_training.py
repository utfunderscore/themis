import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from themis.models.deberta import modal_training


class ModalTrainingTests(unittest.TestCase):
    def test_missing_split_fails_before_starting_modal(self):
        with TemporaryDirectory() as temp, patch.object(modal_training.app, "run") as run:
            with self.assertRaises(FileNotFoundError):
                modal_training.train(temp)
            run.assert_not_called()

    def test_uploads_splits_and_passes_config_to_remote_worker(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            for split in ("train", "validation", "test"):
                (source / f"{split}.parquet").touch()
            with (
                patch.object(modal_training.app, "run"),
                patch.object(modal_training.modal, "enable_output"),
                patch.object(modal_training.volume, "batch_upload") as upload,
                patch.object(modal_training.train_remote, "remote", return_value={"metrics": {}}) as remote,
            ):
                result = modal_training.train(source, config={"bf16": True})
            run_id, config = remote.call_args.args
            self.assertEqual(config, {"bf16": True})
            self.assertEqual(result, {"metrics": {}})
            calls = upload.return_value.__enter__.return_value.put_file.call_args_list
            self.assertEqual(len(calls), 3)
            for call, split in zip(calls, ("train", "validation", "test"), strict=True):
                self.assertEqual(call.args, (
                    source / f"{split}.parquet", f"/runs/{run_id}/splits/{split}.parquet",
                ))
